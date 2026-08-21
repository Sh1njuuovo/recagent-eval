from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, model_validator

from recagent_eval.candidate_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_FINGERPRINT,
    FEATURE_SCHEMA_VERSION,
    build_candidate_feature_rows,
)
from recagent_eval.data import Movie, Rating
from recagent_eval.models import PreferenceState, RecommendedMovie, ScoreBreakdown

ARTIFACT_SCHEMA_VERSION = "lambdamart-artifact/v1"
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
DEFAULT_PARAMETER_GRID: tuple[dict[str, int | float], ...] = tuple(
    {
        "num_leaves": leaves,
        "learning_rate": rate,
        "n_estimators": estimators,
        "min_child_samples": children,
    }
    for leaves in (15, 31)
    for rate in (0.03, 0.05)
    for estimators in (100, 200)
    for children in (20, 50)
)


class RankerEstimator(Protocol):
    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
        *,
        group: Sequence[int],
    ) -> Any: ...

    def predict(
        self,
        features: Sequence[Sequence[float]],
        *,
        pred_contrib: bool = False,
    ) -> Sequence[Any]: ...


@dataclass(frozen=True)
class CandidateQuery:
    user_id: int
    target_movie_id: int
    features_by_movie: Mapping[int, tuple[float, ...]]


@dataclass(frozen=True)
class TrainingMatrix:
    features: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    groups: tuple[int, ...]
    user_ids: tuple[int, ...]
    movie_ids: tuple[int, ...]
    evaluation_users: int
    training_users: int


def build_training_matrix(
    queries: Sequence[CandidateQuery],
    *,
    max_negatives: int | None = None,
) -> TrainingMatrix:
    if max_negatives is not None and max_negatives < 0:
        raise ValueError("max_negatives must be non-negative")
    features: list[tuple[float, ...]] = []
    labels: list[int] = []
    groups: list[int] = []
    users: list[int] = []
    movies: list[int] = []
    for query in sorted(queries, key=lambda item: item.user_id):
        if query.target_movie_id not in query.features_by_movie:
            continue
        negatives = sorted(
            movie_id for movie_id in query.features_by_movie if movie_id != query.target_movie_id
        )
        if max_negatives is not None:
            negatives = negatives[:max_negatives]
        ordered_ids = [query.target_movie_id, *negatives]
        for movie_id in ordered_ids:
            row = tuple(float(value) for value in query.features_by_movie[movie_id])
            _validate_row(row, user_id=query.user_id, movie_id=movie_id)
            features.append(row)
            labels.append(int(movie_id == query.target_movie_id))
            users.append(query.user_id)
            movies.append(movie_id)
        groups.append(len(ordered_ids))
    return TrainingMatrix(
        features=tuple(features),
        labels=tuple(labels),
        groups=tuple(groups),
        user_ids=tuple(users),
        movie_ids=tuple(movies),
        evaluation_users=len({query.user_id for query in queries}),
        training_users=len(groups),
    )


def make_lgbm_ranker(params: Mapping[str, int | float], *, seed: int = 42) -> Any:
    try:
        from lightgbm import LGBMRanker
    except ImportError as exc:  # pragma: no cover - optional install path
        raise RuntimeError("LambdaMART requires the optional 'ml' dependencies") from exc
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        **dict(params),
    )


class LearnedRanker:
    kind = "lambdamart"

    def __init__(
        self,
        estimator: RankerEstimator,
        *,
        legal_train_rows: Sequence[Rating] = (),
    ):
        self.estimator = estimator
        self.legal_train_rows = tuple(legal_train_rows)

    def fit(self, matrix: TrainingMatrix) -> LearnedRanker:
        if not matrix.groups:
            raise ValueError("no trainable query groups; targets were absent from candidate unions")
        self.estimator.fit(
            list(matrix.features),
            list(matrix.labels),
            group=list(matrix.groups),
        )
        return self

    def rank(
        self,
        movies: Mapping[int, Movie],
        *,
        itemcf_scores: Mapping[int, float],
        semantic_scores: Mapping[int, float],
        state: PreferenceState,
        top_k: int = 10,
    ) -> list[RecommendedMovie]:
        history = tuple(
            Rating(0, movie_id, 5, index)
            for index, movie_id in enumerate(sorted(state.liked_movie_ids))
        )
        rows = build_candidate_feature_rows(
            user_id=0,
            movies=movies,
            itemcf_scores=itemcf_scores,
            dense_scores=semantic_scores,
            history=history,
            train_rows=self.legal_train_rows,
            state=state,
        )
        return self.rank_feature_rows(
            movies,
            {row.movie_id: row.values for row in rows},
            top_k=top_k,
        )

    def rank_feature_rows(
        self,
        movies: Mapping[int, Movie],
        features_by_movie: Mapping[int, tuple[float, ...]],
        *,
        top_k: int = 10,
    ) -> list[RecommendedMovie]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        ids = sorted(movie_id for movie_id in features_by_movie if movie_id in movies)
        if not ids or top_k == 0:
            return []
        rows = [features_by_movie[movie_id] for movie_id in ids]
        for movie_id, row in zip(ids, rows, strict=True):
            _validate_row(row, user_id=None, movie_id=movie_id)
        predictions = [float(value) for value in self.estimator.predict(rows)]
        contributions = self.estimator.predict(rows, pred_contrib=True)
        if len(predictions) != len(ids) or len(contributions) != len(ids):
            raise ValueError("LambdaMART returned an unexpected prediction shape")
        ranked: list[RecommendedMovie] = []
        for movie_id, score, raw_contrib in zip(ids, predictions, contributions, strict=True):
            if not math.isfinite(score):
                raise ValueError(f"ranker prediction must be finite: movie={movie_id}")
            values = [float(value) for value in raw_contrib]
            if len(values) != len(FEATURE_NAMES) + 1:
                raise ValueError("prediction contributions do not match feature schema")
            contribution_map = dict(zip((*FEATURE_NAMES, "bias"), values, strict=True))
            if not math.isclose(sum(values), score, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(
                    f"feature contributions do not reconcile with prediction for movie={movie_id}"
                )
            movie = movies[movie_id]
            ranked.append(
                RecommendedMovie(
                    movie_id=movie_id,
                    title=movie.title,
                    genres=movie.genres,
                    year=movie.year,
                    score=ScoreBreakdown(
                        final=score,
                        feature_contributions=contribution_map,
                    ),
                )
            )
        return sorted(ranked, key=lambda item: (-item.score.final, item.movie_id))[:top_k]


class RankerArtifact(BaseModel):
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    kind: str = "lambdamart"
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_fingerprint: str = FEATURE_SCHEMA_FINGERPRINT
    selected_params: dict[str, int | float]
    dataset_fingerprint: str
    training_user_count: int = Field(ge=0)
    training_group_count: int = Field(ge=0)
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    model_string: str
    model_checksum: str = ""

    @model_validator(mode="after")
    def validate_contract(self) -> RankerArtifact:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION or self.kind != "lambdamart":
            raise ValueError("unsupported LambdaMART artifact schema or kind")
        if (
            self.feature_schema_version != FEATURE_SCHEMA_VERSION
            or self.feature_names != FEATURE_NAMES
            or self.feature_fingerprint != FEATURE_SCHEMA_FINGERPRINT
        ):
            raise ValueError("ranker artifact feature schema mismatch")
        expected = hashlib.sha256(self.model_string.encode()).hexdigest()
        if self.model_checksum and self.model_checksum != expected:
            raise ValueError("ranker artifact model checksum mismatch")
        self.model_checksum = expected
        return self


def artifact_from_estimator(
    estimator: Any,
    *,
    selected_params: Mapping[str, int | float],
    dataset_fingerprint: str,
    training_user_count: int,
    training_group_count: int,
) -> RankerArtifact:
    booster = getattr(estimator, "booster_", None)
    if booster is None or not hasattr(booster, "model_to_string"):
        raise ValueError("fitted LightGBM estimator does not expose a booster model")
    versions = {
        package: _dependency_version(package) for package in ("lightgbm", "numpy", "scikit-learn")
    }
    return RankerArtifact(
        selected_params=dict(selected_params),
        dataset_fingerprint=dataset_fingerprint,
        training_user_count=training_user_count,
        training_group_count=training_group_count,
        dependency_versions=versions,
        model_string=str(booster.model_to_string()),
    )


def estimator_from_artifact(artifact: RankerArtifact) -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("loading LambdaMART requires the optional 'ml' dependencies") from exc
    return _BoosterEstimator(lgb.Booster(model_str=artifact.model_string))


class _BoosterEstimator:
    def __init__(self, booster: Any):
        self.booster = booster

    def predict(self, features: Sequence[Sequence[float]], *, pred_contrib: bool = False) -> Any:
        return self.booster.predict(features, pred_contrib=pred_contrib)


def save_ranker_artifact(artifact: RankerArtifact, path: Path) -> None:
    payload = artifact.model_dump_json(indent=2).encode()
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise ValueError("ranker artifact exceeds maximum size")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def load_ranker_artifact(
    path: Path,
    *,
    expected_dataset_fingerprint: str | None = None,
) -> RankerArtifact:
    try:
        size = path.stat().st_size
    except FileNotFoundError as exc:
        raise ValueError(f"LambdaMART artifact is missing: {path}") from exc
    if size > MAX_ARTIFACT_BYTES:
        raise ValueError("ranker artifact exceeds maximum size")
    try:
        payload = json.loads(path.read_bytes())
        required = {
            "schema_version",
            "kind",
            "feature_schema_version",
            "feature_names",
            "feature_fingerprint",
            "selected_params",
            "dataset_fingerprint",
            "training_user_count",
            "training_group_count",
            "dependency_versions",
            "created_at",
            "model_string",
            "model_checksum",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        artifact = RankerArtifact.model_validate(payload)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid LambdaMART artifact: {exc}") from exc
    if (
        expected_dataset_fingerprint is not None
        and artifact.dataset_fingerprint != expected_dataset_fingerprint
    ):
        raise ValueError(
            "ranker artifact dataset fingerprint mismatch: "
            f"expected={expected_dataset_fingerprint}, actual={artifact.dataset_fingerprint}"
        )
    stored_lightgbm = artifact.dependency_versions.get("lightgbm")
    runtime_lightgbm = _dependency_version("lightgbm")
    if (
        stored_lightgbm not in {None, "missing", "test"}
        and runtime_lightgbm != "missing"
        and stored_lightgbm.split(".", 1)[0] != runtime_lightgbm.split(".", 1)[0]
    ):
        raise ValueError(
            "ranker artifact LightGBM version is incompatible: "
            f"artifact={stored_lightgbm}, runtime={runtime_lightgbm}"
        )
    return artifact


def _validate_row(row: Sequence[float], *, user_id: int | None, movie_id: int) -> None:
    if len(row) != len(FEATURE_NAMES):
        raise ValueError(
            f"feature count mismatch for user={user_id}, movie={movie_id}: "
            f"expected={len(FEATURE_NAMES)}, actual={len(row)}"
        )
    for name, value in zip(FEATURE_NAMES, row, strict=True):
        if not math.isfinite(value):
            raise ValueError(
                "candidate feature must be finite: "
                f"user={user_id}, movie={movie_id}, feature={name}, value={value!r}"
            )


def _dependency_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"
