from __future__ import annotations

import json
from types import SimpleNamespace

import recagent_eval.baselines.current_v2b as module
from recagent_eval.baseline_eval import BASELINE_SCORERS
from recagent_eval.baselines.current_v2b import score_current_v2b
from recagent_eval.data import Movie, Rating, leakage_safe_ranking_split


def _movies() -> dict[int, Movie]:
    return {
        movie_id: Movie(movie_id, f"M{movie_id}", ("Drama",), 1990 + movie_id)
        for movie_id in range(1, 9)
    }


def _ratings() -> list[Rating]:
    return [
        Rating(user_id, movie_id, 5, movie_id * 10 + user_id)
        for user_id in range(1, 17)
        for movie_id in range(1, 9)
    ]


def test_current_v2b_is_registered_and_restricts_training_users(
    monkeypatch, tmp_path
) -> None:
    assert "current_v2b" in BASELINE_SCORERS
    movies = _movies()
    split = leakage_safe_ranking_split(_ratings())
    eligible = sorted(split.validation_targets)
    dev = eligible[6:10]
    eval_users = tuple(eligible[10:13])
    calls: dict[str, object] = {}
    real_mkdtemp = module.tempfile.mkdtemp

    def portable_mkdtemp(*, prefix, **kwargs):
        calls["tempdir_kwargs"] = kwargs
        return real_mkdtemp(prefix=prefix, dir=tmp_path)

    def fake_train(movies_, split_, semantic, config, **kwargs):
        del movies_, semantic, config
        calls["training_user_ids"] = kwargs["training_user_ids"]
        calls["eval_user_ids"] = kwargs["eval_user_ids"]
        model_path = kwargs["model_output"]
        evidence_path = kwargs["evidence_output"]
        bundle_path = kwargs["bundle_manifest_output"]
        latent_path = model_path.parent / "latent.npz"
        latent_path.write_bytes(b"latent")
        model_path.write_text("{}")
        bundle_path.write_text("{}")
        rows = []
        for uid in kwargs["eval_user_ids"]:
            rows.append(
                {
                    "user_id": uid,
                    "lambdamart_recall_at_10": 1.0,
                    "lambdamart_ndcg_at_10": 1.0,
                    "lambdamart_ranked_movie_ids": [split_.validation_targets[uid]],
                    "union_candidate_recall": 1.0,
                    "constraint_satisfied": True,
                    "latency_ms": 0.5,
                }
            )
        evidence_path.write_text(
            json.dumps({"per_user_rows": rows}, sort_keys=True) + "\n"
        )
        return {"model_checksum": "m" * 64}

    monkeypatch.setattr(
        module,
        "DenseSemanticRetriever",
        SimpleNamespace(load=lambda *args, **kwargs: SimpleNamespace()),
    )
    monkeypatch.setattr(module.tempfile, "mkdtemp", portable_mkdtemp)
    monkeypatch.setattr(module, "train_lambdamart_pipeline", fake_train)
    result = score_current_v2b(
        movies,
        split,
        eval_users,
        ledger={"cohorts": {"development": dev}},
        max_training_users=8,
    )
    training = list(calls["training_user_ids"])
    assert len(training) == 8
    assert set(training).isdisjoint(set(eval_users))
    assert calls["eval_user_ids"] == eval_users
    assert len(result["rows"]) == len(eval_users)
    assert all(row.recall_at_10 == 1.0 for row in result["rows"])
    assert result["model_fingerprint"] == "m" * 64
    assert calls["tempdir_kwargs"] == {}
