from __future__ import annotations

from recagent_eval.ranker_diagnostics import (
    DiagnosticUserRow,
    aggregate_diagnostics,
)


def _row(
    *,
    in_union: bool,
    in_itemcf: bool = False,
    in_dense: bool = False,
    itemcf_top10: bool = False,
    lambdamart_top10: bool = False,
    itemcf_ndcg: float = 0.0,
    lambdamart_ndcg: float = 0.0,
    target_features: tuple[float, ...] | None = None,
    negative_features: tuple[float, ...] | None = None,
) -> DiagnosticUserRow:
    return DiagnosticUserRow(
        user_id=1,
        in_union=in_union,
        in_itemcf=in_itemcf,
        in_dense=in_dense,
        itemcf_top10=itemcf_top10,
        lambdamart_top10=lambdamart_top10,
        itemcf_ndcg_at_10=itemcf_ndcg,
        lambdamart_ndcg_at_10=lambdamart_ndcg,
        target_itemcf_rank=1 if in_itemcf else None,
        target_lambdamart_rank=1 if in_union else None,
        target_features=target_features,
        negative_mean_features=negative_features,
    )


def test_aggregate_diagnostics_keeps_unrestricted_denominators() -> None:
    rows = [
        _row(
            in_union=True,
            in_itemcf=True,
            itemcf_top10=True,
            itemcf_ndcg=1.0,
            lambdamart_top10=False,
        ),
        _row(in_union=False),
    ]
    summary = aggregate_diagnostics(rows)
    assert summary.user_count == 2
    assert summary.union_recall == 0.5
    assert summary.itemcf_top10_hit == 0.5
    assert summary.lambdamart_top10_hit == 0.0
    assert summary.present_user_count == 1
    assert summary.itemcf_ndcg_at_10_present == 1.0
    assert summary.lambdamart_ndcg_at_10_present == 0.0


def test_aggregate_diagnostics_separation_excludes_target_missing_users() -> None:
    rows = [
        _row(
            in_union=True,
            target_features=(0.8,) + (0.0,) * 9,
            negative_features=(0.4,) + (0.0,) * 9,
        ),
        _row(
            in_union=False,
            target_features=(0.9,) + (0.0,) * 9,
            negative_features=(0.1,) + (0.0,) * 9,
        ),
    ]
    summary = aggregate_diagnostics(rows)
    assert summary.feature_separation["itemcf_score"] == 0.4
    assert summary.feature_separation["dense_score"] == 0.0


def test_aggregate_diagnostics_rank_quantiles_use_present_users() -> None:
    rows = [
        _row(in_union=True, in_itemcf=True),
        _row(in_union=False),
        _row(in_union=True, in_itemcf=True),
    ]
    summary = aggregate_diagnostics(rows)
    assert summary.target_itemcf_rank_quantiles["p50"] == 1.0
    assert summary.target_lambdamart_rank_quantiles["p50"] == 1.0


def test_diagnose_ranker_cli_refuses_overwrite(tmp_path, monkeypatch) -> None:
    from recagent_eval.cli import app

    config = tmp_path / "config.yaml"
    config.write_text(
        "name: diagnose\nretrieval_top_k: 8\nsemantic_profile_history_cap: 2\n"
        "semantic:\n  kind: tfidf\n"
    )
    output = tmp_path / "diagnostics.json"
    output.write_text("existing")
    result = app
    from typer.testing import CliRunner

    runner = CliRunner()
    response = runner.invoke(
        result,
        [
            "diagnose-ranker",
            "--config",
            str(config),
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    assert response.exit_code != 0
    assert "refusing to overwrite" in response.output
    assert output.read_text() == "existing"
