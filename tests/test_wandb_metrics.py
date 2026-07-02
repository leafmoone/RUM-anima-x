import json

from scripts.dev.anima_rum_xpred_train import read_wandb_external_metrics


def test_read_wandb_external_metrics_maps_fid_and_inception_score(tmp_path):
    path = tmp_path / "eval-metrics.json"
    path.write_text(
        json.dumps(
            {
                "fid": 12.3,
                "is": 5.6,
                "clip_score": {"value": 0.31},
                "ignored": "not numeric",
            }
        ),
        encoding="utf-8",
    )

    metrics = read_wandb_external_metrics(str(path))

    assert metrics["eval/fid"] == 12.3
    assert metrics["eval/is"] == 5.6
    assert metrics["eval/clip_score"] == 0.31
    assert "eval/ignored" not in metrics


def test_read_wandb_external_metrics_ignores_missing_file(tmp_path):
    assert read_wandb_external_metrics(str(tmp_path / "missing.json")) == {}
