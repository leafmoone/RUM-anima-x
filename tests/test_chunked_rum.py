import argparse

from scripts.dev.anima_rum_xpred_train import (
    ChunkPlan,
    chunk_train_steps,
    completed_chunk_ids,
    lr_scale_for_step,
    make_chunk_plan,
    planned_chunk_train_steps,
    resolve_chunk_student_init,
    should_run_train_sample,
)


def test_make_chunk_plan_splits_total_samples_from_start_index():
    plan = make_chunk_plan(start_index=10, total_samples=25, chunk_size=8, max_chunks=None)

    assert plan == [
        ChunkPlan(chunk_id=0, start_index=10, num_samples=8),
        ChunkPlan(chunk_id=1, start_index=18, num_samples=8),
        ChunkPlan(chunk_id=2, start_index=26, num_samples=8),
        ChunkPlan(chunk_id=3, start_index=34, num_samples=1),
    ]


def test_make_chunk_plan_honors_max_chunks():
    plan = make_chunk_plan(start_index=0, total_samples=100, chunk_size=16, max_chunks=2)

    assert plan == [
        ChunkPlan(chunk_id=0, start_index=0, num_samples=16),
        ChunkPlan(chunk_id=1, start_index=16, num_samples=16),
    ]


def test_completed_chunk_ids_reads_manifest():
    manifest = {
        "chunks": [
            {"chunk_id": 0, "status": "complete"},
            {"chunk_id": 1, "status": "cache_built"},
            {"chunk_id": 2, "status": "complete"},
        ]
    }

    assert completed_chunk_ids(manifest) == {0, 2}


def test_resolve_chunk_student_init_uses_previous_checkpoint_after_first_chunk():
    args = argparse.Namespace(student_init="/models/teacher.safetensors")

    assert resolve_chunk_student_init(args, previous_checkpoint=None) == "/models/teacher.safetensors"
    assert resolve_chunk_student_init(args, previous_checkpoint="/tmp/chunk-0000/xpred.safetensors") == "/tmp/chunk-0000/xpred.safetensors"


def test_chunk_train_steps_prefers_manifest_then_summary(tmp_path):
    chunk = {"train_steps": 17}
    assert chunk_train_steps(chunk, tmp_path / "missing") == 17

    output_dir = tmp_path / "chunk-0000"
    output_dir.mkdir()
    (output_dir / "train-summary.json").write_text('{"losses": [1.0, 0.5, 0.25]}', encoding="utf-8")

    assert chunk_train_steps({"output_dir": str(output_dir)}, tmp_path) == 3
    assert chunk_train_steps({}, tmp_path) == 0


def test_should_run_train_sample_uses_global_step_offset():
    args = argparse.Namespace(sample_every_steps=500, global_step_offset=1314)

    assert should_run_train_sample(args, 185) is False
    assert should_run_train_sample(args, 186) is True


def test_planned_chunk_train_steps_sums_auto_steps_across_chunks():
    train_args = argparse.Namespace(
        max_train_steps=None,
        train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=1.0,
    )
    chunk_args = argparse.Namespace(train_steps_per_chunk=None)
    plans = [
        ChunkPlan(chunk_id=0, start_index=0, num_samples=16),
        ChunkPlan(chunk_id=1, start_index=16, num_samples=9),
    ]

    assert planned_chunk_train_steps(train_args, chunk_args, plans) == 4


def test_lr_scale_can_use_total_step_across_chunks():
    args = argparse.Namespace(
        lr_scheduler="cosine",
        lr_warmup_steps=100,
        lr_cosine_min=0.1,
        lr_scheduler_total_steps=1000,
        resolved_max_train_steps=200,
        max_train_steps=None,
    )

    first_step_in_later_chunk = lr_scale_for_step(args, 401)
    restarted_chunk_step = lr_scale_for_step(args, 1)

    assert first_step_in_later_chunk < 1.0
    assert restarted_chunk_step == 0.01
