import argparse

from scripts.dev.anima_rum_xpred_train import (
    DEFAULT_CACHE_BUCKETS,
    CacheBucketBatchCursor,
    ChunkPlan,
    chunk_train_steps,
    completed_chunk_ids,
    final_optimizer_state_for_train_args,
    bucket_cache_path,
    choose_cache_bucket,
    collect_cache_buckets,
    lr_scale_for_step,
    make_chunk_plan,
    planned_chunk_train_steps,
    resolve_chunk_optimizer_state,
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


def test_resolve_chunk_optimizer_state_uses_previous_state_after_first_chunk():
    args = argparse.Namespace(optimizer_state=None)

    assert resolve_chunk_optimizer_state(args, previous_optimizer_state=None) is None
    assert resolve_chunk_optimizer_state(args, previous_optimizer_state="/tmp/chunk-0000/train-state.pt") == "/tmp/chunk-0000/train-state.pt"


def test_final_optimizer_state_path_uses_prediction_type(tmp_path):
    x_args = argparse.Namespace(output_dir=str(tmp_path), prediction_type="x")
    v_args = argparse.Namespace(output_dir=str(tmp_path), prediction_type="v")

    assert final_optimizer_state_for_train_args(x_args) == tmp_path / "xpred-train-state.pt"
    assert final_optimizer_state_for_train_args(v_args) == tmp_path / "vpred-train-state.pt"


def test_chunk_train_steps_prefers_manifest_then_summary(tmp_path):
    chunk = {"train_steps": 17}
    assert chunk_train_steps(chunk, tmp_path / "missing") == 17

    output_dir = tmp_path / "chunk-0000"
    output_dir.mkdir()
    (output_dir / "train-summary.json").write_text('{"losses": [1.0, 0.5, 0.25]}', encoding="utf-8")

    assert chunk_train_steps({"output_dir": str(output_dir)}, tmp_path) == 3
    assert chunk_train_steps({}, tmp_path) == 0


def test_chunk_train_steps_can_recover_from_total_completed_steps():
    previous = {"total_completed_steps": 100}
    current = {"total_completed_steps": 117}

    assert chunk_train_steps(current, None, previous_chunk=previous) == 17


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


def test_choose_cache_bucket_is_deterministic_and_uses_fixed_buckets():
    bucket_a = choose_cache_bucket(sample_index=123, seed=7)
    bucket_b = choose_cache_bucket(sample_index=123, seed=7)

    assert bucket_a == bucket_b
    assert bucket_a in DEFAULT_CACHE_BUCKETS


def test_bucket_cache_path_puts_bucketed_samples_in_resolution_dirs(tmp_path):
    flat = bucket_cache_path(tmp_path, 12, width=1024, height=1024, bucket_enabled=False)
    bucketed = bucket_cache_path(tmp_path, 12, width=832, height=1216, bucket_enabled=True)

    assert flat == tmp_path / "sample-000012.safetensors"
    assert bucketed == tmp_path / "832x1216" / "sample-000012.safetensors"


def test_collect_cache_buckets_separates_resolution_dirs(tmp_path):
    root_file = tmp_path / "sample-000000.safetensors"
    root_file.write_text("root", encoding="utf-8")
    bucket_dir = tmp_path / "832x1216"
    bucket_dir.mkdir()
    bucket_file = bucket_dir / "sample-000001.safetensors"
    bucket_file.write_text("bucket", encoding="utf-8")

    buckets = collect_cache_buckets(tmp_path)

    assert [bucket.name for bucket in buckets] == ["root", "832x1216"]
    assert buckets[0].files == [root_file]
    assert buckets[1].files == [bucket_file]


def test_cache_bucket_cursor_returns_one_bucket_per_batch(tmp_path):
    a = tmp_path / "1024x1024"
    b = tmp_path / "832x1216"
    a.mkdir()
    b.mkdir()
    for index in range(2):
        (a / f"sample-{index:06d}.safetensors").write_text("a", encoding="utf-8")
        (b / f"sample-{index + 2:06d}.safetensors").write_text("b", encoding="utf-8")
    buckets = collect_cache_buckets(tmp_path)
    cursor = CacheBucketBatchCursor(buckets, batch_size=2, shuffle=False, seed=1, drop_last=False)

    batch = cursor.next()

    assert len(batch) == 2
    assert {path.parent.name for path in batch} in [{"1024x1024"}, {"832x1216"}]
