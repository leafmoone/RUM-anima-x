import pytest

from scripts.dev.upload_completed_cache_chunks import (
    process_model_train_chunks,
    remote_cache_path,
    remote_model_train_path,
)


def test_remote_cache_path_defaults_to_tag_prefix():
    assert remote_cache_path("69000-71999", "1024x1024", "tag") == "tag/69000-71999/1024x1024/1024x1024.tar"


def test_remote_cache_path_can_disable_prefix():
    assert remote_cache_path("69000-71999", "1024x1024", "") == "69000-71999/1024x1024/1024x1024.tar"


def test_remote_model_train_path_preserves_chunk_directory():
    assert remote_model_train_path("chunk-0031", "train") == "train/chunk-0031"


def test_process_model_train_chunks_preserves_last_name_sorted_chunk(tmp_path):
    train_dir = tmp_path / "train"
    for chunk_name in ["chunk-0031", "chunk-0033", "chunk-0032"]:
        chunk_dir = train_dir / chunk_name
        chunk_dir.mkdir(parents=True)
        (chunk_dir / "xpred-adapter-checkpoint.safetensors").write_text("checkpoint", encoding="utf-8")
        (chunk_dir / "train-summary.json").write_text("{}", encoding="utf-8")

    uploaded = []

    def fake_upload(local_path, remote_path, repo_id, repo_type, token, max_upload_workers):
        uploaded.append((local_path.name, remote_path, repo_id, repo_type, token, max_upload_workers))

    result = process_model_train_chunks(
        model_train_dir=train_dir,
        repo_id="leafmoone/anima-x-test",
        token="token",
        upload_func=fake_upload,
        delete_model_after_upload=True,
        preserve_last_model_chunk=True,
    )

    assert result.uploaded_model_chunks == ["chunk-0031", "chunk-0032"]
    assert result.skipped_model_chunks == ["chunk-0033"]
    assert uploaded == [
        ("chunk-0031", "train/chunk-0031", "leafmoone/anima-x-test", "model", "token", None),
        ("chunk-0032", "train/chunk-0032", "leafmoone/anima-x-test", "model", "token", None),
    ]
    assert not (train_dir / "chunk-0031").exists()
    assert not (train_dir / "chunk-0032").exists()
    assert (train_dir / "chunk-0033").exists()


def test_process_model_train_chunks_deletes_only_after_success(tmp_path):
    train_dir = tmp_path / "train"
    chunk_dir = train_dir / "chunk-0031"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "xpred-adapter-checkpoint.safetensors").write_text("checkpoint", encoding="utf-8")
    (chunk_dir / "train-summary.json").write_text("{}", encoding="utf-8")

    def failing_upload(local_path, remote_path, repo_id, repo_type, token, max_upload_workers):
        raise RuntimeError("upload failed")

    with pytest.raises(RuntimeError, match="upload failed"):
        process_model_train_chunks(
            model_train_dir=train_dir,
            repo_id="leafmoone/anima-x-test",
            token="token",
            upload_func=failing_upload,
            delete_model_after_upload=True,
            preserve_last_model_chunk=False,
        )

    assert chunk_dir.exists()


def test_process_model_train_chunks_skips_incomplete_directories(tmp_path):
    train_dir = tmp_path / "train"
    incomplete_dir = train_dir / "chunk-0031"
    incomplete_dir.mkdir(parents=True)
    (incomplete_dir / "train-summary.json").write_text("{}", encoding="utf-8")

    uploaded = []

    result = process_model_train_chunks(
        model_train_dir=train_dir,
        repo_id="leafmoone/anima-x-test",
        token="token",
        upload_func=lambda *args: uploaded.append(args),
        delete_model_after_upload=True,
        preserve_last_model_chunk=False,
    )

    assert result.uploaded_model_chunks == []
    assert result.skipped_model_chunks == ["chunk-0031"]
    assert uploaded == []
    assert incomplete_dir.exists()
