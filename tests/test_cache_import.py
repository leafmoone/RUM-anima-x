import io
import tarfile

from rum_xpred.cache_import import (
    import_cache_archives,
    is_safe_tar_member,
    sample_index_from_name,
    target_path_for_member,
)


def test_target_path_maps_sample_to_chunk_and_resolution(tmp_path):
    target = target_path_for_member(
        "832x1216/sample-095182.safetensors",
        dst_cache_root=tmp_path,
        start_index=69000,
        chunk_size=3000,
    )

    assert target == tmp_path / "chunk-0008" / "832x1216" / "sample-095182.safetensors"


def test_target_path_can_apply_chunk_offset(tmp_path):
    target = target_path_for_member(
        "832x1216/sample-000000.safetensors",
        dst_cache_root=tmp_path,
        start_index=0,
        chunk_size=3000,
        chunk_offset=14,
    )

    assert target == tmp_path / "chunk-0014" / "832x1216" / "sample-000000.safetensors"


def test_target_path_rejects_samples_before_start(tmp_path):
    assert (
        target_path_for_member(
            "1024x1024/sample-000001.safetensors",
            dst_cache_root=tmp_path,
            start_index=69000,
            chunk_size=3000,
        )
        is None
    )


def test_sample_index_from_name_accepts_bucketed_paths():
    assert sample_index_from_name("1152x896/sample-100212.safetensors") == 100212
    assert sample_index_from_name("not-a-sample.txt") is None


def test_is_safe_tar_member_rejects_traversal():
    assert is_safe_tar_member("1024x1024/sample-000001.safetensors")
    assert not is_safe_tar_member("../sample-000001.safetensors")
    assert not is_safe_tar_member("/tmp/sample-000001.safetensors")


def test_import_cache_archives_extracts_to_chunk_dirs(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "cache"
    src.mkdir()
    archive = src / "832x1216.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        data = b"cache-data"
        info = tarfile.TarInfo("832x1216/sample-095182.safetensors")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    stats = import_cache_archives(src, dst, start_index=69000, chunk_size=3000)

    assert stats.extracted == 1
    assert (dst / "chunk-0008" / "832x1216" / "sample-095182.safetensors").read_bytes() == b"cache-data"
