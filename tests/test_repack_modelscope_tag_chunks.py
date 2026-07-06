from scripts.dev.repack_modelscope_tag_chunks import (
    abnormal_dirs,
    build_output_chunks,
    complete_chunk_index_ranges,
    index_in_any_range,
    parse_range_dir,
    range_fully_covered,
    union_sample_indexes,
)


def test_parse_range_dir_counts_inclusive_samples():
    parsed = parse_range_dir("012480-015700", "tag/012480-015700")

    assert parsed is not None
    assert parsed.start == 12480
    assert parsed.end == 15700
    assert parsed.sample_count == 3221
    assert parsed.path == "tag/012480-015700"


def test_abnormal_dirs_selects_non_chunk_sized_ranges():
    remote_dirs = [
        parse_range_dir("102000-104999"),
        parse_range_dir("012480-015700"),
        parse_range_dir("175000-177646"),
    ]

    selected = abnormal_dirs([item for item in remote_dirs if item is not None], chunk_size=3000)

    assert [item.name for item in selected] == ["012480-015700", "175000-177646"]


def test_complete_chunk_ranges_filter_existing_samples():
    remote_dirs = [
        parse_range_dir("102000-104999"),
        parse_range_dir("105000-107999"),
        parse_range_dir("012480-015700"),
    ]

    ranges = complete_chunk_index_ranges([item for item in remote_dirs if item is not None], chunk_size=3000)

    assert index_in_any_range(102000, ranges)
    assert index_in_any_range(107999, ranges)
    assert not index_in_any_range(108000, ranges)
    assert not index_in_any_range(12480, ranges)


def test_range_fully_covered_detects_abnormal_duplicate_ranges():
    remote_dirs = [
        parse_range_dir("111000-113999"),
        parse_range_dir("114000-116999"),
        parse_range_dir("111000-116999"),
        parse_range_dir("117000-119500"),
    ]
    ranges = complete_chunk_index_ranges([item for item in remote_dirs if item is not None], chunk_size=3000)

    assert range_fully_covered(parse_range_dir("111000-116999"), ranges)
    assert not range_fully_covered(parse_range_dir("117000-119500"), ranges)


def test_build_output_chunks_splits_contiguous_runs_and_reports_leftovers():
    indexes = set(range(0, 6500)) | set(range(8000, 11000))

    chunks, leftovers = build_output_chunks(indexes, chunk_size=3000)

    assert [chunk.name for chunk in chunks] == ["000000-002999", "003000-005999", "008000-010999"]
    assert [(run[0], run[-1], len(run)) for run in leftovers] == [(6000, 6499, 500)]


def test_union_sample_indexes_uses_samples_from_any_resolution():
    sources = {
        "1024x1024": {1: object(), 3: object()},
        "768x1344": {2: object(), 3: object()},
    }

    assert union_sample_indexes(sources) == {1, 2, 3}
