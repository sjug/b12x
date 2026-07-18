from __future__ import annotations

import pytest

from b12x.moe.fused.w4a16.kernel import (
    W4A16HybridMappedGrid48Kernel,
    w4a16_hybrid_mapped_grid188_mapping_proof,
    w4a16_hybrid_mapped_grid188_task_map,
    w4a16_hybrid_mapped_grid48_mapping_proof,
    w4a16_hybrid_mapped_grid48_task_map,
)


def test_grid188_task_map_is_an_exact_partition() -> None:
    proof = w4a16_hybrid_mapped_grid188_mapping_proof()

    assert proof["grid_x"] == 188
    assert sorted(proof["fc1_tasks"]) == list(range(128))
    assert sorted(proof["fc2_tasks"]) == list(range(768))
    assert proof["fc1_per_cta_counts"] == (1,) * 128 + (0,) * 60
    assert proof["fc2_per_cta_counts"] == (5,) * 16 + (4,) * 172
    assert proof["fc1_idle_ctas"] == tuple(range(128, 188))


@pytest.mark.parametrize("task_count", [-1, 0, 127, 129, 767, 769])
def test_grid188_task_map_rejects_other_geometries(task_count: int) -> None:
    with pytest.raises(ValueError, match="exactly 128 or 768"):
        w4a16_hybrid_mapped_grid188_task_map(task_count)


def test_grid48_task_map_is_an_exact_partition() -> None:
    proof = w4a16_hybrid_mapped_grid48_mapping_proof()

    assert proof["grid_x"] == 48
    assert sorted(proof["fc1_tasks"]) == list(range(128))
    assert sorted(proof["fc2_tasks"]) == list(range(768))
    assert proof["fc1_per_cta_counts"] == (3,) * 32 + (2,) * 16
    assert proof["fc2_per_cta_counts"] == (16,) * 48
    assert proof["fc1_waves"] == 3
    assert proof["fc2_waves"] == 16
    assert proof["fc1_idle_ctas"] == ()


@pytest.mark.parametrize("task_count", [-1, 0, 127, 129, 767, 769])
def test_grid48_task_map_rejects_other_geometries(task_count: int) -> None:
    with pytest.raises(ValueError, match="exactly 128 or 768"):
        w4a16_hybrid_mapped_grid48_task_map(task_count)


def test_grid48_profile_is_exact_sm121_one_cta_per_sm() -> None:
    assert W4A16HybridMappedGrid48Kernel.TARGET_CAPABILITY == (12, 1)
    assert W4A16HybridMappedGrid48Kernel.TARGET_SMS == 48
    assert W4A16HybridMappedGrid48Kernel.GRID_X == 48
    assert W4A16HybridMappedGrid48Kernel.FC1_SCHEDULE == "32x3+16x2"
    assert W4A16HybridMappedGrid48Kernel.FC2_SCHEDULE == "48x16"
