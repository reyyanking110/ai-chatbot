from __future__ import annotations

from jarvis_local.hardware import ModelPlan, card_for, detect_hardware, recommend_models


def test_detect_hardware_returns_profile():
    profile = detect_hardware()
    assert profile.ram_gb > 0
    assert profile.cpu_cores >= 1
    assert profile.budget_gb() > 0
    assert profile.accelerator in {"cpu", "cuda", "metal", "rocm", "vulkan"}
    assert "RAM" in profile.summary()


def test_budget_scales_with_ram():
    profile = detect_hardware()
    assert profile.budget_gb() < profile.ram_gb
    assert profile.budget_gb(32) == 32


def test_recommendations_monotonic():
    small = recommend_models(3)
    medium = recommend_models(8)
    large = recommend_models(20)
    assert card_for(small.general).size_gb <= card_for(medium.general).size_gb
    assert card_for(medium.general).size_gb <= card_for(large.general).size_gb
    assert large.general != small.general
    assert large.num_ctx >= small.num_ctx


def test_small_budget_still_yields_a_plan():
    plan = recommend_models(0.1)
    assert isinstance(plan, ModelPlan)
    assert plan.general and plan.coding and plan.fast and plan.embed
    assert plan.notes


def test_all_catalog_entries_have_valid_roles():
    from jarvis_local.hardware import CATALOG

    for card in CATALOG:
        assert card.size_gb > 0
        assert card.roles
        assert set(card.roles) <= {"general", "coding", "fast", "embed", "vision"}


def test_plan_download_total():
    plan = recommend_models(16)
    assert plan.download_gb() > 0
    assert plan.download_gb() < 40
