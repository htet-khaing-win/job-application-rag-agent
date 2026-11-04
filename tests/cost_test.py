import datetime as dt
import threading

import pytest

from cost import PRICES, BudgetExceeded, SpendLedger, estimate_cost, estimate_run_cost
from runtime import Profile, preset_profile


def test_estimate_cost_known_unknown_and_local():
    # gpt-4o-mini: 0.15 in / 0.60 out per 1M tokens
    assert estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert estimate_cost("gpt-4o-mini-2024-07-18", 1000, 0) == estimate_cost("gpt-4o-mini", 1000, 0)
    assert estimate_cost("qwen2.5:7b", 10_000, 10_000) == 0.0
    assert estimate_cost("totally-unknown", 5, 5) == 0.0
    assert estimate_cost("gpt-4o", -5, -5) == 0.0


def test_longest_prefix_wins():
    assert "gpt-4o" in PRICES and "gpt-4o-mini" in PRICES
    assert estimate_cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_cost("gpt-4o", 1_000_000, 0) == pytest.approx(2.50)


def test_estimate_run_cost_scales_with_jd_and_skips_local():
    prof = Profile()
    small = estimate_run_cost(prof, 400)
    big = estimate_run_cost(prof, 40_000)
    assert big["total_usd"] > small["total_usd"] > 0
    assert big["total_tokens"] > small["total_tokens"]
    assert set(small["per_role"]) == {"jd_analyst", "writer", "critic", "editor", "verifier"}
    row = small["per_role"]["writer"]
    assert row["model"] == "gemini:gemini-2.5-flash" and {"in_tok", "out_tok", "usd"} <= set(row)

    local = estimate_run_cost(preset_profile("Fully local (Ollama)"), 4000)
    assert local["total_usd"] == 0.0 and local["total_tokens"] > 0


def test_ledger_add_total_and_check(tmp_path):
    led = SpendLedger(tmp_path)
    led.add("k", 1.25)
    led.add("k", 0.75)
    led.add("other", 9)
    assert led.month_total("k") == pytest.approx(2.0)
    led.check("k", None)
    led.check("k", 2.01)
    with pytest.raises(BudgetExceeded, match="budget"):
        led.check("k", 2.0)                       # boundary: spent >= cap is blocked
    led.add("k", 0)                               # non-positive adds are ignored
    led.add("k", -3)
    assert led.month_total("k") == pytest.approx(2.0)


def test_ledger_month_rollover(tmp_path):
    led = SpendLedger(tmp_path)
    jan = dt.datetime(2026, 1, 31, 23, 59, tzinfo=dt.timezone.utc)
    feb = dt.datetime(2026, 2, 1, 0, 1, tzinfo=dt.timezone.utc)
    led.add("k", 5, now=jan)
    assert led.month_total("k", now=jan) == 5 and led.month_total("k", now=feb) == 0
    led.check("k", 5.0, now=feb)                  # new month => budget available again
    with pytest.raises(BudgetExceeded):
        led.check("k", 5.0, now=jan)


def test_ledger_persists_and_survives_corruption(tmp_path):
    SpendLedger(tmp_path).add("k", 3)
    assert SpendLedger(tmp_path).month_total("k") == 3
    (tmp_path / "usage.json").write_text("{not json", encoding="utf-8")
    led = SpendLedger(tmp_path)
    assert led.month_total("k") == 0
    led.add("k", 1)                               # recovers by rewriting a valid file
    assert led.month_total("k") == 1


def test_ledger_is_thread_safe_across_instances(tmp_path):
    def work():
        led = SpendLedger(tmp_path)               # separate instances must share one lock
        for _ in range(25):
            led.add("k", 0.01)

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert SpendLedger(tmp_path).month_total("k") == pytest.approx(2.0, abs=1e-6)
    assert not list(tmp_path.glob("*.tmp"))       # no orphaned temp files
