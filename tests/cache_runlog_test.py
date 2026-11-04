import asyncio
from types import SimpleNamespace

import pytest

import node
import runlog
from cache import TTLCache, cached_embed_query, get_cached_research, make_key, put_cached_research
from state import GraphState


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBAGENT_HOME", str(tmp_path))
    return tmp_path


# --- TTLCache -------------------------------------------------------------------
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_ttl_expiry_and_hit_miss_counters():
    clk = Clock()
    c = TTLCache(ttl_s=10, clock=clk)
    c.set("a", 1)
    assert c.get("a") == 1
    clk.t += 11
    assert c.get("a", "gone") == "gone"
    assert (c.hits, c.misses) == (1, 1)


def test_lru_evicts_least_recently_used():
    c = TTLCache(ttl_s=100, max_items=2)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")          # a is now most recent
    c.set("c", 3)       # evicts b
    assert c.get("a") == 1 and c.get("b") is None and c.get("c") == 3


def test_disk_persistence_roundtrip_and_expiry(tmp_path):
    clk = Clock()
    path = tmp_path / "c.json"
    c = TTLCache(ttl_s=10, persist_path=path, clock=clk)
    c.set("k1", {"v": 1})
    clk.t += 5
    c.set("k2", {"v": 2})
    clk.t += 6   # k1 (expires +10) is dead, k2 alive
    reloaded = TTLCache(ttl_s=10, persist_path=path, clock=clk)
    assert reloaded.get("k1") is None and reloaded.get("k2") == {"v": 2}


def test_corrupt_disk_cache_is_a_cold_cache(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not json", encoding="utf-8")
    assert len(TTLCache(persist_path=path)) == 0


def test_keys_are_sha256_and_not_raw_input(tmp_path):
    k = make_key("research", "acme corp")
    assert len(k) == 64 and "acme" not in k
    put_cached_research("Acme Corp", {"company_research": "public info"})
    disk = (tmp_path / "research_cache.json").read_text(encoding="utf-8")
    assert "acme corp" not in disk.lower().replace("public info", "")


# --- embeddings --------------------------------------------------------------------
def test_embed_query_cached_per_model_and_text():
    calls = []

    class Emb:
        def embed_query(self, text):
            calls.append(text)
            return (0.1, 0.2)

    e = Emb()
    assert cached_embed_query(e, "m1", "unique text A") == [0.1, 0.2]
    cached_embed_query(e, "m1", "unique text A")
    assert calls == ["unique text A"]
    cached_embed_query(e, "m2", "unique text A")      # other model: separate entry
    assert len(calls) == 2


# --- research cache ------------------------------------------------------------------
def test_company_names_are_normalised_and_secrets_scrubbed():
    put_cached_research("  ACME   Corp ", {"company_research": "see sk-abcdefghijklmnopqrstuvwx", "x": True})
    got = get_cached_research("acme corp")
    assert got and "sk-abcdef" not in got["company_research"] and got["x"] is True
    assert get_cached_research("") is None


class FakeTavily:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def search(self, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("503")
        return {"results": [{"title": "T", "content": "Acme mission text"}]}


def _research(monkeypatch, tavily):
    monkeypatch.setattr(node, "get_runtime", lambda: SimpleNamespace(tavily=lambda: tavily))
    st = GraphState(job_description="jd", company_name="Acme")
    return asyncio.run(node.research_company_node(st))


def test_research_hits_cache_second_time(monkeypatch):
    t = FakeTavily()
    first = _research(monkeypatch, t)
    second = _research(monkeypatch, t)
    assert t.calls == 1
    assert not first.get("company_research_cached") and second["company_research_cached"] is True
    assert second["company_research"] == first["company_research"]


def test_failed_research_is_not_cached(monkeypatch):
    bad = FakeTavily(fail=True)
    assert _research(monkeypatch, bad)["company_research_success"] is False
    good = FakeTavily()
    assert _research(monkeypatch, good)["company_research_success"] is True and good.calls == 1


# --- run log -----------------------------------------------------------------------------
def test_log_run_only_writes_allow_listed_fields(home):
    runlog.log_run({"run_id": "r1", "ok": True, "cover_letter": "SECRET LETTER", "job_description": "JD TEXT",
                    "resume_text": "RESUME", "company_name": "Acme", "cost_usd": 0.01})
    raw = (home / "runs.jsonl").read_text(encoding="utf-8")
    for leaked in ("SECRET LETTER", "JD TEXT", "RESUME", "Acme"):
        assert leaked not in raw
    row = runlog.read_runs()[0]
    assert row["run_id"] == "r1" and "ts" in row and "git_sha" in row


def test_log_run_redacts_secret_shaped_values():
    runlog.log_run({"run_id": "r2", "error_type": "bad sk-abcdefghijklmnopqrstuvwx", "models": {"w": "tvly-abcdefghijkl"}})
    raw = runlog.log_path().read_text(encoding="utf-8")
    assert "sk-abcdef" not in raw and "tvly-abc" not in raw


def test_read_runs_limit_and_torn_line():
    for i in range(5):
        runlog.log_run({"run_id": f"r{i}"})
    with open(runlog.log_path(), "a", encoding="utf-8") as f:
        f.write('{"run_id": "torn"\n')
    assert [r["run_id"] for r in runlog.read_runs(limit=2)] == ["r3", "r4"]
    assert len(runlog.read_runs()) == 5


def test_percentiles_nearest_rank():
    xs = list(range(1, 11))
    assert runlog.percentiles(xs, (50, 95, 100)) == {50: 5, 95: 10, 100: 10}
    assert runlog.percentiles([], (50,)) == {50: 0.0}
    assert runlog.percentiles([7], (50, 95)) == {50: 7, 95: 7}


def test_summarize_runs_per_node_and_totals():
    runs = [{"node_timings": [{"node": "A", "ms": ms}], "cost_usd": 0.5, "tokens_in": 10, "tokens_out": 2,
             "duration_ms": ms * 2} for ms in (10, 20, 30, 40)]
    s = runlog.summarize_runs(runs)
    assert s["runs"] == 4 and s["nodes"]["A"] == {"n": 4, "p50_ms": 20, "p95_ms": 40}
    assert s["cost_usd_total"] == 2.0 and s["tokens_in_total"] == 40 and s["tokens_out_total"] == 8
    assert s["run_ms"][95] == 80


def test_run_log_rotates_at_size_limit(monkeypatch, home):
    monkeypatch.setattr(runlog, "MAX_BYTES", 400)
    for i in range(30):
        runlog.log_run({"run_id": f"run{i:04d}", "models": {"w": "openai:gpt-4o-mini"}})
    assert (home / "runs.jsonl.1").exists()
    assert (home / "runs.jsonl").stat().st_size < 400 + 300
    ids = [r["run_id"] for r in runlog.read_runs()]
    assert ids and ids[-1] == "run0029"
