import datetime as dt
import os
import sqlite3
import threading
from pathlib import Path
from typing import List

import pytest

import history
from history import PAGE_SIZE, HistoryError, HistoryStore

FAKE_KEY = "sk-FAKEHISTORYKEYabcdefghijklmnop"
T0 = dt.datetime(2026, 11, 4, 3, 0, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "history.db"


@pytest.fixture
def store(db: Path):
    s = HistoryStore(db)
    yield s
    s.close()


def _add(s: HistoryStore, n: int = 1, start: int = 0) -> List[int]:
    """Add n entries one second apart (so 'newest first' is decided by timestamp, not just id)."""
    ids = []
    for i in range(start, start + n):
        s._now = lambda i=i: T0 + dt.timedelta(seconds=i)
        ids.append(s.add(f"Company {i}", f"Role {i}", f"JD text {i}", f"Letter body {i}", {"refinements": i}))
    return ids


def _raw_bytes(path: Path) -> bytes:
    """Bytes of the database and every sibling file SQLite may create next to it."""
    blob = b""
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = path.with_name(path.name + suffix)
        if p.exists():
            blob += p.read_bytes()
    return blob


# ------------------------------------------------------------------ basic behaviour
def test_add_and_list_newest_first(store):
    ids = _add(store, 3)
    entries = store.list()
    assert [e.id for e in entries] == list(reversed(ids))
    assert [e.company for e in entries] == ["Company 2", "Company 1", "Company 0"]
    e = entries[0]
    assert e.created_at == "2026-11-04T03:00:02Z" and e.job_title == "Role 2"
    assert e.jd == "JD text 2" and e.letter == "Letter body 2" and e.meta == {"refinements": 2}


def test_same_second_entries_still_sort_newest_first(store):
    store._now = lambda: T0
    a = store.add("A", "t", "jd", "letter a")
    b = store.add("B", "t", "jd", "letter b")
    c = store.add("C", "t", "jd", "letter c")
    assert [e.id for e in store.list()] == [c, b, a]


def test_paging_25_entries_gives_10_10_5(store):
    ids = _add(store, 25)
    pages = [store.list(PAGE_SIZE, off) for off in (0, 10, 20)]
    assert [len(p) for p in pages] == [10, 10, 5]
    assert [e.id for p in pages for e in p] == list(reversed(ids))      # contiguous, no overlap, nothing missing
    assert store.list(PAGE_SIZE, 30) == []


def test_limit_and_offset_are_clamped(store):
    _add(store, 55)
    assert len(store.list(limit=0)) == 1
    assert len(store.list(limit=-5)) == 1
    assert len(store.list(limit=1000)) == PAGE_SIZE * 5
    assert [e.id for e in store.list(limit=3, offset=-7)] == [e.id for e in store.list(limit=3, offset=0)]


def test_get_and_count(store):
    ids = _add(store, 4)
    assert store.count() == 4
    assert store.get(ids[2]).company == "Company 2"
    assert store.get(99999) is None


def test_delete_one_leaves_the_others(store):
    ids = _add(store, 3)
    assert store.delete(ids[1]) is True
    assert store.count() == 2 and store.get(ids[1]) is None
    assert [e.id for e in store.list()] == [ids[2], ids[0]]


def test_delete_unknown_id_returns_false(store):
    _add(store, 1)
    assert store.delete(424242) is False
    assert store.count() == 1


def test_delete_all(store):
    _add(store, 7)
    assert store.delete_all() == 7
    assert store.count() == 0 and store.list() == []
    assert store.delete_all() == 0
    store._now = lambda: T0
    assert store.add("After", "t", "jd", "still works") > 0


# ------------------------------------------------------------------ permanence
def test_deleted_text_is_gone_from_every_file_but_other_entries_remain(db):
    gone, kept = "MARKER-DELETED-7f3a9c41d2e8b6", "MARKER-KEPT-5b1e0d93a7c4f2"
    s = HistoryStore(db)
    victim = s.add(f"Co {gone}", f"Role {gone}", f"jd {gone} " * 40, f"letter {gone} " * 60, {"run_id": "r1"})
    s.add("Keeper Ltd", "Role", f"jd {kept}", f"letter {kept}")
    assert gone.encode() in _raw_bytes(db)                  # sanity: it really was written
    assert s.delete(victim) is True
    s.close()
    HistoryStore(db).close()                                # open and close again: nothing reintroduces it
    raw = _raw_bytes(db)
    assert gone.encode() not in raw                         # the deleted text exists nowhere on disk
    assert kept.encode() in raw                             # ...while the surviving entry is still there


def test_delete_all_erases_everything_from_the_file(db):
    s = HistoryStore(db)
    for i in range(5):
        s.add("Co", "Role", f"jd MARKER-ALL-{i}-9d8c7b6a5f" * 30, f"letter MARKER-ALL-{i}-9d8c7b6a5f" * 30)
    assert s.delete_all() == 5
    s.close()
    raw = _raw_bytes(db)
    assert all(f"MARKER-ALL-{i}-9d8c7b6a5f".encode() not in raw for i in range(5))


def test_no_wal_or_journal_files_are_left_behind(db):
    s = HistoryStore(db)
    ids = _add(s, 3)
    s.delete(ids[0])
    s.close()
    assert not db.with_name(db.name + "-wal").exists() and not db.with_name(db.name + "-journal").exists()


# ------------------------------------------------------------------ input hygiene
def test_api_keys_never_reach_the_database(db):
    s = HistoryStore(db)
    s.add(f"Acme {FAKE_KEY}", "t", f"jd mentions {FAKE_KEY} twice {FAKE_KEY}", f"letter with {FAKE_KEY}")
    s.close()
    raw = _raw_bytes(db)
    assert FAKE_KEY.encode() not in raw and b"FAKEHISTORYKEY" not in raw
    e = HistoryStore(db).list()[0]
    assert "[REDACTED]" in e.letter and "[REDACTED]" in e.jd


def test_a_key_straddling_the_length_cap_is_not_left_half_stored(store):
    letter = "a" * (history.MAX_LETTER - 5) + FAKE_KEY
    e = store.get(store.add("Acme", "t", "jd", letter))
    assert len(e.letter) <= history.MAX_LETTER
    assert "sk-" not in e.letter and "FAKE" not in e.letter


def test_sql_injection_looking_text_is_stored_literally(store):
    evil = "Robert'); DROP TABLE letters;--"
    eid = store.add(evil, "x' OR '1'='1", "jd'; DELETE FROM letters; --", "letter'); DROP TABLE letters;--")
    e = store.get(eid)
    assert e.company == evil and e.job_title == "x' OR '1'='1"
    assert e.jd.startswith("jd'; DELETE") and e.letter.endswith("--")
    assert store.count() == 1                                # the table is intact


def test_control_chars_and_bidi_are_stripped_and_caps_enforced(store):
    e = store.get(store.add("Ac\x00me‮\n  Corp\x07", "Sen\x0bior​Engineer", "jd\x00body‮text",
                            "Dear\x00team,‮\nthanks"))
    for field in (e.company, e.job_title, e.jd, e.letter):
        assert not any(ch in field for ch in "\x00\x07\x0b‮​")
    assert "\n" not in e.company and e.company == "Ac me Corp"
    assert "\n" in e.letter                                  # paragraphs are kept in letters

    big = store.get(store.add("C" * 500, "T" * 500, "J" * 50_000, "L" * 50_000))
    assert len(big.company) == history.MAX_COMPANY and len(big.job_title) == history.MAX_TITLE
    assert len(big.jd) == history.MAX_JD_CHARS and len(big.letter) == history.MAX_LETTER


@pytest.mark.parametrize("company,letter", [("", "ok"), ("   ", "ok"), ("\x00‮", "ok"),
                                            ("Acme", ""), ("Acme", "   \n "), ("Acme", "\x00")])
def test_empty_company_or_letter_is_rejected(store, company, letter):
    with pytest.raises(ValueError):
        store.add(company, "t", "jd", letter)
    assert store.count() == 0


def test_non_string_input_is_rejected(store):
    with pytest.raises(TypeError):
        store.add("Acme", "t", "jd", 123)  # type: ignore[arg-type]


def test_meta_keeps_only_known_keys_with_the_right_types(store):
    meta = {"retrieval_score": 87, "refinements": 2, "verified": True, "cost_usd": 0.0123, "run_id": "abc123def456",
            "api_key": FAKE_KEY, "evil": "x" * 5000, "letter": "stolen"}
    e = store.get(store.add("Acme", "t", "jd", "letter", meta))
    assert e.meta == {"retrieval_score": 87.0, "refinements": 2, "verified": True, "cost_usd": 0.0123,
                      "run_id": "abc123def456"}
    bad = {"retrieval_score": "high", "refinements": True, "verified": "yes", "cost_usd": float("nan"),
           "run_id": "r" * 33}
    assert store.get(store.add("Acme", "t", "jd", "letter", bad)).meta == {}
    assert store.get(store.add("Acme", "t", "jd", "letter")).meta == {}
    assert store.get(store.add("Acme", "t", "jd", "letter", {"cost_usd": float("inf")})).meta == {}
    with pytest.raises(ValueError):
        store.add("Acme", "t", "jd", "letter", ["not", "a", "mapping"])  # type: ignore[arg-type]


# ------------------------------------------------------------------ concurrency and sharing
def test_concurrent_adds_to_one_store(db):
    s = HistoryStore(db)
    errors: List[BaseException] = []

    def work(n: int) -> None:
        try:
            for j in range(5):
                s.add(f"Co {n}", "t", "jd", f"letter {n}-{j}")
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and s.count() == 40
    assert len({e.id for e in s.list(limit=50)}) == 40
    s.close()


def test_concurrent_adds_from_separate_instances(db):
    HistoryStore(db).close()
    errors: List[BaseException] = []

    def work(n: int) -> None:
        try:
            with HistoryStore(db) as s:                       # a new instance per thread, like Streamlit re-runs
                for j in range(5):
                    s.add(f"Co {n}", "t", "jd", f"letter {n}-{j}")
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    with HistoryStore(db) as s:
        assert s.count() == 40


def test_two_instances_on_one_file_see_each_others_writes(db):
    a, b = HistoryStore(db), HistoryStore(db)
    eid = a.add("Acme", "t", "jd", "from a")
    assert b.get(eid).letter == "from a"
    b.add("Beta", "t", "jd", "from b")
    assert a.count() == 2
    assert b.delete(eid) is True
    assert a.get(eid) is None and a.count() == 1
    a.close()
    b.close()


# ------------------------------------------------------------------ errors, paths, schema
def test_corrupt_file_raises_history_error(db):
    db.write_bytes(b"this is definitely not a sqlite database " * 60)
    with pytest.raises(HistoryError):
        HistoryStore(db)


def test_locked_database_raises_history_error(db, monkeypatch):
    s = HistoryStore(db)
    s.add("Acme", "t", "jd", "letter")
    monkeypatch.setattr(history, "BUSY_TIMEOUT_MS", 150)
    other = sqlite3.connect(str(db), isolation_level=None)
    other.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(HistoryError):
            HistoryStore(db)                                  # cannot even open while locked
        short = HistoryStore.__new__(HistoryStore)            # an already-open store whose connection is impatient
        short.__dict__.update(s.__dict__)
        short._conn = sqlite3.connect(str(db), timeout=0.1, isolation_level=None, check_same_thread=False)
        short._conn.execute("PRAGMA busy_timeout = 100")
        with pytest.raises(HistoryError):
            short.add("Beta", "t", "jd", "blocked")
        with pytest.raises(HistoryError):
            short.count()
        short._conn.close()
    finally:
        other.execute("ROLLBACK")
        other.close()
    assert s.count() == 1                                      # still healthy after the lock is released
    s.close()


def test_directory_in_place_of_the_file_raises_history_error(tmp_path):
    (tmp_path / "history.db").mkdir()
    with pytest.raises(HistoryError):
        HistoryStore(tmp_path / "history.db")


def test_closed_store_raises_history_error_and_close_is_idempotent(db):
    s = HistoryStore(db)
    s.close()
    s.close()
    for call in (lambda: s.add("A", "t", "jd", "l"), s.list, s.count, lambda: s.get(1), lambda: s.delete(1)):
        with pytest.raises(HistoryError):
            call()


def test_missing_parent_directory_is_created(tmp_path):
    p = tmp_path / "a" / "b" / "history.db"
    with HistoryStore(p) as s:
        s.add("Acme", "t", "jd", "letter")
    assert p.exists()


def test_default_path_honours_jobagent_home(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBAGENT_HOME", str(tmp_path / "custom_home"))
    with HistoryStore() as s:
        s.add("Acme", "t", "jd", "letter")
    assert (tmp_path / "custom_home" / "history.db").exists()


def test_default_path_without_env_uses_home_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("JOBAGENT_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    with HistoryStore() as s:
        s.add("Acme", "t", "jd", "letter")
    assert (tmp_path / ".job_application_agent" / "history.db").exists()


def test_default_path_is_resolved_at_call_time(tmp_path, monkeypatch):
    for name in ("one", "two"):
        monkeypatch.setenv("JOBAGENT_HOME", str(tmp_path / name))
        HistoryStore().close()
        assert (tmp_path / name / "history.db").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_database_file_is_owner_only(db):
    HistoryStore(db).close()
    assert (db.stat().st_mode & 0o077) == 0


def test_schema_version_is_recorded(db):
    HistoryStore(db).close()
    con = sqlite3.connect(str(db))
    assert con.execute("PRAGMA user_version").fetchone()[0] == history.SCHEMA_VERSION == 1
    assert con.execute("SELECT name FROM sqlite_master WHERE name='idx_letters_created'").fetchone()
    con.close()


def test_a_database_from_a_newer_version_is_refused(db):
    HistoryStore(db).close()
    con = sqlite3.connect(str(db))
    con.execute("PRAGMA user_version = 99")
    con.commit()
    con.close()
    with pytest.raises(HistoryError, match="newer"):
        HistoryStore(db)


def test_migrate_forward_upgrades_in_place_and_keeps_data(db, monkeypatch):
    with HistoryStore(db) as s:
        eid = s.add("Acme", "t", "jd", "kept across the upgrade")

    def to_v2(conn: sqlite3.Connection) -> None:
        conn.execute("ALTER TABLE letters ADD COLUMN starred INTEGER NOT NULL DEFAULT 0")

    monkeypatch.setattr(history, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(history, "_MIGRATIONS", {**history._MIGRATIONS, 2: to_v2})
    with HistoryStore(db) as s:
        assert s.get(eid).letter == "kept across the upgrade"
    con = sqlite3.connect(str(db))
    assert con.execute("PRAGMA user_version").fetchone()[0] == 2
    assert "starred" in [r[1] for r in con.execute("PRAGMA table_info(letters)")]
    con.close()
    with HistoryStore(db) as s:                                 # re-opening must not re-run the migration
        assert s.count() == 1
