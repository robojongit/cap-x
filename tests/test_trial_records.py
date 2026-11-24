"""Per-trial records, the client's infrastructure-event logging, and ``--trial-ids``.

No simulator and no real model server: a small local HTTP server stands in for the model.
Run with ``uv run pytest tests/test_trial_records.py``.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import requests

from capx.llm.client import ModelQueryArgs, query_model
from capx.utils import trial_records
from capx.utils.launch_utils import TrialSummary, parse_trial_ids
from capx.utils.trial_records import RECORDER


class _Server:
    """Answers POSTs from a script: a list of (status, delay_seconds, body) consumed in order."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests += 1
                status, delay, body = outer.script.pop(0) if outer.script else (200, 0, _ok())
                time.sleep(delay)
                data = json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _ok(prompt=100, completion=40, usage=True):
    body = {"choices": [{"message": {"content": "```python\nprint(1)\n```", "reasoning": "r"}}]}
    if usage:
        body["usage"] = {"prompt_tokens": prompt, "completion_tokens": completion}
    return body


@pytest.fixture()
def server():
    made = []

    def make(script):
        s = _Server(script)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


def _args(url):
    return ModelQueryArgs(model="inkling-2-mini", server_url=url)


@pytest.fixture(autouse=True)
def _fast_client(monkeypatch):
    monkeypatch.setenv("CAPX_LLM_TIMEOUT_S", "2")
    monkeypatch.setenv("CAPX_LLM_RETRY_SLEEP_S", "0.2")
    monkeypatch.delenv("CAPX_LLM_MAX_RETRIES", raising=False)


# ----------------------------------------------------------------------------- --trial-ids


def test_parse_trial_ids():
    assert parse_trial_ids(None) is None
    assert parse_trial_ids("") is None
    assert parse_trial_ids("3,7,10-12") == [3, 7, 10, 11, 12]
    assert parse_trial_ids("5,5,2") == [2, 5]
    for bad in ("0", "a", "4-2", "1,,2x"):
        with pytest.raises(ValueError):
            parse_trial_ids(bad)


# ----------------------------------------------------------------------------- client + recorder


def test_successful_call_counts_tokens_and_calls(server):
    srv = server([(200, 0, _ok(prompt=120, completion=30))])
    RECORDER.begin_attempt()
    out = query_model(_args(srv.url), [{"role": "user", "content": "hi"}])
    stats = RECORDER.end_attempt()
    assert out["content"].startswith("```python")
    assert stats["model_calls"] == 1 and stats["total_tokens"] == 150
    assert stats["prompt_tokens"] == 120 and stats["completion_tokens"] == 30
    assert stats["events"] == [] and stats["seconds_llm"] > 0


def test_missing_usage_is_counted_not_guessed(server):
    srv = server([(200, 0, _ok(usage=False))])
    RECORDER.begin_attempt()
    query_model(_args(srv.url), [])
    stats = RECORDER.end_attempt()
    assert stats["model_calls"] == 1 and stats["total_tokens"] == 0 and stats["usage_missing_calls"] == 1


def test_server_error_is_logged_and_retried_with_the_configured_sleep(server):
    srv = server([(503, 0, {"error": "busy"}), (502, 0, {"error": "busy"}), (200, 0, _ok())])
    RECORDER.begin_attempt()
    t0 = time.time()
    query_model(_args(srv.url), [])
    elapsed = time.time() - t0
    stats = RECORDER.end_attempt()
    assert [(e["kind"], e["status"]) for e in stats["events"]] == [("http_error", 503), ("http_error", 502)]
    assert stats["model_calls"] == 1 and srv.requests == 3
    assert elapsed < 5  # two sleeps of ~0.2 s, not 2 x 4 minutes


def test_default_retry_sleep_is_unchanged(monkeypatch):
    from capx.llm import client

    monkeypatch.delenv("CAPX_LLM_RETRY_SLEEP_S", raising=False)
    sleeps = {client._retry_sleep_seconds() for _ in range(200)}
    assert min(sleeps) >= 150 and max(sleeps) <= 330 and max(sleeps) - min(sleeps) > 60  # 240 +- 90


def test_retries_give_up_when_a_limit_is_set(server, monkeypatch):
    monkeypatch.setenv("CAPX_LLM_MAX_RETRIES", "2")
    srv = server([(500, 0, {"error": "x"})] * 10)
    RECORDER.begin_attempt()
    with pytest.raises(requests.HTTPError):
        query_model(_args(srv.url), [])
    stats = RECORDER.end_attempt()
    assert srv.requests == 3  # the first request and two retries
    assert len(stats["events"]) == 3 and all(e["status"] == 500 for e in stats["events"])


def test_request_timeout_is_logged_and_raised(server, monkeypatch):
    monkeypatch.setenv("CAPX_LLM_TIMEOUT_S", "0.5")
    srv = server([(200, 3, _ok())])
    RECORDER.begin_attempt()
    with pytest.raises(requests.exceptions.Timeout):
        query_model(_args(srv.url), [])
    stats = RECORDER.end_attempt()
    assert [e["kind"] for e in stats["events"]] == ["request_timeout"] and stats["model_calls"] == 0


def test_connection_error_is_logged_and_raised():
    RECORDER.begin_attempt()
    with pytest.raises(requests.exceptions.ConnectionError):
        query_model(_args("http://127.0.0.1:9/v1/chat/completions"), [])
    stats = RECORDER.end_attempt()
    assert [e["kind"] for e in stats["events"]] == ["connection_error"]


def test_client_error_is_not_retried_but_logged(server):
    srv = server([(400, 0, {"error": "bad"})])
    RECORDER.begin_attempt()
    with pytest.raises(requests.HTTPError):
        query_model(_args(srv.url), [])
    stats = RECORDER.end_attempt()
    assert srv.requests == 1 and [(e["kind"], e["status"]) for e in stats["events"]] == [("http_error", 400)]


def test_unparseable_answer_is_logged(server):
    srv = server([(200, 0, "not json at all")])
    RECORDER.begin_attempt()
    with pytest.raises(Exception, match=".*"):
        query_model(_args(srv.url), [])
    stats = RECORDER.end_attempt()
    assert [e["kind"] for e in stats["events"]] == ["bad_response"]


def test_calls_outside_a_trial_are_ignored(server):
    srv = server([(200, 0, _ok())])
    RECORDER.begin_attempt()
    RECORDER.end_attempt()  # inactive
    query_model(_args(srv.url), [])
    RECORDER.begin_attempt()
    assert RECORDER.end_attempt()["model_calls"] == 0


# ----------------------------------------------------------------------------- records on disk


def test_write_record_appends_and_archive_moves_old_artifacts(tmp_path: Path):
    out = tmp_path / "out"
    (out / "trial_07_sandboxrc_0_reward_0.000_taskcompleted_0").mkdir(parents=True)
    (out / "trial_07_sandboxrc_0_reward_0.000_taskcompleted_0" / "code.py").write_text("x")
    (out / "trial_08_sandboxrc_0_reward_1.000_taskcompleted_1").mkdir()
    moved = trial_records.archive_previous_attempts(out, 7)
    assert len(moved) == 1 and (out / "superseded" / "07_1" / moved[0].name / "code.py").exists()
    assert (out / "trial_08_sandboxrc_0_reward_1.000_taskcompleted_1").exists()  # other trials untouched
    assert trial_records.archive_previous_attempts(out, 7) == []
    trial_records.write_record(out, 7, {"task_completed": False})
    trial_records.write_record(out, 7, {"task_completed": np.bool_(True), "reward": np.float64(1.0)})
    lines = [json.loads(x) for x in (out / "trial_records" / "trial_07.jsonl").read_text().splitlines()]
    assert [r["task_completed"] for r in lines] == [False, True] and lines[0]["seed"] == 7 and lines[0]["schema"] == 1


def _summary(trial, **kw):
    base = {
        "trial": trial, "success": True, "reward": 1.0, "terminated": True, "truncated": False,
        "sandbox_rc": 0, "log": "", "task_completed": True, "num_code_blocks": 1,
    }  # fmt: skip
    return TrialSummary(**{**base, **kw})


def test_runner_writes_one_record_per_trial_and_aggregates_attempt_events(tmp_path, monkeypatch):
    from capx.envs import runner

    calls = []

    def fake_single(env, trial, args, config, multi_turn_prompt, timeout_s, raise_on_timeout=False):
        calls.append((trial, raise_on_timeout))
        RECORDER.mark_reset_done()
        RECORDER.call_ok(0.5, {"prompt_tokens": 10, "completion_tokens": 5})
        if len(calls) == 1:
            RECORDER.event("trial_timeout", detail="alarm")
            raise TimeoutError("Trial timed out")
        return _summary(trial)

    monkeypatch.setattr(runner, "_run_single_trial_with_timeout", fake_single)
    config = {"output_dir": str(tmp_path)}
    summary = runner._run_trial_with_retries(None, 4, None, config, None)
    assert summary.task_completed is True
    rec = json.loads((tmp_path / "trial_records" / "trial_04.jsonl").read_text())
    assert rec["attempts"] == 2 and rec["model_calls"] == 1 and rec["total_tokens"] == 15
    assert [e["kind"] for e in rec["events"]] == ["trial_timeout"] and rec["events"][0]["attempt"] == 1
    assert rec["task_completed"] is True and rec["error"] is None and rec["seconds_after_reset"] is not None


def test_runner_records_a_crashed_trial_and_reraises(tmp_path, monkeypatch):
    from capx.envs import runner

    def boom(*a, **k):
        RECORDER.event("connection_error", detail="refused")
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(runner, "_run_single_trial_with_timeout", boom)
    with pytest.raises(requests.exceptions.ConnectionError):
        runner._run_trial_with_retries(None, 9, None, {"output_dir": str(tmp_path)}, None)
    rec = json.loads((tmp_path / "trial_records" / "trial_09.jsonl").read_text())
    assert rec["task_completed"] is None and "ConnectionError" in rec["error"]
    assert [e["kind"] for e in rec["events"]] == ["connection_error"]


def test_runner_all_attempts_timing_out_still_records(tmp_path, monkeypatch):
    from capx.envs import runner

    def always_timeout(env, trial, args, config, multi_turn_prompt, timeout_s, raise_on_timeout=False):
        RECORDER.event("trial_timeout")
        if raise_on_timeout:
            raise TimeoutError("t")
        return _summary(trial, success=False, task_completed=False, truncated=True)

    monkeypatch.setattr(runner, "_run_single_trial_with_timeout", always_timeout)
    runner._run_trial_with_retries(None, 2, None, {"output_dir": str(tmp_path)}, None)
    rec = json.loads((tmp_path / "trial_records" / "trial_02.jsonl").read_text())
    assert rec["attempts"] == runner.MAX_TRIAL_RETRIES and len(rec["events"]) == runner.MAX_TRIAL_RETRIES
