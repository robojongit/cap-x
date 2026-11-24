"""Per-trial result records for unattended benchmark runs.

CaP-X's own summary keeps no per-trial timing, token counts or error log, and it drops trials
that crash a worker. This module adds one JSON line per trial and launch to
``<output_dir>/trial_records/trial_NN.jsonl``, written by ``capx.envs.runner`` when the trial ends
(however it ends), so a benchmark driver can score trials, cost them, and see which ones were hit
by infrastructure problems (model-server timeouts, 5xx answers, connection errors) and need a rerun.

Record schema (``schema`` 1)::

    trial, seed, time, pid
    attempts            internal CaP-X retries after a trial timeout (1 = none)
    task_completed, success, reward, sandbox_rc, terminated, truncated,
    num_code_blocks, num_regenerations, num_finishes        from the trial's summary
    seconds_total       trial start to end, including the scene reset
    seconds_after_reset reset done to end: the episode proper
    seconds_llm         time inside model requests
    model_calls         answered model requests; prompt_tokens, completion_tokens, total_tokens
                        from the server's ``usage``; usage_missing_calls counts answers without it
    events              every infrastructure event of every attempt:
                        {kind, status, attempt, t, detail}, kind in request_timeout,
                        connection_error, bad_response, http_error, trial_timeout
    error               repr of the exception that ended the trial, else null

Timing, calls and tokens are those of the last attempt; events cover all attempts.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = 1


class TrialRecorder:
    """Collects model-call statistics and infrastructure events of the trial running in this process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False
        self._reset_state(1)

    def _reset_state(self, attempt: int) -> None:
        self._attempt = attempt
        self._t0 = time.time()
        self._t_reset: float | None = None
        self._llm_seconds = 0.0
        self._calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._usage_missing = 0
        self._events: list[dict[str, Any]] = []

    def begin_attempt(self, attempt: int = 1) -> None:
        with self._lock:
            self._reset_state(attempt)
            self._active = True

    def mark_reset_done(self) -> None:
        with self._lock:
            if self._active:
                self._t_reset = time.time()

    def call_ok(self, seconds: float, usage: dict[str, Any] | None) -> None:
        with self._lock:
            if not self._active:
                return
            self._calls += 1
            self._llm_seconds += seconds
            if usage and usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
                self._prompt_tokens += int(usage["prompt_tokens"])
                self._completion_tokens += int(usage["completion_tokens"])
            else:
                self._usage_missing += 1

    def event(self, kind: str, status: int | None = None, detail: str = "") -> None:
        with self._lock:
            if not self._active:
                return
            self._events.append(
                {
                    "kind": kind,
                    "status": status,
                    "attempt": self._attempt,
                    "t": round(time.time() - self._t0, 2),
                    "detail": detail[:300],
                }
            )

    def end_attempt(self) -> dict[str, Any]:
        """Statistics of the attempt that just ended (the recorder goes inactive)."""
        with self._lock:
            now = time.time()
            self._active = False
            return {
                "seconds_total": now - self._t0,
                "seconds_after_reset": (now - self._t_reset) if self._t_reset is not None else None,
                "seconds_llm": self._llm_seconds,
                "model_calls": self._calls,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._prompt_tokens + self._completion_tokens,
                "usage_missing_calls": self._usage_missing,
                "events": list(self._events),
            }


RECORDER = TrialRecorder()


def _json_default(o: Any) -> Any:
    """numpy scalars (``task_completed`` is a numpy bool in some tasks) and anything else odd."""
    return o.item() if hasattr(o, "item") else str(o)


def records_dir(output_dir: str | os.PathLike[str]) -> Path:
    return Path(output_dir) / "trial_records"


def write_record(output_dir: str | os.PathLike[str], trial: int, record: dict[str, Any]) -> Path:
    """Append one record line for ``trial`` (a rerun adds a line; the last line is the result)."""
    d = records_dir(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"trial_{trial:02d}.jsonl"
    line = json.dumps(
        {
            "schema": SCHEMA,
            "trial": trial,
            "seed": trial,
            "time": time.strftime("%FT%TZ", time.gmtime()),
            "pid": os.getpid(),
            **record,
        },
        default=_json_default,
    )
    with path.open("a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return path


def archive_previous_attempts(output_dir: str | os.PathLike[str], trial: int) -> list[Path]:
    """Move artifact directories of an earlier launch of ``trial`` to ``superseded/<n>/``.

    The directory names encode the outcome (``trial_07_sandboxrc_0_reward_..._taskcompleted_1``),
    so without this a rerun would leave two contradicting directories for one trial.
    """
    out = Path(output_dir)
    old = sorted(p for p in out.glob(f"trial_{trial:02d}_*") if p.is_dir())
    if not old:
        return []
    n = 1
    while (out / "superseded" / f"{trial:02d}_{n}").exists():
        n += 1
    dest = out / "superseded" / f"{trial:02d}_{n}"
    dest.mkdir(parents=True)
    return [Path(shutil.move(str(p), str(dest / p.name))) for p in old]
