"""Shared per-process attempt timing and failure accounting for all recorders."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import builtins
import json
from pathlib import Path
import re
import sys
import time


_ACTIVE_METRICS = None


def activate_metrics(metrics):
    global _ACTIVE_METRICS
    _ACTIVE_METRICS = metrics


def finalize_active_metrics(runtime_error=None):
    if _ACTIVE_METRICS is not None:
        return _ACTIVE_METRICS.finish(runtime_error)
    return None


def _forwarded_value(args: list[str], names: tuple[str, ...], default: str) -> str:
    for index, value in enumerate(args):
        if value in names and index + 1 < len(args):
            return args[index + 1]
        for name in names:
            prefix = name + "="
            if value.startswith(prefix):
                return value[len(prefix):]
    return default


class _ObservedStream:
    def __init__(self, wrapped, callback):
        self._wrapped = wrapped
        self._callback = callback
        self._buffer = ""

    def write(self, text):
        result = self._wrapped.write(text)
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._callback(line.rstrip("\r"))
        return result

    def flush(self):
        self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def finish(self):
        if self._buffer:
            self._callback(self._buffer.rstrip("\r"))
            self._buffer = ""


class RunMetrics:
    def __init__(self, object_name: str, forwarded_args: list[str]):
        self.object_name = object_name
        self.out_dir = Path(_forwarded_value(
            forwarded_args, ("--out_dir", "--out-dir"), "datasets/phase3_grid_split"
        )).resolve()
        self.wall_started = time.perf_counter()
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.attempt_started = None
        self.attempt_number = None
        self.attempts = []
        self.failure_reasons = Counter()
        self._original_stdout = None
        self._observed_stdout = None
        self._original_print = None
        self._finished_payload = None
        self.initial_h5_count = self._count_saved_episodes()

    def _count_saved_episodes(self) -> int:
        """Count complete successful demonstrations without double-counting segments."""
        from src.recorder.episode_split import episode_files
        pick = len(episode_files(self.out_dir,'pick'))
        place = len(episode_files(self.out_dir,'place'))
        return max(pick, place)

    def _start_attempt(self, number: int):
        if self.attempt_started is not None:
            self._finish_attempt("unclassified_fail", "next attempt started before terminal result")
        self.attempt_number = number
        self.attempt_started = time.perf_counter()

    def _finish_attempt(self, status: str, reason: str):
        if self.attempt_started is None:
            return
        seconds = time.perf_counter() - self.attempt_started
        self.attempts.append({
            "attempt": self.attempt_number,
            "status": status,
            "reason": reason,
            "seconds": round(seconds, 3),
            "minutes": round(seconds / 60.0, 3),
        })
        if status != "success":
            self.failure_reasons[status] += 1
        self.attempt_started = None
        self.attempt_number = None

    def observe_line(self, line: str):
        # Strip ANSI codes so parsing also works when colored logging is enabled.
        plain = re.sub(r"\x1b\[[0-9;]*m", "", str(line))
        match = re.search(r"ATTEMPT\s+(\d+)\s*\|", plain, flags=re.IGNORECASE)
        if match:
            self._start_attempt(int(match.group(1)))
            return
        if "[SPAWN FAIL]" in plain:
            self._finish_attempt("spawn_fail", plain.strip())
            return
        if "[SPAWN] no valid" in plain or "settle validation" in plain.lower():
            self._finish_attempt("spawn_fail", plain.strip())
            return
        if "[PHASE FAIL]" in plain or "QUALITY FAIL" in plain:
            failure = "place_fail" if "place" in plain.lower() else "pick_fail"
            self._finish_attempt(failure, plain.strip())
            return
        if re.search(r"\[RESULT\]\s+success\s*=\s*True", plain, flags=re.IGNORECASE):
            self._finish_attempt("success", plain.strip())
            return
        if re.search(r"\[RESULT\]\s+success\s*=\s*False", plain, flags=re.IGNORECASE):
            self._finish_attempt("execution_fail", plain.strip())

    @contextmanager
    def observe_stdout(self):
        self._original_stdout = sys.stdout
        # Isaac/Kit may replace sys.stdout during startup. Intercepting print as
        # a function keeps attempt accounting alive across that replacement.
        self._original_print = builtins.print

        def observed_print(*args, **kwargs):
            separator = kwargs.get("sep", " ")
            ending = kwargs.get("end", "\n")
            rendered = separator.join(str(value) for value in args)
            if ending and "\n" in rendered + ending:
                for line in (rendered + ending).splitlines():
                    self.observe_line(line)
            return self._original_print(*args, **kwargs)

        builtins.print = observed_print
        try:
            yield
        finally:
            builtins.print = self._original_print

    def finish(self, runtime_error: BaseException | None = None):
        if self._finished_payload is not None:
            return self._finished_payload
        if self.attempt_started is not None:
            status = "runtime_error" if runtime_error is not None else "incomplete_fail"
            reason = repr(runtime_error) if runtime_error is not None else "recorder ended before terminal result"
            self._finish_attempt(status, reason)

        wall_seconds = time.perf_counter() - self.wall_started
        saved_episode_delta = max(0, self._count_saved_episodes() - self.initial_h5_count)
        metrics_source = "observed_print"
        if not self.attempts and saved_episode_delta:
            # We can recover successful count exactly from atomic episode H5
            # files. Failed-attempt timing cannot be reconstructed afterward.
            metrics_source = "h5_fallback"
            for index in range(saved_episode_delta):
                self.attempts.append({
                    "attempt": index + 1,
                    "status": "success",
                    "reason": "recovered from completed H5",
                    "seconds": 0.0,
                    "minutes": 0.0,
                })
        success_seconds = sum(a["seconds"] for a in self.attempts if a["status"] == "success")
        failed_seconds = sum(a["seconds"] for a in self.attempts if a["status"] != "success")
        overhead_seconds = max(0.0, wall_seconds - success_seconds - failed_seconds)
        payload = {
            "format_version": 2,
            "metrics_source": metrics_source,
            "object": self.object_name,
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "out_dir": str(self.out_dir),
            "total_attempts": len(self.attempts),
            "success_count": sum(a["status"] == "success" for a in self.attempts),
            "failure_count": sum(a["status"] != "success" for a in self.attempts),
            "spawn_fail_count": self.failure_reasons["spawn_fail"],
            "pick_fail_count": self.failure_reasons["pick_fail"],
            "place_fail_count": self.failure_reasons["place_fail"],
            "other_fail_count": sum(v for k, v in self.failure_reasons.items()
                                    if k not in {"spawn_fail", "pick_fail", "place_fail"}),
            "total_wall_seconds": round(wall_seconds, 3),
            "total_wall_minutes": round(wall_seconds / 60.0, 3),
            "successful_attempt_seconds": round(success_seconds, 3),
            "successful_attempt_minutes": round(success_seconds / 60.0, 3),
            "failed_attempt_seconds": round(failed_seconds, 3),
            "failed_attempt_minutes": round(failed_seconds / 60.0, 3),
            "startup_save_shutdown_seconds": round(overhead_seconds, 3),
            "startup_save_shutdown_minutes": round(overhead_seconds / 60.0, 3),
            "attempts": self.attempts,
        }

        self.out_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.out_dir / "run_metrics.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        summary_lines = [
            "================================================================",
            "                    RECORDER RUN SUMMARY",
            "================================================================",
            f"  {'Object':<24} : {self.object_name}",
            f"  {'Attempts':<24} : {payload['total_attempts']}",
            f"  {'Successful':<24} : {payload['success_count']}",
            f"  {'Failed':<24} : {payload['failure_count']}",
            f"  {'Spawn failures':<24} : {payload['spawn_fail_count']}",
            f"  {'Pick failures':<24} : {payload['pick_fail_count']}",
            f"  {'Place failures':<24} : {payload['place_fail_count']}",
            f"  {'Other failures':<24} : {payload['other_fail_count']}",
            f"  {'Successful minutes':<24} : {payload['successful_attempt_minutes']:.3f}",
            f"  {'Failed minutes':<24} : {payload['failed_attempt_minutes']:.3f}",
            f"  {'Startup/save minutes':<24} : {payload['startup_save_shutdown_minutes']:.3f}",
            f"  {'Total wall minutes':<24} : {payload['total_wall_minutes']:.3f}",
            f"  {'Metrics source':<24} : {payload['metrics_source']}",
            "----------------------------------------------------------------",
            "  FAILED ATTEMPTS",
        ]
        failed_attempts = [a for a in self.attempts if a["status"] != "success"]
        if failed_attempts:
            for attempt in failed_attempts:
                summary_lines.extend([
                    f"  Attempt #{attempt['attempt']:04d}",
                    f"    status   : {attempt['status']}",
                    f"    minutes  : {attempt['minutes']:.3f}",
                    f"    reason   : {attempt['reason']}",
                ])
        else:
            summary_lines.append("  None")
        summary_lines.append("================================================================")
        text_path = self.out_dir / "run_metrics_summary.txt"
        text_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        green, red, yellow, cyan, reset = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[0m"
        width = 64
        print("\n" + cyan + "=" * width + reset)
        print(cyan + "                  RECORDER RUN SUMMARY" + reset)
        print(cyan + "=" * width + reset)
        rows = (
            ("Object", self.object_name),
            ("Attempts", payload["total_attempts"]),
            ("Successful", green + str(payload["success_count"]) + reset),
            ("Failed", (red if payload["failure_count"] else green) + str(payload["failure_count"]) + reset),
            ("Spawn failures", payload["spawn_fail_count"]),
            ("Pick failures", payload["pick_fail_count"]),
            ("Place failures", payload["place_fail_count"]),
            ("Other failures", payload["other_fail_count"]),
            ("Successful minutes", f"{payload['successful_attempt_minutes']:.3f}"),
            ("Failed minutes", f"{payload['failed_attempt_minutes']:.3f}"),
            ("Startup/save minutes", f"{payload['startup_save_shutdown_minutes']:.3f}"),
            ("Total wall minutes", yellow + f"{payload['total_wall_minutes']:.3f}" + reset),
            ("Metrics source", payload["metrics_source"]),
        )
        for label, value in rows:
            print(f"  {label:<24} : {value}")
        print(cyan + "-" * width + reset)
        print(f"  JSON : {output_path}")
        print(f"  TXT  : {text_path}")
        print(cyan + "=" * width + reset)
        self._finished_payload = payload
        return payload
