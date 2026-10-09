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


def failure_signature(attempt):
    """Return stable stage/reason labels without copying noisy numeric telemetry."""
    text = str(attempt.get("reason", ""))
    stage_match = re.search(r"\[PHASE FAIL\]\s+(?:pick|place)\s*\|\s*([^|;]+)", text, re.I)
    stage = stage_match.group(1).strip() if stage_match else {
        "spawn_fail": "SPAWN", "sensor_fail": "SENSOR", "runtime_error": "RUNTIME"
    }.get(attempt.get("status"), "OTHER")
    detail = text.split("|", 2)[-1].strip() if "[PHASE FAIL]" in text else text.strip()
    detail = detail.split(";", 1)[0].strip()
    detail = re.sub(r":\s*(?:error|distance)=[^,;]+(?:[,;]\s*angle=[^,;]+)?", "", detail, flags=re.I)
    detail = re.sub(r"\b(?:error|angle|base_budget|progressing_grace|steps?)\s*[=:<>]+\s*[-+\d.]+\s*(?:mm|m|deg|rad)?", "", detail, flags=re.I)
    detail = re.sub(r"\s+", " ", detail).strip(" ,;:-")
    if not detail:
        detail = attempt.get("status", "unknown_failure")
    return stage, detail


def failure_breakdown(attempts):
    counts = Counter(failure_signature(a) for a in attempts if a.get("status") != "success")
    total = sum(counts.values())
    return [{"stage": stage, "reason": reason, "count": count,
             "percent": round(100.0 * count / total, 1) if total else 0.0}
            for (stage, reason), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def reconcile_attempt_timeline(data):
    """Fill legacy/consolidated aggregate-only attempts so graph endpoints match totals."""
    attempts = list(data.get("attempts", []))
    have_success = sum(a.get("status") == "success" for a in attempts)
    have_failure = len(attempts) - have_success
    missing_success = max(0, int(data.get("success_count", have_success)) - have_success)
    missing_failure = max(0, int(data.get("failure_count", have_failure)) - have_failure)
    attempts.extend({"status": "success", "reason": "aggregate-only recovered attempt"}
                    for _ in range(missing_success))
    attempts.extend({"status": "legacy_fail", "reason": "aggregate-only recovered attempt"}
                    for _ in range(missing_failure))
    return attempts


def write_failure_chart(path, rows, object_name):
    """Write a compact dependency-free PNG ranking; detailed attempts stay in JSON."""
    if not rows:
        return None
    from PIL import Image, ImageDraw, ImageFont
    width, left, right, top, row_h = 1200, 390, 90, 90, 52
    height = top + row_h * len(rows) + 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        title = ImageFont.truetype("arialbd.ttf", 25)
    except OSError:
        font = title = ImageFont.load_default()
    draw.text((35, 25), f"{object_name} - failure ranking", fill="#172033", font=title)
    maximum = max(row["count"] for row in rows)
    bar_width = width - left - right
    colors = {"pick_fail": "#dc6b5d", "place_fail": "#7656c9", "spawn_fail": "#d99a2b",
              "sensor_fail": "#278ca5", "runtime_error": "#777777"}
    for index, row in enumerate(rows):
        y = top + index * row_h
        label = f'{row["stage"]}: {row["reason"]}'
        if len(label) > 42:
            label = label[:39] + "..."
        draw.text((35, y + 10), label, fill="#263244", font=font)
        length = round(bar_width * row["count"] / maximum)
        draw.rounded_rectangle((left, y + 7, left + length, y + 39), radius=7, fill="#4776c5")
        draw.text((left + length + 12, y + 10), f'{row["count"]} ({row["percent"]:.1f}%)', fill="#172033", font=font)
    image.save(path)
    return path


def write_attempt_timeline(path, attempts, object_name, goal=100):
    """Plot cumulative success/failure lines against attempt number."""
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1200, 650
    left, right, top, bottom = 95, 45, 80, 85
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        title = ImageFont.truetype("arialbd.ttf", 27)
    except OSError:
        font = title = ImageFont.load_default()
    draw.text((left, 25), f"{object_name} - cumulative recording outcome", fill="#172033", font=title)
    plot_w, plot_h = width - left - right, height - top - bottom
    x_max = max(1, len(attempts))
    success_total = sum(a.get("status") == "success" for a in attempts)
    failure_total = len(attempts) - success_total
    y_max = max(goal, success_total, failure_total, 1)
    for tick in range(0, 6):
        value = round(y_max * tick / 5)
        y = top + plot_h - round(plot_h * value / y_max)
        draw.line((left, y, left + plot_w, y), fill="#dfe5ec", width=1)
        draw.text((45, y - 10), str(value), fill="#526070", font=font)
    for tick in range(0, 6):
        value = round(x_max * tick / 5)
        x = left + round(plot_w * value / x_max)
        draw.line((x, top, x, top + plot_h), fill="#eef1f5", width=1)
        draw.text((x - 10, top + plot_h + 12), str(value), fill="#526070", font=font)
    draw.line((left, top, left, top + plot_h), fill="#344154", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#344154", width=2)
    draw.text((width // 2 - 65, height - 38), "Attempt", fill="#172033", font=font)
    draw.text((15, top + plot_h // 2), "Count", fill="#172033", font=font)
    successes = failures = 0
    success_points = [(left, top + plot_h)]
    failure_points = [(left, top + plot_h)]
    for index, attempt in enumerate(attempts, 1):
        if attempt.get("status") == "success": successes += 1
        else: failures += 1
        x = left + round(plot_w * index / x_max)
        success_points.append((x, top + plot_h - round(plot_h * successes / y_max)))
        failure_points.append((x, top + plot_h - round(plot_h * failures / y_max)))
    if len(success_points) > 1:
        draw.line(success_points, fill="#218c63", width=4, joint="curve")
        draw.line(failure_points, fill="#d35b53", width=4, joint="curve")
    draw.rounded_rectangle((left + 12, top + 12, left + 330, top + 62), 8, fill="#f7f9fb", outline="#d5dce5")
    draw.line((left + 28, top + 29, left + 73, top + 29), fill="#218c63", width=4)
    draw.text((left + 82, top + 18), f"Success {successes}", fill="#172033", font=font)
    draw.line((left + 185, top + 29, left + 230, top + 29), fill="#d35b53", width=4)
    draw.text((left + 239, top + 18), f"Fail {failures}", fill="#172033", font=font)
    image.save(path)
    return path


def write_failure_reason_line(path, rows, object_name):
    """Plot ranked failure categories as a line with a readable reason legend."""
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1300, 720
    left, top, plot_w, plot_h = 85, 90, 720, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 17)
        bold = ImageFont.truetype("arialbd.ttf", 18)
        title = ImageFont.truetype("arialbd.ttf", 27)
    except OSError:
        font = bold = title = ImageFont.load_default()
    draw.text((left, 25), f"{object_name} - failure reasons ranked", fill="#172033", font=title)
    if not rows:
        draw.text((left + 250, top + 220), "NO FAILURES RECORDED", fill="#6b7480", font=title)
        image.save(path)
        return path
    maximum = max(row["count"] for row in rows)
    y_max = max(5, maximum)
    for tick in range(6):
        value = round(y_max * tick / 5)
        y = top + plot_h - round(plot_h * value / y_max)
        draw.line((left, y, left + plot_w, y), fill="#dfe5ec", width=1)
        draw.text((35, y - 10), str(value), fill="#526070", font=font)
    draw.line((left, top, left, top + plot_h), fill="#344154", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#344154", width=2)
    denominator = max(1, len(rows) - 1)
    points = []
    for index, row in enumerate(rows):
        x = left + round(plot_w * index / denominator)
        y = top + plot_h - round(plot_h * row["count"] / y_max)
        points.append((x, y))
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#7656c9", outline="white", width=2)
        draw.text((x - 12, top + plot_h + 14), f"R{index + 1}", fill="#172033", font=bold)
        draw.text((x - 8, y - 32), str(row["count"]), fill="#172033", font=bold)
    if len(points) > 1:
        draw.line(points, fill="#7656c9", width=4, joint="curve")
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#7656c9", outline="white", width=2)
    draw.text((left + 290, height - 38), "Failure reason rank", fill="#172033", font=font)
    draw.text((15, top + 245), "Count", fill="#172033", font=font)
    draw.text((850, 70), "REASON KEY", fill="#172033", font=bold)
    for index, row in enumerate(rows):
        y = 108 + index * 57
        reason = str(row["reason"])
        if len(reason) > 35:
            reason = reason[:32] + "..."
        draw.text((850, y), f"R{index + 1}  {row['count']} ({row['percent']:.1f}%)", fill="#7656c9", font=bold)
        draw.text((850, y + 22), str(row["stage"]), fill="#263244", font=font)
        draw.text((850, y + 40), reason, fill="#526070", font=font)
    image.save(path)
    return path


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
        payload["failure_breakdown"] = failure_breakdown(self.attempts)

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
            "  FAILURE BREAKDOWN (highest first)",
        ]
        if payload["failure_breakdown"]:
            summary_lines.append("  COUNT   PERCENT   STAGE                         REASON")
            for row in payload["failure_breakdown"]:
                summary_lines.append(
                    f"  {row['count']:>5}   {row['percent']:>6.1f}%   {row['stage']:<29} {row['reason']}"
                )
        else:
            summary_lines.append("  None")
        summary_lines.append("================================================================")
        text_path = self.out_dir / "run_metrics_summary.txt"
        text_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        chart_path = write_failure_chart(
            self.out_dir / "failure_breakdown.png", payload["failure_breakdown"], self.object_name)
        reason_line_path = write_failure_reason_line(
            self.out_dir / "failure_reason_line.png", payload["failure_breakdown"], self.object_name)
        timeline_path = write_attempt_timeline(
            self.out_dir / "attempt_timeline.png", self.attempts, self.object_name)

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
        if chart_path:
            print(f"  PNG  : {chart_path}")
        print(f"  FAIL LINE: {reason_line_path}")
        print(f"  LINE : {timeline_path}")
        print(cyan + "=" * width + reset)
        self._finished_payload = payload
        return payload
