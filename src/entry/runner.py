"""Compatibility runner for the isolated Phase-3 refactor.

It deliberately executes the already-verified object backend unchanged.  The
new folder owns the shared environment modules by putting itself first on
``sys.path``.  Consequently environment edits can be made in refactored_v2
without altering the active with_env_cfg implementation.
"""

from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path

from object_handlers import ObjectHandler
from phase3_run_metrics import activate_metrics
from phase4_metrics import RunMetrics, require_saved_goal


HERE = Path(__file__).resolve().parent
LEGACY_ROOT = HERE
BACKEND_ROOT = HERE / "backends"


def _validate_files(handler: ObjectHandler) -> Path:
    backend = BACKEND_ROOT / handler.backend
    required = (
        backend,
        HERE / "phase3_shared_env_cfg.py",
        HERE / "phase3_recorder_camera_patch.py",
        HERE / "phase3_camera_tuning.py",
        HERE/'env'/"shared_layout.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing refactor dependency:\n" + "\n".join(missing))

    source = backend.read_text(encoding="utf-8-sig")
    target_match = re.search(
        r'^PHASE3_TARGET_OBJECT\s*=\s*[\"\']([^\"\']+)[\"\']',
        source,
        flags=re.MULTILINE,
    )
    if not target_match or target_match.group(1) != handler.name:
        actual = target_match.group(1) if target_match else "<not found>"
        raise RuntimeError(
            f"Backend target mismatch: requested={handler.name}, backend={actual}")
    return backend


def run_handler(handler: ObjectHandler, forwarded_args: list[str]) -> None:
    backend = _validate_files(handler)

    # Refactored shared modules win; the proven recorder remains the motion
    # backend. Remove cached copies to avoid accidentally using another folder.
    for module_name in (
        "phase3_shared_env_cfg",
        "phase3_recorder_camera_patch",
        "phase3_camera_tuning",
    ):
        sys.modules.pop(module_name, None)

    sys.path.insert(0, str(LEGACY_ROOT))
    sys.path.insert(0, str(HERE))
    sys.argv = [str(backend), *forwarded_args]

    print("=" * 72)
    print("[PHASE4 HOSPITAL - P3 OBJECT HANDLERS]")
    print("object       =", handler.name)
    print("object id    =", handler.object_type_id)
    print("motion       =", backend)
    print("shared env   =", HERE / "phase3_shared_env_cfg.py")
    print("scene layout =", HERE/'env'/"scene_layout.json")
    print("camera layout=", HERE/'env'/"camera_layout.json")
    print("=" * 72)
    from phase4_scene import write_run_manifest
    write_run_manifest(handler.name, forwarded_args)
    metrics = RunMetrics(handler.name, forwarded_args)
    activate_metrics(metrics)
    runtime_error = None
    loaded = None
    def terminal_status(state,message):
        try:
            from phase4_session import status
            ns=loaded['main'].__globals__ if loaded else {'PHASE3_TARGET_OBJECT':handler.name}
            status(ns,state,message=message,saved=ns.get('_p4_requested_count',0) if state=='complete' else ns.get('_p4_saved_count',0))
        except Exception as status_error:
            print('[P4 STATUS ERROR]',repr(status_error),flush=True)
    def coverage_report(require_complete=False):
        from phase4_session import load_session
        cfg=load_session()
        if not cfg or cfg.get('collection')!='grid_cycles': return
        import json
        from phase4_coverage import audit_directory
        from phase4_scene import LAYOUT
        from phase3_run_metrics import _forwarded_value
        report=audit_directory(metrics.out_dir,handler.name,_forwarded_value(forwarded_args,('--record_mode',),'both'),
                               int(cfg['cycles']),LAYOUT['grid_rows'],LAYOUT['grid_cols'])
        (metrics.out_dir/'coverage_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if require_complete and not report['complete']:
            raise RuntimeError('COVERAGE_INCOMPLETE: inspect coverage_report.json; no cells are silently skipped')
    try:
        # Bootstrap Kit without replacing __main__ or wrapping print during
        # extension startup. Enter the unchanged recorder main afterwards,
        # matching the working preview/direct-backend initialization path.
        loaded = runpy.run_path(str(backend), run_name=f"p4_{handler.name}_backend")
        import os
        loaded['main'].__globals__['RANDOM_SEED'] = int(os.environ.get('P4_RANDOMIZATION_SEED', '17'))
        app=loaded.get('simulation_app')
        original_close=app.close if app else None
        if app:
            def checked_close(*args,**kwargs):
                # Kit fast shutdown can terminate Python before finally runs.
                # Validate and flush metrics BEFORE entering native shutdown.
                require_saved_goal(metrics,forwarded_args)
                coverage_report(require_complete=True)
                metrics.finish()
                terminal_status('complete','Requested saved-success goal reached; closing simulator')
                sys.stdout.flush(); sys.stderr.flush()
                if os.environ.get('P4_SHUTDOWN_MODE')=='verified-exit' and not any(
                        flag in forwarded_args for flag in ('--camera_tuner','--workspace_tuner','--layout_tuner')):
                    from src.recorder.verified_exit import verify_and_exit
                    from phase3_run_metrics import _forwarded_value
                    verify_and_exit(metrics.out_dir,handler.name,
                        _forwarded_value(forwarded_args,('--record_mode',),'both'),
                        int(_forwarded_value(forwarded_args,('--episodes',),'1')))
                return original_close(*args,**kwargs)
            app.close=checked_close
        with metrics.observe_stdout():
            loaded["main"]()
        require_saved_goal(metrics,forwarded_args)
        coverage_report(require_complete=True)
    except BaseException as error:
        if isinstance(error,SystemExit) and error.code in (None,0):
            try:
                require_saved_goal(metrics,forwarded_args)
                coverage_report(require_complete=True)
                terminal_status('complete','Validated saved-success goal reached')
                return
            except RuntimeError as incomplete:
                error = incomplete
        runtime_error = error
        metrics.finish(runtime_error)
        try: coverage_report()
        except Exception as report_error: print('[P4 COVERAGE REPORT ERROR]',repr(report_error),flush=True)
        terminal_status('error',str(error))
        import traceback
        traceback.print_exception(type(error),error,error.__traceback__)
        sys.stdout.flush(); sys.stderr.flush()
        if loaded and loaded.get("simulation_app"):
            env = loaded["main"].__globals__.get("_p4_env")
            try:
                if env is not None:
                    env.close()
                import omni.kit.app
                omni.kit.app.get_app().post_quit(1)
                original_close()
            except BaseException as shutdown_error:
                print('[P4 SHUTDOWN]',repr(shutdown_error),flush=True)
        raise error
    finally:
        metrics.finish(runtime_error)
