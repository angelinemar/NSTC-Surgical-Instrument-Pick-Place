"""P4 shared spawn, camera readiness and stage instrumentation hooks."""


def bounded_settle(settle, env, hold_action, scene_key_for_object, **kwargs):
    """Allow one bounded continuation with unchanged gates and hold action."""
    import math

    def valid_states(report):
        from phase3_settle_contract import SETTLE_OBJECTS
        for name in kwargs.get('names', SETTLE_OBJECTS):
            state = report.get('states', {}).get(name, {})
            if state.get('missing'):
                return False
            for key, size in (('position_w', 3), ('quaternion_wxyz', 4),
                              ('linear_velocity_w', 3), ('angular_velocity_w', 3)):
                values = state.get(key, [])
                if len(values) != size or not all(math.isfinite(float(v)) for v in values):
                    return False
        return True

    first = settle(env, hold_action, scene_key_for_object, **kwargs)
    if not valid_states(first):
        return dict(first, success=False, reason='settle_missing_or_nonfinite_state')
    if first.get('success') or first.get('reason') != 'settle_timeout_or_missing_object':
        return first
    print('[P4 SETTLE CONTINUE] same hold and gates; at most 300 extra steps', flush=True)
    extra = dict(kwargs, min_steps=0)
    # The legacy loop uses max(300, consecutive_ok * poll_every).
    if extra['consecutive_ok'] * extra['poll_every'] > 300:
        return first
    final = settle(env, hold_action, scene_key_for_object, **extra)
    if not valid_states(final):
        final = dict(final, success=False, reason='settle_missing_or_nonfinite_state')
    return dict(final, steps=int(first['steps']) + int(final['steps']),
                continuation_steps=int(final['steps']), initial_reason=first['reason'])


def install_runtime_hooks(namespace):
    from src.recorder.scene_semantics import install
    install(namespace)
    from phase4_fsm import install_fsm
    install_fsm(namespace)
    from phase4_cell_spawn import install_cell_spawn
    install_cell_spawn(namespace)
    if namespace.get('_p4_render_hook_installed'):
        return
    original_settle_all = namespace['settle_all_objects']
    def settle_including_clutter(env, hold, scene_key, **kw):
        from phase3_settle_contract import SETTLE_OBJECTS
        extra = [row['instance'] for row in namespace.get('_p4_clutter',{}).get('instances',[])]
        kw['names'] = tuple(kw.get('names', SETTLE_OBJECTS)) + tuple(extra)
        return bounded_settle(original_settle_all, env, hold,
                              lambda name: name if name in extra else scene_key(name), **kw)
    namespace['settle_all_objects'] = settle_including_clutter
    original_settle = namespace['wait_for_settle']
    original_force = namespace['force_episode_objects']
    original_capture = namespace['capture_object_states']
    namespace['_p4_capture_before'] = original_capture
    def remember_force(env, spawn, pose_mode):
        from src.recorder.capture_contract import storage_budget
        import os
        args=namespace.get('args_cli')
        if args is not None:
            budget=storage_budget(args.out_dir,1,int(os.environ.get('P4_CAMERA_SIZE','448')))
            if not budget['capacity_pass']:
                raise RuntimeError('DISK_RESERVE: stopped before next attempt; saved episodes preserved')
        from src.recorder.domain_randomization import apply_episode_lighting
        from src.recorder.scene_clutter import park
        park(env, namespace)
        apply_episode_lighting(env, namespace)
        namespace.pop('_p4_panel_geometry',None)
        namespace['_p4_force_args'] = (env,spawn,pose_mode)
        namespace['_p4_prepare_clutter_pending'] = True
        from phase4_reset import reset_robot,restore_objects
        reset_robot(env)
        if namespace.get('_p4_spawn_templates'):
            restore_objects(env,namespace,spawn)
            namespace['_p4_prepare_tray_pending']=False
            return
        result = original_force(env,spawn,pose_mode)
        namespace['_p4_prepare_tray_pending'] = True
        return result
    def remember_before(*args, **kwargs):
        result = original_capture(*args,**kwargs)
        namespace['_p4_before_reference'] = result
        return result
    namespace['force_episode_objects'] = remember_force
    # Every body is already handled by force_episode_objects. Several legacy
    # backends call this again afterwards, overwriting prepared tray bodies.
    namespace['force_extra_distractors'] = lambda *args,**kwargs: None
    choose_mode=namespace['choose_scalpel_pose_mode']
    namespace['choose_scalpel_pose_mode']=lambda attempt: namespace.get('_p4_canonical_scalpel_mode',choose_mode(attempt))
    namespace['capture_object_states'] = remember_before

    def settle_then_warm_cameras(env, *args, **kwargs):
        namespace['_p4_env'] = env
        result = original_settle(env, *args, **kwargs)
        if not result.get('success', False):
            return result
        from phase4_reset import cache_templates
        cache_templates(env,namespace)
        if namespace.pop('_p4_prepare_tray_pending',False):
            # Imported assets resolve authored roots to PhysX link frames on
            # the first steps. Only align their measured, settled bodies.
            from phase4_tray_slots import prepare_tray, EvidenceFailure
            first_steps=int(result.get('steps',0))
            try:
                prepare_tray(env,namespace)
            except EvidenceFailure as error:
                return dict(result,success=False,reason=str(error))
            if namespace.get('_p4_tray_objects'):
                fresh=original_capture(env,namespace['phase3_scene_key_for_object'])
                for obj_name in namespace['_p4_tray_objects']:
                    namespace['_p4_before_reference'][obj_name]=fresh[obj_name]
                env.scene.write_data_to_sim()
                result=original_settle(env,*args,**kwargs)
                result=dict(result,steps=first_steps+int(result.get('steps',0)))
                if not result.get('success',False):
                    return result
        if namespace.pop('_p4_prepare_clutter_pending', False):
            from src.recorder.scene_clutter import prepare, validate
            try:
                prepare(env, namespace)
            except RuntimeError as exc:
                if not str(exc).startswith('CLUTTER_'):
                    raise
                print('[P4 SPAWN REJECT]', str(exc), flush=True)
                return dict(result, success=False, reason=str(exc))
            first_steps = int(result.get('steps', 0))
            result = original_settle(env, *args, **kwargs)
            result = dict(result, steps=first_steps + int(result.get('steps', 0)))
            if not result.get('success', False):
                return result
            try:
                validate(env, namespace)
            except RuntimeError as exc:
                if not str(exc).startswith('CLUTTER_'):
                    raise
                print('[P4 SPAWN REJECT]', str(exc), flush=True)
                return dict(result, success=False, reason=str(exc))
        from phase4_cell_spawn import settled_cell_failures
        cell_failures = settled_cell_failures(env,namespace)
        total_settle_steps = int(result.get('steps',0))
        for correction in range(2):
            if not cell_failures:
                break
            settled_cell_failures(env,namespace,recenter=True)
            env.scene.write_data_to_sim()
            result = original_settle(env,*args,**kwargs)
            total_settle_steps += int(result.get('steps',0))
            result = dict(result,steps=total_settle_steps,cell_recenter_passes=correction+1)
            if not result.get('success',False):
                return result
            cell_failures = settled_cell_failures(env,namespace)
        if cell_failures:
            result = dict(result,success=False,cell_fit_failures=cell_failures)
            print('[P4 CELL SPAWN FAIL]',cell_failures,flush=True)
            return result
        if namespace.get('_p4_cell_assignments'):
            print('[P4 CELL FIT OK] table instruments inside their assigned cells',flush=True)
        from phase4_tray_slots import check_preloaded, EvidenceFailure
        try:
            check_preloaded(env,namespace,capture=True)
        except EvidenceFailure as error:
            return dict(result,success=False,reason=str(error))
        # Semantic/depth can be ready before RGB on a newly loaded room.
        # Render without physics before collecting any observation/action.
        names = ('camera','grip_cam_b','cam_top','cam_left','cam_right','cam_tray')
        good = 0
        stats = {}
        import carb.settings
        settings = carb.settings.get_settings()
        print('[P4 RENDER MODE]', env.sim.render_mode, settings.get('/rtx/rendermode'), flush=True)
        for frame in range(60):
            env.sim.render()
            if frame < 19 or frame % 5 != 4:
                continue
            stats = {}
            for name in names:
                camera = env.scene[name]
                camera.update(0.0, force_recompute=True)
                rgb = camera.data.output['rgb'][0, ..., :3].float()
                stats[name] = (float(rgb.mean().item()), float(rgb.std().item()))
            ready = all(mean > 1.0 and std > 1.0 for mean, std in stats.values())
            good = good + 1 if ready else 0
            if good >= 2:
                print(f'[P4 RGB READY] render_frames={frame+1} no_physics_steps; mean/std={stats}', flush=True)
                from src.recorder.scene_semantics import audit_preview
                audit_preview(env, namespace)
                import os
                if os.environ.get('P4_SCENE_AUDIT_ONLY') == '1':
                    import sys
                    print('[P4 SCENE AUDIT COMPLETE] preview and class counts saved; no demonstrations recorded',flush=True)
                    sys.stdout.flush(); sys.stderr.flush()
                    os._exit(0)
                from phase4_session import wait_for_start
                import os
                if os.environ.get('P4_SESSION_CONFIG'):
                    try:
                        from phase4_panel_geometry import capture
                        namespace['_p4_panel_geometry']=capture(env,namespace)
                    except Exception as exc:
                        namespace.pop('_p4_panel_geometry',None)
                        print(f'[P4 PANEL WARNING] Measured outline unavailable: {type(exc).__name__}: {exc}; using calibrated preview.',flush=True)
                if not wait_for_start(env,namespace):
                    return settle_then_warm_cameras(env,*args,**kwargs)
                return result
        # Capture relevant scene state for diagnosing a failed rendering test.
        import json
        from pathlib import Path
        from pxr import Usd, UsdGeom, UsdLux
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        lights = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI):
                lights.append({'path': str(prim.GetPath()), 'intensity': prim.GetAttribute('inputs:intensity').Get(),
                               'visibility': str(UsdGeom.Imageable(prim).ComputeVisibility()),
                               'position': list(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())})
        out = Path(namespace['args_cli'].out_dir)
        cameras = {}
        from PIL import Image
        for name in names:
            camera = env.scene[name]
            outputs = camera.data.output
            Image.fromarray(outputs['rgb'][0, ..., :3].detach().cpu().numpy()).save(out/f'rgb_failure_{name}.png')
            cameras[name] = {
                'position': camera.data.pos_w.detach().cpu().tolist(),
                'outputs': {key: {'shape': list(value.shape),
                                  'min': float(value.float().min().item()),
                                  'max': float(value.float().max().item())}
                            for key, value in outputs.items()},
            }
        (out/'rgb_failure_diagnostics.json').write_text(json.dumps({'lights':lights,'cameras':cameras,'rtx':settings.get('/rtx')}, default=str,indent=2))
        raise RuntimeError(f'P4 RGB not ready after 60 render frames; recording stopped to avoid black dataset: {stats}')

    namespace['wait_for_settle'] = settle_then_warm_cameras
    namespace['_p4_render_hook_installed'] = True
