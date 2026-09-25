"""Read-only scene preview through the same P4 scalpel environment builder.

No episodes are collected; only the camera tuner can persist camera edits.
"""
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cameras', action='store_true')
    parser.add_argument('--camera-save-file',type=Path,help='Optional separate camera tuning output; defaults to active camera_layout.json')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--capture-dir')
    parser.add_argument('--diagnose-rgb', action='store_true')
    parser.add_argument('--lighting-compare', action='store_true', help='Capture current and former P3 light levels in the same pose; restore current afterward')
    parser.add_argument('--probe-recording', action='store_true')
    parser.add_argument('--motion-check', action='store_true', help='Diagnostic: execute unchanged scalpel pick/place; do not save a training episode')
    parser.add_argument('--pose-h5', type=Path, help='Diagnostic: replay final robot joint state from an H5 segment')
    args = parser.parse_args()
    backend = ROOT / 'backends' / 'phase3_grid_split_scalpel_recorder.py'
    sys.argv = [str(backend), '--episodes', '0', '--out_dir', str(ROOT/'validation'/'preview'),
                '--camera_tuner' if args.cameras else '--layout_tuner']
    if args.headless:
        sys.argv.append('--headless')
    module = runpy.run_path(str(backend), run_name='p4_preview_backend')
    scope = module['main'].__globals__
    # Tuning must use the recorder's native sensor, then the SAME pixel crop.
    original_camera_cfg = scope['phase3_apply_final_cameras_to_env_cfg']
    scope['phase3_apply_final_cameras_to_env_cfg'] = lambda cfg, **kwargs: original_camera_cfg(cfg, preview_output_crop=False)
    from phase3_shared_env_cfg import RecorderEnvRequest
    scope['RecorderEnvRequest'] = lambda **kwargs: RecorderEnvRequest(
        target='scalpel', distractors=('scissor','love_retractor','kelly','scalpel_type2'),
        environment=(), use_canonical_target_only=True)

    def prepare_scene(env):
        import numpy as np
        env.reset()
        rng = np.random.default_rng(scope['RANDOM_SEED'])
        spawn = scope['sample_episode_spawn_grid'](rng, 2, 'scalpel')
        spawn = scope['ensure_two_distractors'](spawn, 'scalpel', rng)
        spawn = scope['phase3_ensure_all_5_objects_in_spawn'](spawn, 'scalpel', rng)
        scope['force_episode_objects'](env, spawn, scope['choose_scalpel_pose_mode'](2))
        try:
            scope['wait_for_settle'](env)
        except RuntimeError as error:
            if args.probe_recording or 'P4 RGB not ready' not in str(error):
                raise
            print('[P4 PREVIEW WARNING] RGB readiness failed. Continue inspection only; no training episode will be saved.', flush=True)
        if args.pose_h5:
            import h5py
            import torch
            with h5py.File(args.pose_h5, 'r') as f:
                joints = f['observations/state'][-1, :9]
            robot = env.scene['robot']
            q = torch.tensor(joints, device=env.device).reshape(1,9)
            assert robot.data.joint_pos.shape[1] == 9
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            env.scene.write_data_to_sim()
            env.sim.forward()
            print('[P4 POSE REPLAY]', args.pose_h5, joints.tolist(), flush=True)

    original_camera_tuner = scope['run_camera_layout_tuner']

    def camera_tuner(env, stage):
        prepare_scene(env)
        from phase4_camera_preview import tune_cameras
        tune_cameras(env,stage,scope,frames=args.frames,save_file=args.camera_save_file)

    scope['run_camera_layout_tuner'] = camera_tuner

    def preview(env, stage, save_path):
        from pxr import UsdGeom
        from phase4_scene import LAYOUT
        import carb.settings
        settings = carb.settings.get_settings()
        for setting in ('/rtx/rendermode', '/rtx/modes/rt/enabled', '/persistent/rtx/modes/rt/enabled',
                        '/rtx/post/tonemap/op', '/rtx/post/tonemap/filmIso', '/rtx/post/tonemap/cameraShutter',
                        '/rtx/post/tonemap/fNumber', '/rtx/post/histogram/enabled'):
            print('[P4 RENDER SETTING]', setting, settings.get(setting), flush=True)
        # Show the actual five-object spawn procedure, not legacy initial poses.
        prepare_scene(env)
        if args.motion_check:
            import torch
            output = Path(args.capture_dir or ROOT/'validation'/'motion_preview')
            recorder = scope['EpisodeRecorder'](str(output/'diagnostic_only'), 'P4 layout motion check; not training data', 25)
            slot = torch.tensor(scope['TRAY_FIXED_POS'], device=env.device) + torch.tensor(scope['SCALPEL_SLOT_OFFSET'], device=env.device)
            result = scope['pick_and_place_object'](
                env, recorder, stage, 'SCALPEL', 0,
                scope['SCALPEL_GRASP_Z_ABOVE_TABLE'], scope['SCALPEL_BODY_OFFSET_X'],
                scope['SCALPEL_BODY_OFFSET_Y'], scope['SCALPEL_BODY_OFFSET_Z'],
                scope['get_scalpel_pos_w'], scope['get_scalpel_quat_w'], slot,
                torch.tensor([0.,0.7071,0.7071,0.], device=env.device),
                scalpel_pose_mode=scope['choose_scalpel_pose_mode'](2))
            ok, max_z, close_z, final_pos = result
            error = torch.linalg.norm((final_pos-slot)[:2]).item()
            passed = bool(ok and max_z>close_z+0.06 and error<0.09 and float(final_pos[2])<0.12)
            print('[P4 MOTION CHECK]', {'success':passed, 'steps':len(recorder.actions), 'xy_error_m':error, 'max_z':max_z, 'close_z':close_z, 'final_pos':final_pos.tolist()}, flush=True)
            if not passed:
                raise RuntimeError('P4 diagnostic motion check failed; no training episode saved')
        if LAYOUT.get('show_grid_in_preview', True):
            gx, gy = LAYOUT['grid_x'], LAYOUT['grid_y']
            nx, ny = LAYOUT['grid_cols'], LAYOUT['grid_rows']
            points = []
            for i in range(nx+1):
                x = gx[0] + (gx[1]-gx[0])*i/nx
                points.extend([(x,gy[0],0.01),(x,gy[1],0.01)])
            for i in range(ny+1):
                y = gy[0] + (gy[1]-gy[0])*i/ny
                points.extend([(gx[0],y,0.01),(gx[1],y,0.01)])
            lines = UsdGeom.BasisCurves.Define(stage, '/World/P4PreviewGrid')
            lines.CreateTypeAttr('linear')
            lines.CreateCurveVertexCountsAttr([2]*(nx+ny+2))
            lines.CreatePointsAttr(points)
            lines.CreateWidthsAttr([0.003])
            lines.CreateDisplayColorAttr([(0.05,0.9,0.25)])
        print('[P4 PREVIEW] No recording and no layout autosave. Close window or Ctrl+C to exit.', flush=True)
        print('[P4 PREVIEW] Configure scene_layout.json before next launch; camera_layout.json owns cameras.', flush=True)
        frame = 0
        panel = None
        if not args.headless:
            from phase4_camera_preview import RecorderPreview
            panel = RecorderPreview(env,scope)
        try:
            while scope['simulation_app'].is_running() and (args.frames <= 0 or frame < args.frames):
                env.sim.render()
                frame += 1
            if args.capture_dir:
                from PIL import Image
                output = Path(args.capture_dir)
                output.mkdir(parents=True, exist_ok=True)
                def capture(label):
                    for _ in range(20):
                        env.sim.render()
                    for name in ('camera','grip_cam_b','cam_top','cam_left','cam_right','cam_tray'):
                        env.scene[name].update(0.0, force_recompute=True)
                        rgb, _ = scope['read_camera_rgb_depth'](env, name)
                        if rgb is not None:
                            print('[P4 RGB]', label, name, rgb.shape, rgb.min(), rgb.max(), float(rgb.mean()), flush=True)
                            h, w = rgb.shape[:2]
                            Image.fromarray(rgb).save(output/f'{label}_{name}.png')
                capture('baseline')
                if args.lighting_compare:
                    paths = {'/World/SharedAmbientLight':1800.,
                             '/World/envs/env_0/SharedKeyLight':45000.,
                             '/World/envs/env_0/SharedFillLight':22000.}
                    attrs = [(stage.GetPrimAtPath(path).GetAttribute('inputs:intensity'), value)
                             for path,value in paths.items()]
                    saved = [(attr,attr.Get()) for attr,_ in attrs]
                    try:
                        for attr,value in attrs:
                            attr.Set(value)
                        capture('old_lighting')
                    finally:
                        for attr,value in saved:
                            attr.Set(value)
                if args.probe_recording:
                    recorder = scope['EpisodeRecorder'](str(output/'record_probe'), 'P4 diagnostic only', 1)
                    for i in range(6):
                        target = scope['get_ee_pos_w'](env).clone()
                        quat = scope['get_ee_quat_w'](env).clone()
                        state = scope['make_state'](env, target, 0)
                        action = scope['make_abs_action'](env, target, quat, 1.0)
                        recorder.add_step(env, 'SCALPEL_OPEN_HOVER', i, state, action)
                        print('[P4 RECORD PROBE]', i, 'storedRGB', recorder.front_rgb[-1].min(), recorder.front_rgb[-1].max(), float(recorder.front_rgb[-1].mean()), flush=True)
                        env.step(action)
                    print('[P4 RECORD PROBE AFTER STEPS]', [float(a.mean()) for a in recorder.front_rgb], flush=True)
                    scope['save_realcompat_segment'](recorder, 0, 'pick', str(output/'record_probe'), True, 'scalpel', meta={})
                    capture('after_steps')
                if args.diagnose_rgb:
                    # Temporary A/B checks; never save these edits or settings.
                    original_lights = settings.get('/rtx/raytracing/showLights')
                    settings.set('/rtx/raytracing/showLights', 2)
                    capture('lights_hidden')
                    settings.set('/rtx/raytracing/showLights', original_lights)
                    room = UsdGeom.Imageable(stage.GetPrimAtPath('/World/envs/env_0/HospitalRoom'))
                    room.MakeInvisible()
                    capture('without_room')
                    room.MakeVisible()
                    original = settings.get('/rtx/rendermode')
                    settings.set('/rtx/rendermode', 'PathTracing')
                    capture('pathtracing')
                    settings.set('/rtx/rendermode', original)
                print('[P4 PREVIEW CAPTURE]', output, flush=True)
        except KeyboardInterrupt:
            print('[P4 PREVIEW] Closed; layout unchanged.', flush=True)
        finally:
            if panel is not None:
                panel.close()

    scope['run_shared_layout_tuner'] = preview
    try:
        scope['main']()
    except BaseException:
        env = scope.get('_p4_env')
        if env is not None:
            env.close()
        scope['simulation_app'].close()
        raise


if __name__ == '__main__':
    main()
