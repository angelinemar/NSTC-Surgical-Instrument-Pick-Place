"""Display the recorder's actual 224x224 sensor crop, not a resized viewport."""
import time
import json
from pathlib import Path
from phase4_camera_names import public_name

CAMERAS = ('camera','grip_cam_b','cam_top','cam_left','cam_right','cam_tray')
PATHS = dict(zip(CAMERAS, (
    '/World/envs/env_0/cam_front', '/World/envs/env_0/Robot/panda_hand/GripCamB_Final',
    '/World/envs/env_0/CamTop', '/World/envs/env_0/CamLeft',
    '/World/envs/env_0/CamRight', '/World/envs/env_0/CamTray')))


def recorded_crop(env, scope, name):
    import numpy as np
    env.scene[name].update(0.0, force_recompute=True)
    rgb,_ = scope['read_camera_rgb_depth'](env,name,rotate_grip_b=False)
    h,w = rgb.shape[:2]
    if min(h,w)<224:
        raise ValueError(f'{name}: sensor resolution smaller than recorder crop')
    return np.ascontiguousarray(rgb[(h-224)//2:(h-224)//2+224,(w-224)//2:(w-224)//2+224,:3])


class RecorderPreview:
    def __init__(self, env, scope):
        import omni.ui as ui
        import omni.kit.app
        self.env,self.scope = env,scope
        self.providers = {}
        self.last_update = 0.0
        self.warning = False
        self.uploaded = False
        self.window = ui.Window('P4 Recording RGB | 224 x 224 | LIVE',width=710,height=550)
        with self.window.frame:
            with ui.VStack(spacing=4):
                ui.Label('Actual sensor center crop; no resize. Wrist = gripper.',height=20)
                for row in range(2):
                    with ui.HStack(height=248,spacing=8):
                        for name in CAMERAS[row*3:row*3+3]:
                            with ui.VStack(width=224):
                                ui.Label(f'{public_name(name)} | 224 x 224',height=20)
                                provider = ui.ByteImageProvider()
                                self.providers[name] = provider
                                ui.ImageWithProvider(provider,width=224,height=224,pixel_aligned=True,
                                    fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
        self.subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self.update,name='P4 recorder RGB crop preview')
        print('[P4 EXACT PREVIEW] six 224x224 sensor crops, native camera sizes unchanged',flush=True)

    def update(self, event=None):
        if not self.window.visible or time.monotonic()-self.last_update<0.15:
            return
        self.last_update=time.monotonic()
        try:
            import numpy as np
            for name,provider in self.providers.items():
                rgb=recorded_crop(self.env,self.scope,name)
                rgba=np.concatenate((rgb,np.full((224,224,1),255,dtype=np.uint8)),axis=2)
                provider.set_bytes_data(rgba.reshape(-1).tolist(),[224,224])
            if not self.uploaded:
                print('[P4 EXACT PREVIEW FRAME] 6 RGB uploads; each shape=(224,224,3); no resize',flush=True)
                self.uploaded=True
        except Exception as error:
            if not self.warning:
                print('[P4 EXACT PREVIEW WARN]',repr(error),flush=True)
                self.warning=True

    def close(self):
        self.subscription=None
        self.window.visible=False


def tune_cameras(env, stage, scope, frames=0, save_file=None):
    from isaaclab.utils.math import convert_camera_frame_orientation_convention
    import torch
    from phase3_shared_env_cfg import WORKSPACE
    panel=RecorderPreview(env,scope)
    destination=Path(save_file) if save_file else Path(__file__).parent / 'env' / 'camera_layout.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    def save():
        payload={'format_version':1,'preview':'224x224_h5_center_crop','cameras':{}}
        for name,path in PATHS.items():
            if not stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f'Missing camera: {path}; layout not saved')
            pose=scope['_layout_local_pose'](stage,path)
            pos=pose['pos']
            if name!='grip_cam_b':
                pos=[pos[i]-WORKSPACE.offset[i] for i in range(3)]
            quat=torch.tensor(pose['rot'],dtype=torch.float32,device=env.device).reshape(1,4)
            rot=convert_camera_frame_orientation_convention(quat,origin='opengl',target='ros')[0].cpu().tolist()
            payload['cameras'][public_name(name)]={'pos':pos,'rot':rot}
        tmp=destination.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        tmp.replace(destination)
        print('[P4 CAMERA SAVED]',destination,flush=True)
    print('[P4 CAMERA TUNER] Edit camera prim XYZ/rotation; check the Recording RGB panel, not the main viewport dimensions.',flush=True)
    print('[P4 CAMERA TUNER] Camera poses autosave every 2 seconds; Ctrl+C makes final save. No Ctrl+S needed.',flush=True)
    count=0; last_save=time.monotonic()
    try:
        while scope['simulation_app'].is_running() and (frames<=0 or count<frames):
            env.sim.render();count+=1
            if time.monotonic()-last_save>2:
                save();last_save=time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        save();panel.close()
