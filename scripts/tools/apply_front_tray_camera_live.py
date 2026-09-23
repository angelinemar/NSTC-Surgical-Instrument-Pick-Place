"""Run inside the OPEN P4 Isaac Sim Script Editor, not a new Python process.

Change only front/tray camera poses. Preserve optics, render size, other cameras,
scene and recorder logic. Existing tuner autosave reads these new USD poses.
"""
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
import omni.usd
from pxr import Gf, UsdGeom

ROOT = Path('C:/IsaacLab/scripts/custom/i4h_project/p4')
POSES = {
    'camera': {
        'pos':[1.5338732585526496,-0.20257852835859724,1.127255031448907],
        'rot':[-0.3238106369972229,0.6220670342445374,0.6303048133850098,-0.3330092132091522],
    },
    'cam_tray': {
        'pos':[0.15,-0.89,0.8842166533715959],
        'rot':[0.,0.7071067811865476,0.7071067811865476,0.],
    },
}
PATHS = {'camera':'/World/envs/env_0/cam_front','cam_tray':'/World/envs/env_0/CamTray'}


def apply():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError('Open the P4 camera tuner first')
    if not stage.GetPrimAtPath('/World/envs/env_0/HospitalRoom/HospitalScene').IsValid():
        # References may compose below the default prim without its name.
        room = stage.GetPrimAtPath('/World/envs/env_0/HospitalRoom')
        if not room.IsValid() or 'hospital_isaac.usdz' not in str(room.GetMetadata('references')):
            raise RuntimeError('This is not the expected P4 hospital stage; no cameras changed')
    for name,path in PATHS.items():
        prim=stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f'Missing camera {path}; no cameras changed')
        if not math.isclose(sum(v*v for v in POSES[name]['rot']),1.,abs_tol=1e-8):
            raise RuntimeError('Invalid quaternion')
    active=ROOT/'env'/'camera_layout.json'
    data=json.loads(active.read_text(encoding='utf-8'))
    backup=ROOT/'archive'/('camera_before_front_tray_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True)
    shutil.copy2(active,backup/'camera_layout.json')
    for name,pose in POSES.items():
        # Saved ROS optical frame (+Z forward,+Y down) -> USD/OpenGL camera
        # (-Z forward,+Y up): right-multiply quaternion by Rx(pi).
        w,x,y,z=pose['rot']
        gl=Gf.Quatd(-x,Gf.Vec3d(w,z,-y))
        matrix=Gf.Matrix4d(1.)
        matrix.SetRotate(gl)
        matrix.SetTranslateOnly(Gf.Vec3d(*pose['pos']))
        UsdGeom.Xformable(stage.GetPrimAtPath(PATHS[name])).MakeMatrixXform().Set(matrix)
        key='cam_front' if name=='camera' else name
        if name=='camera':
            data['cameras'].pop('camera',None)
        data['cameras'].setdefault(key,{}).update(pose)
        print('[P4 CAMERA APPLIED]',name,'pos=',pose['pos'],'rot_ros_wxyz=',pose['rot'])
    pending=active.with_suffix('.live_update.tmp')
    pending.write_text(json.dumps(data,indent=2),encoding='utf-8')
    pending.replace(active)
    print('[P4 FRONT/TRAY SAVED]',active)
    print('Front: elevated FRONT view. Tray: centered vertical TOP-DOWN.')
    print('Recording stays native448x336 -> center224x224, no resize. Check live RGB panel.')
    print('Tray long side ~216px; ~4px margin at either end. Rectangle cannot fill a square in both axes.')
    print('Other cameras, robot, table, lights, instrument logic unchanged. Backup:',backup)


apply()
