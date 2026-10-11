"""Trial 02: deterministic, configuration-disjoint hemisphere/PBR capture."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SPLITS=('train','valid','test')
SEEDS=dict(train=1000000000,valid=2000000000,test=3000000000)
LIGHT_MASKS=dict(train=(3,5,6,7,9,10,11,12,13,14,15),valid=(1,2),test=(4,8))
AZIMUTH_BANDS=dict(train=(1.,35.),valid=(37.,47.),test=(49.,59.))


def load_protocol(path):
    config=json.loads(Path(path).read_text())
    assert config['contract']=='nstc_seg_hemisphere_v1'
    assert config['split'] in SPLITS
    assert config['resolution']>=384 and config['resolution']%24==0
    assert config['seed']==SEEDS[config['split']]
    reservation=Path(config['dp_reservation'])
    actual=hashlib.sha256(reservation.read_bytes()).hexdigest()
    if actual!=config['dp_reservation_sha256']:raise ValueError('DP viewpoint reservation changed')
    config['reserved_eyes']=json.loads(reservation.read_text())['world_positions']
    return config


def hemisphere_views(rng,layout,count,config):
    center=np.array([sum(layout['grid_x'])/2,sum(layout['grid_y'])/2,.035])
    reserved=np.asarray(config['reserved_eyes'],float)
    low,high=AZIMUTH_BANDS[config['split']]
    result=[]
    for i in range(count):
        for attempt in range(10000):
            az=math.radians(60*rng.randrange(6)+rng.uniform(low,high))
            # Uniform solid-angle sampling within this hemisphere elevation band.
            el=math.asin(rng.uniform(math.sin(math.radians(15)),math.sin(math.radians(85))))
            band=i%3
            distance=rng.uniform(*[(.4,.65),(.7,1.1),(1.2,1.85)][band])
            eye=center+distance*np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])
            if eye[2]<.18:continue
            clearance=float(np.linalg.norm(reserved-eye,axis=1).min()) if len(reserved) else float('inf')
            # Covers the 2cm reservation quantization error as well as the margin.
            if clearance<config['dp_exclusion_radius_m']+.02:continue
            result.append(dict(name=f'hemisphere_{i:02d}',eye=eye.tolist(),target=center.tolist(),
                azimuth_deg=math.degrees(az)%360,elevation_deg=math.degrees(el),distance_m=distance,
                distance_band=('close','medium','far')[band],dp_clearance_m=clearance))
            break
        else:raise RuntimeError('Cannot sample a camera outside DP reservation')
    return result


def lighting(rng,layout,split):
    center=[sum(layout['grid_x'])/2,sum(layout['grid_y'])/2,0.]
    mask=rng.choice(LIGHT_MASKS[split]);lights=[]
    for i,name in enumerate(('SharedKeyLight','SharedFillLight','SegRimLight','SegBackLight')):
        angle=i*math.pi/2+rng.uniform(-.6,.6);radius=rng.uniform(.35,1.4)
        enabled=bool(mask&(1<<i))
        lights.append(dict(name=name,enabled=enabled,
            position=[center[0]+radius*math.cos(angle),center[1]+radius*math.sin(angle),rng.uniform(.7,2.2)],
            target=center,intensity=math.exp(rng.uniform(math.log(1000),math.log(10000))) if enabled else 0.,
            color=[rng.uniform(.8,1.) for _ in range(3)],cone_angle_deg=rng.uniform(55,85)))
    return dict(contract='seg_four_light_holdout_v1',split=split,enabled_mask=mask,lights=lights,ambient_intensity=rng.uniform(120,500))


def install_lighting(config):
    import random
    import src.recorder.domain_randomization as module
    module.sample_lighting=lambda seed,layout:dict(lighting(random.Random(seed),layout,config['split']),seed=seed)


def ensure_lights(stage):
    from pxr import UsdLux
    for name in ('SegRimLight','SegBackLight'):
        light=UsdLux.SphereLight.Define(stage,'/World/envs/env_0/'+name)
        light.CreateRadiusAttr(.12);light.CreateIntensityAttr(0.)


def encode_rle(mask):
    """Uncompressed COCO RLE preserves disconnected components and holes exactly."""
    flat=np.asarray(mask,dtype=np.uint8).ravel(order='F')
    edges=np.flatnonzero(np.diff(flat))+1
    counts=np.diff(np.r_[0,edges,len(flat)]).tolist()
    if len(flat) and flat[0]:counts.insert(0,0)
    return dict(size=list(mask.shape),counts=counts)
