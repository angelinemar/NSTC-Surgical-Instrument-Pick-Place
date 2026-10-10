"""Detector-only wide randomization. Values are recorded for exact resume."""
import copy
import json
import math
import random

DEFAULT_WIDE = dict(light_radius=[.1,1.4],light_height=[.65,2.4],
    light_intensity=[800.,12000.],ambient_intensity=[100.,900.],
    light_color=[.55,1.],aim_jitter=.25,cone_angle=[50.,90.],
    camera_elevation=[25.,75.],camera_distance=[1.,1.2])


def load_profile(path=None):
    config=copy.deepcopy(DEFAULT_WIDE)
    if path:
        custom=json.loads(path.read_text())
        if not isinstance(custom,dict) or set(custom)-set(config):
            raise ValueError('Unknown wide randomization settings')
        config.update(custom)
    limits=dict(light_radius=(0.,3.),light_height=(.3,4.),light_intensity=(1.,30000.),
        ambient_intensity=(1.,3000.),light_color=(.1,1.),cone_angle=(20.,90.),
        camera_elevation=(20.,85.),camera_distance=(1.,1.5))
    for key,(low,high) in limits.items():
        value=config[key]
        if not isinstance(value,list) or len(value)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in value) or not low<=value[0]<=value[1]<=high:
            raise ValueError(f'{key} must be [min,max] within {low}..{high}')
    if not isinstance(config['aim_jitter'],(int,float)) or not 0<=config['aim_jitter']<=.5:
        raise ValueError('aim_jitter must be 0..0.5 metres')
    return config


def sample_wide_lighting(seed,layout,config):
    rng=random.Random(seed)
    target=[(sum(layout['grid_x'])/2+layout['tray_xy'][0])/2,
            (sum(layout['grid_y'])/2+layout['tray_xy'][1])/2,0.]
    lights=[]
    for name in ('SharedKeyLight','SharedFillLight'):
        angle=rng.uniform(-math.pi,math.pi)
        radius=rng.uniform(*config['light_radius'])
        lights.append(dict(name=name,position=[target[0]+radius*math.cos(angle),target[1]+radius*math.sin(angle),rng.uniform(*config['light_height'])],
            target=[target[0]+rng.uniform(-config['aim_jitter'],config['aim_jitter']),target[1]+rng.uniform(-config['aim_jitter'],config['aim_jitter']),0.],
            intensity=math.exp(rng.uniform(*(math.log(x) for x in config['light_intensity']))),
            color=[rng.uniform(*config['light_color']) for _ in range(3)],cone_angle_deg=rng.uniform(*config['cone_angle'])))
    return dict(contract='detector_wide_lighting_v1',seed=seed,lights=lights,ambient_intensity=rng.uniform(*config['ambient_intensity']))


def install_wide_lighting(config):
    # This is an isolated detector process; never modify DP's default sampler.
    import src.recorder.domain_randomization as lighting
    lighting.sample_lighting=lambda seed,layout: sample_wide_lighting(seed,layout,config)
