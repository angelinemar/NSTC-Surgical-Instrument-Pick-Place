"""Seeded episode lighting. Changes occur before settling and RGB warmup only."""
import json
import math
import os
import random


def sample_lighting(seed, layout):
    rng = random.Random(seed)
    # Include table AND tray in the aim region. No moving illumination in a demo.
    target = [(sum(layout['grid_x']) / 2 + layout['tray_xy'][0]) / 2,
              (sum(layout['grid_y']) / 2 + layout['tray_xy'][1]) / 2, 0.0]
    lights = []
    for name, base, phase in [('SharedKeyLight', 5000., 0.), ('SharedFillLight', 2500., math.pi)]:
        angle = rng.uniform(-math.pi, math.pi) + phase
        radius, height = rng.uniform(.25, .65), rng.uniform(1.25, 1.75)
        pos = [target[0] + radius * math.cos(angle), target[1] + radius * math.sin(angle), height]
        aim = [target[0] + rng.uniform(-.08, .08), target[1] + rng.uniform(-.08, .08), 0.]
        lights.append(dict(name=name, position=pos, target=aim,
                           intensity=base * rng.uniform(.75, 1.25),
                           color=[rng.uniform(.85, 1.), rng.uniform(.88, 1.), rng.uniform(.85, 1.)],
                           cone_angle_deg=75.))
    return dict(contract='table_aimed_lighting_v1', seed=seed, lights=lights,
                ambient_intensity=rng.uniform(320., 480.))


def sample_background(seed):
    rng=random.Random(seed+7919)
    palette={'surgical_green':(.045,.24,.13),'deep_green':(.035,.18,.10),'teal_green':(.045,.25,.19)}
    name=rng.choice(tuple(palette))
    return dict(profile=name,color=[min(1.,v*rng.uniform(.8,1.2)) for v in palette[name]],
                roughness=rng.uniform(.65,.95),physical_scale=1.,
                scope='selected_table_drape_only; geometry and instrument scale unchanged')


def apply_background(stage, layout, config):
    """Tint only the selected table drape. Never alter collision or instrument materials."""
    from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
    root=stage.GetPrimAtPath('/World/envs/env_0/HospitalRoom/Furniture/'+layout['selected_table'])
    if not root.IsValid(): raise RuntimeError('Missing selected table for appearance variation')
    # Referenced table visuals contain nested instances; make ONLY the drape
    # branch editable, not the hospital or instrument assets.
    while True:
        instance=next((p for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies())
                       if p.IsInstance() and ('SurgicalInstrumentTable' in p.GetName()
                           or 'Cloth' in p.GetName() or 'GreenDrape' in p.GetName())),None)
        if instance is None: break
        instance.SetInstanceable(False)
    meshes=[p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh) and 'Green_Drape' in p.GetName()]
    if not meshes: raise RuntimeError('No editable table drape mesh; refusing silent randomization omission')
    material=UsdShade.Material.Define(stage,'/World/P4RandomizedDrape')
    shader=UsdShade.Shader.Define(stage,'/World/P4RandomizedDrape/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*config['color']))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(config['roughness'])
    shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    for mesh in meshes:
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    config['mesh_paths']=[str(p.GetPath()) for p in meshes]


def apply_episode_lighting(env, ns):
    if os.environ.get('P4_RANDOMIZATION', 'train') == 'off':
        ns['_p4_domain_randomization'] = dict(contract='disabled')
        return
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux
    from phase4_scene import LAYOUT
    from phase4_tray_slots import ORDER
    session_seed = int(os.environ.get('P4_RANDOMIZATION_SEED', '17'))
    seed = session_seed
    attempt = int(ns.get('_p4_attempt_number', 1))
    seed += attempt * 1009 + ORDER.index(ns['PHASE3_TARGET_OBJECT']) * 1000003
    config = sample_lighting(seed, LAYOUT)
    config.update(session_seed=session_seed, attempt=attempt)
    stage = omni.usd.get_context().get_stage()
    config['background']=sample_background(seed)
    apply_background(stage,LAYOUT,config['background'])
    for item in config['lights']:
        prim = stage.GetPrimAtPath('/World/envs/env_0/' + item['name'])
        if not prim.IsValid():
            raise RuntimeError('Missing randomization light: ' + item['name'])
        direction = Gf.Vec3d(*item['target']) - Gf.Vec3d(*item['position'])
        rotation = Gf.Rotation(Gf.Vec3d(0., 0., -1.), direction.GetNormalized())
        transform = UsdGeom.Xformable(prim)
        matrix = Gf.Matrix4d(1.)
        matrix.SetRotate(rotation)
        matrix.SetTranslateOnly(Gf.Vec3d(*item['position']))
        transform.MakeMatrixXform().Set(matrix)
        api = UsdLux.LightAPI(prim)
        api.GetIntensityAttr().Set(item['intensity'])
        api.GetColorAttr().Set(Gf.Vec3f(*item['color']))
        shaping = UsdLux.ShapingAPI.Apply(prim)
        shaping.CreateShapingConeAngleAttr().Set(item['cone_angle_deg'])
        shaping.CreateShapingConeSoftnessAttr().Set(.3)
    ambient = stage.GetPrimAtPath('/World/SharedAmbientLight')
    if not ambient.IsValid():
        raise RuntimeError('Missing ambient light')
    UsdLux.LightAPI(ambient).GetIntensityAttr().Set(config['ambient_intensity'])
    ns['_p4_domain_randomization'] = config
    print('[P4 EPISODE LIGHTING] ' + json.dumps(config), flush=True)
