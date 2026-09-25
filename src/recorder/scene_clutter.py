"""Extra rigid instances, independent of the five legacy grasp-handler bodies."""
import os
import random


def limits():
    from phase4_session import load_session
    cfg = load_session() or {}
    lo = int(cfg.get('distractor_min', os.environ.get('P4_DISTRACTOR_MIN', 12)))
    hi = int(cfg.get('distractor_max', os.environ.get('P4_DISTRACTOR_MAX', 18)))
    if not 4 <= lo <= hi <= 30:
        raise ValueError('Distractors must satisfy 4 <= minimum <= maximum <= 30')
    return lo, hi


def pool(target):
    from phase4_tray_slots import ORDER
    rng = random.Random(int(os.environ.get('P4_RANDOMIZATION_SEED', 17)) + 4381)
    eligible = [n for n in ORDER if n != target]
    return [(f'clutter_{i:02d}', rng.choice(eligible)) for i in range(limits()[1] - 4)]


def add_to_scene(scene, target, rigid_cfg):
    for i, (key, name) in enumerate(pool(target)):
        cfg = rigid_cfg(name, prim_name=f'Clutter_{i:02d}_{name}')
        cfg.init_state.pos = (3. + i * .3, 3., -2.)
        setattr(scene, key, cfg)


def disjoint(a, b, margin=.008):
    return any(a[1][i] + margin < b[0][i] or b[1][i] + margin < a[0][i] for i in range(2))


def table_footprint_allowed(points, layout, margin=.005):
    """Reserve the entire tray, including empty lanes, for fixed-slot objects."""
    import numpy as np
    from phase4_tray_slots import axes
    down, right = axes(layout)
    center = np.asarray(layout['tray_xy'])
    length, width = layout['tray_dimensions_local_xy']
    half = np.abs(down[:2]) * length / 2 + np.abs(right[:2]) * width / 2
    bounds = (points[:, :2].min(0), points[:, :2].max(0))
    return (bool(np.all(bounds[0] >= [layout['grid_x'][0]+margin, layout['grid_y'][0]+margin]))
            and bool(np.all(bounds[1] <= [layout['grid_x'][1]-margin, layout['grid_y'][1]-margin]))
            and disjoint(bounds, (center-half, center+half), margin=.012))


def park(env, ns):
    import torch
    for i, (key, _) in enumerate(pool(ns['PHASE3_TARGET_OBJECT'])):
        obj = env.scene[key]
        pose = torch.tensor([[3.+i*.3,3.,-2.,1.,0.,0.,0.]],device=env.device,dtype=obj.data.root_state_w.dtype)
        obj.write_root_link_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_state_w[:,7:13]))
    ns['_p4_clutter'] = {}


def prepare(env, ns):
    """Place extra bodies only on the table; never use tray lanes for clutter."""
    import torch
    from phase4_scene import LAYOUT
    from phase4_grasp_validation import rigid_world_points, rotate
    from phase4_reset import table_pose
    from phase4_tray_slots import ORDER
    target = ns['PHASE3_TARGET_OBJECT']
    seed = int(os.environ.get('P4_RANDOMIZATION_SEED', 17)) + 1009 * int(ns.get('_p4_attempt_number', 1))
    rng = random.Random(seed + 2753)
    count = rng.randint(*limits()) - 4
    active = set(rng.sample([key for key, _ in pool(target)], count))
    occupied = []
    for name in ORDER:
        points = rigid_world_points(env.scene[ns['phase3_scene_key_for_object'](name)])
        margin = .035 if name == target else .005
        occupied.append((points[:, :2].min(0) - margin, points[:, :2].max(0) + margin))
    rows = []
    for i, (key, name) in enumerate(pool(target)):
        obj = env.scene[key]
        if key not in active:
            pose = [3. + i*.3, 3., -2., 1., 0., 0., 0.]
        else:
            template = ns['_p4_spawn_templates'][name]
            chosen = None
            for trial in range(2500):
                yaw = rng.uniform(-180., 180.)
                xy = [rng.uniform(*LAYOUT['grid_x']), rng.uniform(*LAYOUT['grid_y'])]
                candidate = table_pose(template, *xy, yaw)
                points = rotate(candidate[3:], template['points']) + candidate[:3]
                if not table_footprint_allowed(points, LAYOUT):
                    continue
                bounds = (points[:,:2].min(0), points[:,:2].max(0))
                if all(disjoint(bounds, previous) for previous in occupied):
                    occupied.append(bounds)
                    chosen = candidate.tolist()
                    break
            if chosen is None:
                raise RuntimeError(f'CLUTTER_PACKING_FAILED: {count+4} distractors cannot fit safely; no recording saved')
            pose = chosen
            rows.append(dict(instance=key, object=name, region='table', requested_pose=pose))
        obj.write_root_link_pose_to_sim(torch.tensor([pose], device=env.device, dtype=obj.data.root_state_w.dtype))
        obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_state_w[:,7:13]))
        obj.update(env.physics_dt)
    ns['_p4_clutter'] = dict(contract='table_only_distractors_v2', seed=seed, distractor_count=count+4,
                              target_count=1, instances=rows)
    ns['_p4_force_args'][1]['grid'].update(requested_num_distractors=count+4,
                                          actual_num_distractors=count+4)
    env.scene.write_data_to_sim()
    env.sim.forward()
    print(f'[P4 CLUTTER] target=1 distractors={count+4} extra_instances={len(rows)}', flush=True)


def validate(env, ns):
    """Reject escaped, moving or unsupported clutter after physics settling."""
    import numpy as np
    from phase4_grasp_validation import rigid_world_points
    from phase4_scene import LAYOUT
    for item in ns.get('_p4_clutter', {}).get('instances', []):
        if item['region'] != 'table':
            raise RuntimeError('CLUTTER_TRAY_FORBIDDEN: ' + item['instance'])
        obj = env.scene[item['instance']]
        points = rigid_world_points(obj)
        # Rotation stability is already checked from observed pose deltas by
        # settle_all_objects; imported meshes have noisy raw angular velocities.
        if not np.isfinite(points).all() or float(obj.data.root_lin_vel_w[0].norm()) > .01:
            raise RuntimeError('CLUTTER_NOT_SETTLED: ' + item['instance'])
        if not table_footprint_allowed(points, LAYOUT, margin=0.) or abs(float(points[:,2].min())) > .008:
            raise RuntimeError('CLUTTER_SUPPORT_FAILED: ' + item['instance'])
        item['settled_pose'] = obj.data.root_link_state_w[0,:7].detach().cpu().tolist()
