"""Expert-only settled scene outlines for the GUI, never training inputs."""
def capture(env,ns):
    import numpy as np
    from scipy.spatial import ConvexHull
    from phase4_grasp_validation import rigid_world_points
    from phase4_tray_slots import ORDER,object_axis
    result={}
    for name in ORDER:
        obj=env.scene[ns['phase3_scene_key_for_object'](name)]
        p=rigid_world_points(obj)
        hull=p[ConvexHull(p[:,:2]).vertices,:2]
        result[name]=dict(hull_xy=hull.tolist(),center_xy=((p.min(0)+p.max(0))/2)[:2].tolist(),
                         axis_xy=object_axis(obj)[:2].tolist(),source='measured after settle')
    return result
