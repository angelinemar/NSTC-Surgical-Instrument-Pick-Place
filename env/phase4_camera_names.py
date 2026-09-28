"""Public P4 names with explicit, non-duplicating legacy compatibility."""
FRONT_PUBLIC = 'cam_front'
FRONT_SENSOR = 'camera'  # Kept for existing backend/environment observations.
FRONT_PRIM = '/World/envs/env_0/cam_front'
FRONT_DATA = {'rgb':'front_rgb','depth':'front_depth','semantic':'front_semantic',
              'calibration':'camera_calibration/front'}


def public_name(sensor_name):
    return FRONT_PUBLIC if sensor_name == FRONT_SENSOR else sensor_name


def sensor_names(cameras):
    result = dict(cameras)
    if FRONT_PUBLIC in result:
        pose = result.pop(FRONT_PUBLIC)
        if FRONT_SENSOR in result and result[FRONT_SENSOR] != pose:
            raise ValueError('Conflicting camera/cam_front poses; keep only cam_front')
        result[FRONT_SENSOR] = pose
    return result
