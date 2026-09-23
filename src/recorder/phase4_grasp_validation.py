"""Measured expert-only grasp and placement evidence. Never policy inputs."""
import math
import numpy as np


class EvidenceFailure(RuntimeError):
    pass


def rotate(q, vectors):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    vectors = np.asarray(vectors, dtype=float)
    cross = 2 * np.cross(q[1:], vectors)
    return vectors + q[0] * cross + np.cross(q[1:], cross)


def mesh_points_in_frame(prim):
    """Ignore stale authored extents; use actual visible mesh vertices."""
    from pxr import Usd, UsdGeom
    inverse = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).GetInverse()
    batches=[]
    for child in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
        if not child.IsA(UsdGeom.Mesh): continue
        if UsdGeom.Imageable(child).ComputeVisibility() == 'invisible': continue
        points=np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get(),dtype=float)
        if points.ndim!=2 or not len(points): continue
        matrix=np.asarray(UsdGeom.Xformable(child).ComputeLocalToWorldTransform(0)*inverse)
        batches.append((np.column_stack((points,np.ones(len(points))))@matrix)[:,:3])
    if not batches: raise RuntimeError('No mesh vertices: '+str(prim.GetPath()))
    points=np.unique(np.concatenate(batches),axis=0)
    from scipy.spatial import ConvexHull
    return points[ConvexHull(points).vertices]


def rigid_world_points(obj):
    """Visual bounds expressed in the actual PhysX link frame, not USD defaultPrim."""
    if not hasattr(obj, '_p4_link_corners'):
        from pxr import Usd, UsdGeom, Gf
        import omni.usd
        path = obj.root_physx_view.prim_paths[0]
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
        raw = mesh_points_in_frame(prim)
        scale=np.abs(np.asarray(Gf.Transform(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).GetScale()))
        obj._p4_link_corners=raw*scale
        print(f'[P4 QC GEOMETRY] rigid_link={path} vertex_hull={len(raw)} local_min={obj._p4_link_corners.min(axis=0).tolist()} local_max={obj._p4_link_corners.max(axis=0).tolist()}',flush=True)
    return rotate(obj.data.root_link_quat_w[0].detach().cpu().numpy(),obj._p4_link_corners)+obj.data.root_link_pos_w[0].detach().cpu().numpy()


class GraspEvidence:
    def __init__(self, tray_xy, tray_size, tray_yaw, tray_z):
        self.tray_xy = np.asarray(tray_xy, dtype=float)
        self.tray_size = np.asarray(tray_size, dtype=float)
        self.tray_yaw = math.radians(tray_yaw)
        self.tray_z = float(tray_z)
        self.lift_verified = False
        self.place_verified = False
        self.empty_samples = 0
        self.slip_samples = 0
        self.peak_rise = 0.
        self.trace = []
        self.centers = []
        self.anchor = None
        self.release_center = None
        self.last_placement = {}

    @staticmethod
    def local_anchor(ee, quat, center):
        q = np.array(quat, dtype=float)
        q[1:] *= -1
        return rotate(q, np.asarray(center)-ee)

    def begin_lift(self, ee, quat, center, bottom):
        self.lift_ee = np.array(ee, dtype=float)
        self.lift_center = np.array(center, dtype=float)
        self.lift_bottom = float(bottom)
        self.anchor = self.local_anchor(ee, quat, center)

    def check_lift(self, ee, quat, center, bottom, step):
        ee_rise = float(ee[2]-self.lift_ee[2])
        object_rise = float(center[2]-self.lift_center[2])
        self.peak_rise = max(self.peak_rise, object_rise)
        empty = ee_rise >= .020 and object_rise < max(.006, .35*ee_rise)
        self.empty_samples = self.empty_samples+1 if empty else 0
        self.trace.append(dict(stage='LIFT_CLEAR',step=step,ee_rise=ee_rise,object_rise=object_rise,bottom=float(bottom)))
        if self.empty_samples >= 2:
            raise EvidenceFailure(f'empty_grasp_early: EE rose {ee_rise*1000:.1f}mm but object rose {object_rise*1000:.1f}mm')
        if ee_rise >= .020:
            self.check_held(ee, quat, center, step, 'LIFT_CLEAR')
        if object_rise >= .035 and bottom >= .025:
            self.lift_verified = True

    def finish_lift(self):
        if not self.lift_verified:
            raise EvidenceFailure(f'lift_not_verified: maximum object center rise={self.peak_rise:.4f}m')

    def check_held(self, ee, quat, center, step, stage):
        if self.anchor is None:
            raise EvidenceFailure('missing_grasp_anchor')
        shift = float(np.linalg.norm(self.local_anchor(ee, quat, center)-self.anchor))
        self.slip_samples = self.slip_samples+1 if shift > .025 else 0
        self.trace.append(dict(stage=stage,step=step,relative_error_m=shift,
                               ee_position=np.asarray(ee).tolist(),ee_quaternion=np.asarray(quat).tolist(),
                               object_center=np.asarray(center).tolist()))
        if self.slip_samples >= 2:
            raise EvidenceFailure(f'object_slipped: displacement within gripper={shift*1000:.1f}mm')

    def inside_tray(self, points, margin=.004):
        xy = np.asarray(points)[:,:2] - self.tray_xy
        c, s = math.cos(self.tray_yaw), math.sin(self.tray_yaw)
        local = np.column_stack((c*xy[:,0]+s*xy[:,1], -s*xy[:,0]+c*xy[:,1]))
        return bool(np.all(np.abs(local) <= self.tray_size/2-margin))

    def begin_release(self, ee, quat, center, points, bottom):
        self.last_placement=dict(center=np.asarray(center).tolist(),bottom=float(bottom),
            min_xyz=np.min(points,axis=0).tolist(),max_xyz=np.max(points,axis=0).tolist(),
            inside_tray=self.inside_tray(points),tray_xy=self.tray_xy.tolist())
        self.finish_lift()
        self.check_held(ee, quat, center, 0, 'BEFORE_OPEN')
        # Strict single-sample gate before deliberately releasing the object.
        shift = float(np.linalg.norm(self.local_anchor(ee,quat,center)-self.anchor))
        if shift > .025 or not self.inside_tray(points):
            raise EvidenceFailure(f'release_not_ready: held_error={shift:.4f}m, geometry={self.last_placement}')
        if not self.tray_z-.015 <= bottom <= self.tray_z+.045:
            raise EvidenceFailure(f'release_height_invalid: object_bottom={bottom:.4f}m tray_z={self.tray_z:.4f}m')
        self.release_center = np.array(center)
        self.release_ee = np.array(ee)

    def check_retreat(self, ee, center, points, bottom):
        if self.release_center is None:
            raise EvidenceFailure('missing_release_evidence')
        self.centers.append(np.array(center))
        self.centers = self.centers[-6:]
        if ee[2]-self.release_ee[2] >= .045 and center[2]-self.release_center[2] >= .025:
            raise EvidenceFailure('object_still_attached_after_open')
        stable = len(self.centers) >= 6 and float(np.ptp(self.centers,axis=0).max()) < .002
        self.place_verified = (self.inside_tray(points) and self.tray_z-.015 <= bottom <= self.tray_z+.018 and stable)
        self.last_placement = dict(center=np.asarray(center).tolist(),bottom=float(bottom),inside_tray=self.inside_tray(points),stable=stable,
                                   min_xyz=np.min(points,axis=0).tolist(),max_xyz=np.max(points,axis=0).tolist())
        return self.place_verified

    def summary(self):
        return dict(version='20260915-v5',geometry_source='mesh_vertices_physx_link_v1',lift_verified=self.lift_verified,
                    place_verified=self.place_verified,maximum_center_lift_m=self.peak_rise,
                    final_placement=self.last_placement,
                    evidence_source='simulator_expert_QC_only',trace=self.trace)
