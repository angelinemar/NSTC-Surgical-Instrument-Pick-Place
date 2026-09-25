"""Sensor-only inference boundary, shared by replay and future live evaluation."""
from collections import deque
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .export_sensor_only import CAMERAS, assert_sensor_inputs
    from .sensor_policy import SensorPolicy, decode_actions
else:
    from export_sensor_only import CAMERAS, assert_sensor_inputs
    from sensor_policy import SensorPolicy, decode_actions


class PolicyRuntime:
    def __init__(self, checkpoint, device='cpu', allow_smoke=False):
        data = torch.load(Path(checkpoint), map_location=device, weights_only=True)
        if data.get('input_contract') != 'p4_sensor_only_v1':
            raise ValueError('Incompatible checkpoint input contract')
        if data.get('smoke_only') and not allow_smoke:
            raise ValueError('Smoke checkpoint is not approved for deployment')
        self.device, self.stats = device, data['stats']
        self.skill = data['skill']
        self.controller_contract = data.get('controller_contract')
        self.model = SensorPolicy(data['observation_horizon'], data['prediction_horizon']).to(device)
        self.model.load_state_dict(data['model'])
        self.model.eval()
        self.history = deque(maxlen=data['observation_horizon'])

    def reset(self):
        self.history.clear()

    def attach_controller(self, env):
        """Use the same robot controller contract as the v6 expert for this skill."""
        from phase4_approach import configure, set_precision_skill
        if not self.controller_contract or self.controller_contract[0] != 'tcp_frame_correct_bounded_place_integral_v6':
            raise ValueError('Checkpoint controller does not match the live v6 controller')
        configure(env, True)
        set_precision_skill(env, self.skill)

    def observe(self, sensors):
        """Call every control tick, including ticks that execute a cached action prefix."""
        assert_sensor_inputs(sensors)
        images = []
        for name in CAMERAS:
            image = np.asarray(sensors[name + '_rgb'])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError('Expected the recorded 224x224 uint8 sensor crop')
            images.append(image)
        proprio = np.asarray(sensors['robot_proprio'], dtype=np.float32).copy()
        if proprio.shape != (16,) or not np.isfinite(proprio).all():
            raise ValueError('Invalid measured robot proprioception')
        norm = np.linalg.norm(proprio[12:16])
        if norm < 1e-6:
            raise ValueError('Invalid measured robot quaternion')
        proprio[12:16] /= norm
        if self.history:
            reference = self.history[-1][1][12:16]
            if np.dot(reference, proprio[12:16]) < 0:
                proprio[12:16] *= -1
        elif proprio[12] < 0:
            proprio[12:16] *= -1
        self.history.append((np.stack(images), proprio))
        while len(self.history) < self.history.maxlen:
            self.history.appendleft(self.history[0])

    def predict(self, sensors=None, inference_steps=20):
        if sensors is not None:
            self.observe(sensors)
        if not self.history:
            raise ValueError('Observe sensors before requesting an action')
        rgb = torch.from_numpy(np.stack([r[0] for r in self.history])).permute(0, 1, 4, 2, 3).float().unsqueeze(0).to(self.device) / 255
        state = np.stack([r[1] for r in self.history])
        state = (state - self.stats['proprio_mean']) / self.stats['proprio_std']
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        normalized, semantic_prediction = self.model.act(rgb, state, inference_steps)
        return decode_actions(normalized, self.stats)[0].cpu().numpy(), semantic_prediction[0].cpu().numpy()


def read_live_sensors(env):
    """Only robot encoders/FK and camera RGB. Never access an instrument prim."""
    from isaaclab.utils.math import subtract_frame_transforms
    robot = env.scene['robot']
    ee = env.scene['ee_frame'].data
    position, quaternion = subtract_frame_transforms(robot.data.root_pos_w[:1], robot.data.root_quat_w[:1],
        ee.target_pos_w[:, 0], ee.target_quat_w[:, 0])
    names = ['panda_joint' + str(i) for i in range(1, 8)] + ['panda_finger_joint1', 'panda_finger_joint2']
    ids = [robot.joint_names.index(name) for name in names]
    values = torch.cat((robot.data.joint_pos[:, ids], position, quaternion), dim=-1)[0]
    result = dict(robot_proprio=values.detach().cpu().numpy())
    for public, sensor in zip(CAMERAS, ('camera', 'grip_cam_b', 'cam_top', 'cam_left', 'cam_right', 'cam_tray')):
        frame = env.scene[sensor].data.output['rgb'][0].detach().cpu().numpy()[..., :3]
        h, w = frame.shape[:2]
        if (h,w) not in ((224,224),(448,448)):
            raise ValueError('Expected native square P4 camera; legacy wide sensors need explicit migration')
        if (h,w) == (448,448):
            from PIL import Image
            frame = np.asarray(Image.fromarray(frame).resize((224,224), Image.Resampling.LANCZOS))
        result[public + '_rgb'] = frame.copy()
    return result
