import dataclasses

import einops
import numpy as np

from openpi import transforms

# YamAbc (XDOF/ABC) bimanual EEF layout, per arm: [x, y, z, r00, r10, r20, r01, r11, r21, gripper], left then right.
# The rotation is a 6D representation: the first two columns of the rotation matrix in each arm's base frame.
ABC_EEF_ARM_DIM = 10
ABC_EEF_DIM = 2 * ABC_EEF_ARM_DIM


def make_yam_abc_example() -> dict:
    """Creates a random input example for the YamAbc EEF policy."""
    arm = np.array([0.3, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    return {
        "state": np.concatenate([arm, arm]),
        "images": {
            "top": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """[..., 6] (first two columns of R) -> [..., 3, 3]. Gram-Schmidt makes it valid for noisy model outputs."""
    a1, a2 = rot6d[..., 0:3], rot6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _matrix_to_rot6d(rot: np.ndarray) -> np.ndarray:
    """[..., 3, 3] -> [..., 6] (first two columns of R)."""
    return np.concatenate([rot[..., :, 0], rot[..., :, 1]], axis=-1)


def _relative_eef(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Expresses absolute EEF action poses in the current EEF frame: inv(T_state) @ T_action, per arm.

    state: [20] absolute EEF pose at the current frame. actions: [H, 20] absolute EEF poses.
    The gripper stays absolute.
    """
    out = actions.copy()
    for start in (0, ABC_EEF_ARM_DIM):
        pos0 = state[start : start + 3]
        rot0 = _rot6d_to_matrix(state[start + 3 : start + 9])
        pos = actions[:, start : start + 3]
        rot = _rot6d_to_matrix(actions[:, start + 3 : start + 9])
        out[:, start : start + 3] = (pos - pos0) @ rot0
        out[:, start + 3 : start + 9] = _matrix_to_rot6d(rot0.T @ rot)
    return out


def _absolute_eef(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Inverse of `_relative_eef`: T_state @ T_relative, per arm."""
    out = actions.copy()
    for start in (0, ABC_EEF_ARM_DIM):
        pos0 = state[start : start + 3]
        rot0 = _rot6d_to_matrix(state[start + 3 : start + 9])
        rel_pos = actions[:, start : start + 3]
        rel_rot = _rot6d_to_matrix(actions[:, start + 3 : start + 9])
        out[:, start : start + 3] = pos0 + rel_pos @ rot0.T
        out[:, start + 3 : start + 9] = _matrix_to_rot6d(rot0 @ rel_rot)
    return out


@dataclasses.dataclass(frozen=True)
class YamAbcInputs(transforms.DataTransformFn):
    """Inputs for the YamAbc bimanual EEF policy.

    Expected inputs:
    - images: dict with keys "top", "left_wrist", "right_wrist" (HWC or CHW, uint8 or float in [0, 1]).
    - state: [20] absolute EEF pose, 2x (xyz + rot6d + gripper).
    - actions: [action_horizon, 20] absolute EEF poses (training only). Use `RelativeEefActions` to make them
      relative to the current EEF pose.
    """

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        images = {
            "base_0_rgb": _parse_image(in_images["top"]),
            "left_wrist_0_rgb": _parse_image(in_images["left_wrist"]),
            "right_wrist_0_rgb": _parse_image(in_images["right_wrist"]),
        }
        inputs = {
            "image": images,
            "image_mask": dict.fromkeys(images, np.True_),
            "state": np.asarray(data["state"]),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RelativeEefActions(transforms.DataTransformFn):
    """Converts an absolute EEF action chunk into poses relative to the current EEF pose (the state).

    Runs per sample, so every chunk is expressed in the frame of its own first observation.
    """

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        state = np.asarray(data["state"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        return {**data, "actions": _relative_eef(state[:ABC_EEF_DIM], actions[:, :ABC_EEF_DIM])}


@dataclasses.dataclass(frozen=True)
class AbsoluteEefActions(transforms.DataTransformFn):
    """Converts relative EEF actions predicted by the model back into absolute EEF poses."""

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        state = np.asarray(data["state"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        return {**data, "actions": _absolute_eef(state[:ABC_EEF_DIM], actions[:, :ABC_EEF_DIM])}


@dataclasses.dataclass(frozen=True)
class YamAbcOutputs(transforms.DataTransformFn):
    """Outputs for the YamAbc bimanual EEF policy."""

    def __call__(self, data: dict) -> dict:
        # Only the first 20 dims are used; the rest is padding to the model action dim.
        return {"actions": np.asarray(data["actions"][:, :ABC_EEF_DIM])}
