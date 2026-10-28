import dataclasses

import einops
import numpy as np

from openpi import transforms

# YamAbc (XDOF/ABC) bimanual joint layout, per arm: [joint1..joint6, gripper], left then right.
ABC_JOINT_DIM = 14


def make_yam_abc_joint_example() -> dict:
    """Creates a random input example for the YamAbc joint policy."""
    return {
        "state": np.ones((ABC_JOINT_DIM,)),
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


@dataclasses.dataclass(frozen=True)
class YamAbcJointInputs(transforms.DataTransformFn):
    """Inputs for the YamAbc bimanual policy with absolute joint actions.

    Expected inputs:
    - images: dict with keys "top", "left_wrist", "right_wrist" (HWC or CHW, uint8 or float in [0, 1]).
    - state: [14] absolute joint positions, 2x (6 joints + gripper).
    - actions: [action_horizon, 14] absolute joint positions (training only).
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
class YamAbcJointOutputs(transforms.DataTransformFn):
    """Outputs for the YamAbc bimanual joint policy."""

    def __call__(self, data: dict) -> dict:
        # Only the first 14 dims are used; the rest is padding to the model action dim.
        return {"actions": np.asarray(data["actions"][:, :ABC_JOINT_DIM])}
