import dataclasses

import einops
import numpy as np

from openpi import transforms


def make_yam_ma2_example() -> dict:
    """Creates a random input example for the YAM Ma2 bimanual policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "top": np.random.randint(256, size=(360, 640, 3), dtype=np.uint8),
            "left": np.random.randint(256, size=(360, 640, 3), dtype=np.uint8),
            "right": np.random.randint(256, size=(360, 640, 3), dtype=np.uint8),
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
class YamMa2Inputs(transforms.DataTransformFn):
    """Inputs for the YAM Ma2 bimanual policy (absolute joint actions).

    Expected inputs:
    - images: dict with keys "top", "left", "right" (HWC or CHW, uint8 or float in [0, 1]).
    - state: [14] absolute joint positions, 2x (6 joints + gripper).
    - actions: [action_horizon, 14] (training only).
    """

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        base_image = _parse_image(in_images["top"])

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}
        for dest, source in (("left_wrist_0_rgb", "left"), ("right_wrist_0_rgb", "right")):
            if source in in_images:
                images[dest] = _parse_image(in_images[source])
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"]),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class YamMa2Outputs(transforms.DataTransformFn):
    """Outputs for the YAM Ma2 bimanual policy."""

    def __call__(self, data: dict) -> dict:
        # Only the first 14 dims are used; the rest is padding to the model action dim.
        return {"actions": np.asarray(data["actions"][:, :14])}
