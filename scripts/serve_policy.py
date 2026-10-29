import dataclasses
import enum
import logging
import pathlib
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    # Norm stats to load, from <dir>/assets/<asset_id>. Defaults to the config's asset id, or, when the checkpoint
    # does not have that one (trained with a --data.assets.asset-id override), to the only norm stats it has.
    asset_id: str | None = None


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Hide the state from the prompt at inference. Only for data configs with `inference_drop_state` (e.g. YamAbc).
    drop_state: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def _checkpoint_asset_id(checkpoint: Checkpoint, train_config: _config.TrainConfig) -> str | None:
    """The asset id of the norm stats saved with the checkpoint, if it differs from the config's."""
    if checkpoint.asset_id is not None:
        return checkpoint.asset_id
    assets_dir = pathlib.Path(checkpoint.dir) / "assets"
    if not assets_dir.is_dir():  # e.g. a gs:// checkpoint; keep the config's asset id.
        return None
    config_asset_id = train_config.data.create(train_config.assets_dirs, train_config.model).asset_id
    if config_asset_id is not None and (assets_dir / config_asset_id / "norm_stats.json").is_file():
        return None
    found = sorted(str(p.parent.relative_to(assets_dir)) for p in assets_dir.rglob("norm_stats.json"))
    if len(found) != 1:
        raise ValueError(
            f"{assets_dir} has no norm stats for asset id {config_asset_id!r} but {found}; pass --policy.asset-id."
        )
    logging.info("Using the checkpoint's norm stats for asset id %s", found[0])
    return found[0]


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            if asset_id := _checkpoint_asset_id(args.policy, train_config):
                data = train_config.data
                train_config = dataclasses.replace(
                    train_config,
                    data=dataclasses.replace(data, assets=dataclasses.replace(data.assets, asset_id=asset_id)),
                )
            if args.drop_state:
                if not hasattr(train_config.data, "inference_drop_state"):
                    raise ValueError(f"Config {args.policy.config} does not support --drop-state.")
                train_config = dataclasses.replace(
                    train_config, data=dataclasses.replace(train_config.data, inference_drop_state=True)
                )
            return _policy_config.create_trained_policy(
                train_config, args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
