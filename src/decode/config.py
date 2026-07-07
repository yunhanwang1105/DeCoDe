from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    num_ways: int
    num_shots: int
    num_episodes: int
    max_new_tokens: int
    batch_size: int
    prompt_mode: str
    use_domain_info: bool
    output_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    model_name_or_path: str
    cache_dir: Path
    torch_dtype: str
    device_map: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    images_path: Path
    sampling_log: Path
    resolve_mode: str
    support_key: str
    query_key: str
    domain_info: str | None


@dataclass(frozen=True)
class LoggingConfig:
    use_wandb: bool
    wandb_project: str
    run_name_prefix: str


@dataclass(frozen=True)
class DeCoDeConfig:
    repo_root: Path
    experiment: ExperimentConfig
    model: ModelConfig
    logging: LoggingConfig
    datasets: dict[str, DatasetConfig]


def load_config(config_path: str | Path) -> DeCoDeConfig:
    config_path = Path(config_path).expanduser().resolve()
    repo_root = config_path.parents[1]

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    experiment = raw["experiment"]
    model = raw["model"]
    logging = raw.get("logging", {})

    datasets = {
        name: DatasetConfig(
            name=name,
            images_path=_resolve_path(repo_root, values["images_path"]),
            sampling_log=_resolve_path(repo_root, values["sampling_log"]),
            resolve_mode=values["resolve_mode"],
            support_key=values.get("support_key", "sampled_imgs_support"),
            query_key=values.get("query_key", "sampled_imgs_query"),
            domain_info=values.get("domain_info"),
        )
        for name, values in raw["datasets"].items()
    }

    return DeCoDeConfig(
        repo_root=repo_root,
        experiment=ExperimentConfig(
            name=experiment["name"],
            seed=int(experiment.get("seed", 42)),
            num_ways=int(experiment.get("num_ways", 5)),
            num_shots=int(experiment.get("num_shots", 1)),
            num_episodes=int(experiment.get("num_episodes", 200)),
            max_new_tokens=int(experiment.get("max_new_tokens", 10)),
            batch_size=int(experiment.get("batch_size", 1)),
            prompt_mode=experiment.get("prompt_mode", "decompose_semantic"),
            use_domain_info=bool(experiment.get("use_domain_info", False)),
            output_dir=_resolve_path(repo_root, experiment.get("output_dir", "episode_logs/runs")),
        ),
        model=ModelConfig(
            backend=model.get("backend", "qwen3vl"),
            model_name_or_path=model["model_name_or_path"],
            cache_dir=_resolve_path(repo_root, model.get("cache_dir", "cache")),
            torch_dtype=model.get("torch_dtype", "auto"),
            device_map=model.get("device_map", "auto"),
        ),
        logging=LoggingConfig(
            use_wandb=bool(logging.get("use_wandb", False)),
            wandb_project=logging.get("wandb_project", "few-shot-vlm-decode"),
            run_name_prefix=logging.get("run_name_prefix", "decode"),
        ),
        datasets=datasets,
    )


def update_config(raw: DeCoDeConfig, overrides: dict[str, Any]) -> DeCoDeConfig:
    """Return a config with CLI-level experiment/model overrides applied."""
    experiment = raw.experiment
    model = raw.model
    logging = raw.logging

    if overrides.get("num_episodes") is not None:
        experiment = _replace_dataclass(experiment, num_episodes=overrides["num_episodes"])
    if overrides.get("max_new_tokens") is not None:
        experiment = _replace_dataclass(experiment, max_new_tokens=overrides["max_new_tokens"])
    if overrides.get("batch_size") is not None:
        experiment = _replace_dataclass(experiment, batch_size=overrides["batch_size"])
    if overrides.get("prompt_mode") is not None:
        experiment = _replace_dataclass(experiment, prompt_mode=overrides["prompt_mode"])
    if overrides.get("use_domain_info") is not None:
        experiment = _replace_dataclass(experiment, use_domain_info=overrides["use_domain_info"])
    if overrides.get("model_name_or_path") is not None:
        model = _replace_dataclass(model, model_name_or_path=overrides["model_name_or_path"])
    if overrides.get("cache_dir") is not None:
        model = _replace_dataclass(model, cache_dir=Path(overrides["cache_dir"]).expanduser().resolve())
    if overrides.get("use_wandb") is not None:
        logging = _replace_dataclass(logging, use_wandb=overrides["use_wandb"])

    return DeCoDeConfig(
        repo_root=raw.repo_root,
        experiment=experiment,
        model=model,
        logging=logging,
        datasets=raw.datasets,
    )


def _resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _replace_dataclass(instance: Any, **changes: Any) -> Any:
    values = instance.__dict__.copy()
    values.update(changes)
    return type(instance)(**values)
