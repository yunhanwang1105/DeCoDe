from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from decode.config import DeCoDeConfig
from decode.datasets.episodes import Episode, load_episodes
from decode.models.base import MultimodalLLM
from decode.prompts.decomposed import build_pair_message
from decode.utils.labels import clean_class_label


@dataclass(frozen=True)
class Prediction:
    query_index: int
    query_image: str
    correct_class: str
    predicted_class: str
    predicted_index: int
    yes_log_probs: list[float]
    is_correct: bool


@dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    accuracy: float
    predictions: list[Prediction]


def evaluate_dataset(
    config: DeCoDeConfig,
    dataset_name: str,
    model: MultimodalLLM,
) -> float:
    if dataset_name not in config.datasets:
        available = ", ".join(sorted(config.datasets))
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}")

    dataset = config.datasets[dataset_name]
    episodes = load_episodes(
        dataset=dataset,
        num_ways=config.experiment.num_ways,
        num_episodes=config.experiment.num_episodes,
    )
    if not episodes:
        raise ValueError(f"No episodes loaded from {dataset.sampling_log}")
    if config.experiment.batch_size <= 0:
        raise ValueError("experiment.batch_size must be > 0")

    output_path = _make_output_path(config, dataset_name)
    wandb_run = _start_wandb(config, dataset_name, len(episodes))

    running_accuracy = 0.0
    with output_path.open("w", encoding="utf-8") as f:
        for episode_index, episode in enumerate(tqdm(episodes, desc=f"Evaluating {dataset_name}")):
            result = evaluate_episode(
                config=config,
                episode=episode,
                model=model,
                episode_index=episode_index,
                domain_info=dataset.domain_info,
            )
            running_accuracy += result.accuracy
            mean_accuracy = running_accuracy / (episode_index + 1)

            record = asdict(result)
            record["running_accuracy"] = mean_accuracy
            f.write(json.dumps(record) + "\n")

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "episode_accuracy": result.accuracy,
                        "accuracy": mean_accuracy,
                    }
                )

    final_accuracy = running_accuracy / len(episodes)
    if wandb_run is not None:
        wandb_run.finish()

    print(f"Saved episode predictions to: {output_path}")
    print(f"Final accuracy: {final_accuracy:.4f}")
    return final_accuracy


def evaluate_episode(
    config: DeCoDeConfig,
    episode: Episode,
    model: MultimodalLLM,
    episode_index: int,
    domain_info: str | None,
) -> EpisodeResult:
    pair_messages = []
    pair_indices = []
    for query_index, query_image in enumerate(episode.query_images):
        for support_index, support_image in enumerate(episode.support_images):
            support_label = (
                episode.sampled_classes[support_index]
                if config.experiment.prompt_mode == "decompose_semantic"
                else None
            )
            pair_messages.append(
                build_pair_message(
                    support_image_path=support_image,
                    query_image_path=query_image,
                    prompt_mode=config.experiment.prompt_mode,
                    support_label=support_label,
                    use_domain_info=config.experiment.use_domain_info,
                    domain_info=domain_info,
                )
            )
            pair_indices.append((query_index, support_index))

    pair_yes_log_probs = []
    for start_idx in range(0, len(pair_messages), config.experiment.batch_size):
        batch_messages = pair_messages[start_idx:start_idx + config.experiment.batch_size]
        pair_yes_log_probs.extend(
            model.score_yes_batch(
                messages_batch=batch_messages,
                max_new_tokens=config.experiment.max_new_tokens,
            )
        )

    num_queries = len(episode.query_images)
    num_support = len(episode.support_images)
    yes_log_prob_matrix = np.full((num_queries, num_support), -np.inf, dtype=np.float32)
    for (query_index, support_index), score in zip(pair_indices, pair_yes_log_probs):
        yes_log_prob_matrix[query_index, support_index] = score

    predictions = []
    for query_index, query_image in enumerate(episode.query_images):
        yes_log_probs = yes_log_prob_matrix[query_index].tolist()
        predicted_index = int(np.argmax(yes_log_probs))
        correct_class = episode.sampled_classes[query_index]
        predicted_class = episode.sampled_classes[predicted_index]

        print("-" * 23)
        print(query_image)
        print(f"Correct class: {clean_class_label(correct_class)} (index {query_index})")
        print(f"Predicted class: {clean_class_label(predicted_class)} (index {predicted_index})")
        print(f"Yes log probs: {yes_log_probs}")

        predictions.append(
            Prediction(
                query_index=query_index,
                query_image=str(query_image),
                correct_class=correct_class,
                predicted_class=predicted_class,
                predicted_index=predicted_index,
                yes_log_probs=yes_log_probs,
                is_correct=predicted_class == correct_class,
            )
        )

    accuracy = sum(pred.is_correct for pred in predictions) / len(predictions)
    return EpisodeResult(
        episode_index=episode_index,
        accuracy=accuracy,
        predictions=predictions,
    )


def _make_output_path(config: DeCoDeConfig, dataset_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.experiment.output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_tag = _prompt_tag(config)
    return output_dir / f"{config.experiment.name}_{prompt_tag}_{timestamp}.jsonl"


def _start_wandb(config: DeCoDeConfig, dataset_name: str, num_episodes: int):
    if not config.logging.use_wandb:
        return None

    import wandb

    run_name = (
        f"{config.logging.run_name_prefix}_"
        f"{dataset_name}_{_prompt_tag(config)}"
    )
    return wandb.init(
        name=run_name,
        project=config.logging.wandb_project,
        config={
            "dataset": dataset_name,
            "domain_info": config.datasets[dataset_name].domain_info,
            "prompt_mode": config.experiment.prompt_mode,
            "use_domain_info": config.experiment.use_domain_info,
            "num_ways": config.experiment.num_ways,
            "num_shots": config.experiment.num_shots,
            "num_episodes": num_episodes,
            "max_new_tokens": config.experiment.max_new_tokens,
            "batch_size": config.experiment.batch_size,
            "model_name_or_path": config.model.model_name_or_path,
            "model_backend": config.model.backend,
        },
    )


def _prompt_tag(config: DeCoDeConfig) -> str:
    if config.experiment.use_domain_info:
        return f"{config.experiment.prompt_mode}_domain_info"
    return config.experiment.prompt_mode
