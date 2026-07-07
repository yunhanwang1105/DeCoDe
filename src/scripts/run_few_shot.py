from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeCoDe few-shot evaluation.")
    parser.add_argument("--config", default="configs/qwen3vl_decompose.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--prompt_mode",
        choices=["decompose_anonymous", "decompose_semantic"],
        default=None,
    )
    parser.add_argument("--domain_info", action="store_true", dest="use_domain_info", default=None)
    parser.add_argument("--no_domain_info", action="store_false", dest="use_domain_info")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--use_wandb", action="store_true", default=None)
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from decode.config import load_config, update_config
    from decode.evaluator import evaluate_dataset
    from decode.models.factory import build_model
    from decode.utils.reproducibility import seed_everything

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_config(config_path)
    config = update_config(
        config,
        {
            "num_episodes": args.num_episodes,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "prompt_mode": args.prompt_mode,
            "use_domain_info": args.use_domain_info,
            "model_name_or_path": args.model_name_or_path,
            "cache_dir": args.cache_dir,
            "use_wandb": args.use_wandb,
        },
    )

    seed_everything(config.experiment.seed)
    model = build_model(config.model)
    evaluate_dataset(config, args.dataset, model)


if __name__ == "__main__":
    main()
