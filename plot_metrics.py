import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument("--metrics-path", type=Path, default=Path("models/training_metrics.json"),
                        help="Path to the metrics JSON file.")
    parser.add_argument("--output-path", type=Path, default=Path("models/training_curve.png"),
                        help="Path to save the generated plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    history = payload["history"]
    episodes = history["episode"]
    rewards = history["reward"]
    losses = history["loss"]
    epsilons = history["epsilon"]
    eval_entries = history.get("eval", [])

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(episodes, rewards, label="Episode reward", color="#1f77b4")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc="best")

    if eval_entries:
        eval_eps = [entry["episode"] for entry in eval_entries]
        eval_rewards = [entry["reward"] for entry in eval_entries]
        axes[0].plot(eval_eps, eval_rewards, "o-", label="Eval reward", color="#ff7f0e")
        axes[0].legend(loc="best")

    if any(loss is not None for loss in losses):
        cleaned_losses = [loss if loss is not None else np.nan for loss in losses]
        axes[1].plot(episodes, cleaned_losses, label="Training loss", color="#d62728")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True, alpha=0.2)
        axes[1].legend(loc="best")
    else:
        axes[1].set_visible(False)

    axes[2].plot(episodes, epsilons, label="Epsilon", color="#2ca02c")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Exploration rate")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(loc="best")

    plt.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()
