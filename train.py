import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from agents import DQNAgent, DQNConfig
from envs import MazeKeyDoorEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on the maze environment.")
    parser.add_argument("--episodes", type=int, default=800, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=250, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-frequency", type=int, default=100, help="Episodes between checkpoints.")
    parser.add_argument("--eval-interval", type=int, default=50, help="Episodes between evaluation runs.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory to store models and logs.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Computation device.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_policy(agent: DQNAgent, episodes: int = 5, max_steps: int = 250) -> Tuple[float, float]:
    eval_env = MazeKeyDoorEnv()
    agent.set_eval_mode()
    total = 0.0
    successes = 0
    try:
        for _ in range(episodes):
            state, _ = eval_env.reset()
            episode_reward = 0.0
            reached_goal = False
            for _ in range(max_steps):
                action = agent.act(state, eval_mode=True)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                state = next_state
                if terminated or truncated:
                    if terminated:
                        reached_goal = True
                    break
            total += episode_reward
            if reached_goal:
                successes += 1
    finally:
        eval_env.close()
        agent.set_train_mode()
    return total / episodes, successes / episodes


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Enforce CUDA availability if explicitly requested
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.stderr.write("[Error] --device cuda specified but CUDA is not available.\n")
        sys.exit(1)

    # Informative device printout
    if device.type == "cuda":
        try:
            dev_index = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_index)
            print(f"Using CUDA device {dev_index}: {dev_name}")
        except Exception:
            print("Using CUDA device")
    else:
        print("Using CPU device")

    env = MazeKeyDoorEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    config = DQNConfig(state_dim=state_dim, action_dim=action_dim)
    agent = DQNAgent(config, device=device)
    agent.set_train_mode()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "dqn_maze_best.pth"

    history: Dict[str, List] = {
        "episode": [],
        "reward": [],
        "length": [],
        "epsilon": [],
        "loss": [],
        "eval": [],
    }

    best_reward = -float("inf")

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        losses: List[float] = []

        for step in range(1, args.max_steps + 1):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            agent.decay_epsilon()
            state = next_state
            total_reward += reward
            if done:
                break

        history["episode"].append(episode)
        history["reward"].append(total_reward)
        history["length"].append(step)
        history["epsilon"].append(agent.epsilon)
        history["loss"].append(float(np.mean(losses)) if losses else None)

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(best_model_path)

        if episode % args.save_frequency == 0:
            checkpoint_path = args.output_dir / f"dqn_maze_ep{episode}.pth"
            agent.save(checkpoint_path)

        if episode % args.eval_interval == 0:
            eval_reward, eval_win_rate = evaluate_policy(agent, episodes=3, max_steps=args.max_steps)
            history["eval"].append({
                "episode": episode,
                "reward": eval_reward,
                "win_rate": eval_win_rate,
            })
            print(
                f"[Eval] Episode {episode}: avg reward {eval_reward:.3f} | win rate {eval_win_rate:.2%}"
            )

        if episode % 10 == 0 or episode == 1:
            recent_rewards = history["reward"][-10:]
            mean_reward = float(np.mean(recent_rewards)) if recent_rewards else total_reward
            print(
                f"Episode {episode:4d}/{args.episodes} | reward {total_reward:6.2f} | "
                f"avg10 {mean_reward:6.2f} | epsilon {agent.epsilon:5.3f} | steps {step:3d}"
            )

    env.close()

    metrics_path = args.output_dir / "training_metrics.json"
    serializable_history = {}
    for key, values in history.items():
        if key == "eval":
            serializable_history[key] = values
        else:
            cleaned = []
            for v in values:
                if v is None:
                    cleaned.append(None)
                else:
                    numeric = float(v)
                    cleaned.append(None if math.isnan(numeric) else numeric)
            serializable_history[key] = cleaned

    payload = {
        "history": serializable_history,
        "best_reward": best_reward,
        "agent_config": asdict(config),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Training complete. Best model saved to {best_model_path}")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
