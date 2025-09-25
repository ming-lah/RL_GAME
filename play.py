import argparse
import time
from pathlib import Path

import torch

from agents import DQNAgent
from envs import MazeKeyDoorEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained agent navigating the maze.")
    parser.add_argument("--model-path", type=Path, default=Path("models/dqn_maze_best.pth"),
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play.")
    parser.add_argument("--max-steps", type=int, default=250, help="Maximum steps per episode.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep time between frames in seconds.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to run inference on.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    env = MazeKeyDoorEnv(render_mode="human")
    agent = None

    if args.model_path.exists():
        print(f"Loading agent from {args.model_path}")
        agent = DQNAgent.load(args.model_path, device=device)
    else:
        print("Model path not found. Running with a random agent.")

    try:
        for episode in range(1, args.episodes + 1):
            state, _ = env.reset()
            total_reward = 0.0
            print(f"Episode {episode}")

            for step in range(1, args.max_steps + 1):
                env.render()
                if agent is None:
                    action = env.action_space.sample()
                else:
                    action = agent.act(state, eval_mode=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                state = next_state
                if args.sleep > 0:
                    time.sleep(args.sleep)
                if terminated or truncated:
                    break

            print(f"  Reward: {total_reward:.3f} | Steps: {step}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
