from pathlib import Path
import torch

from train import evaluate_policy
from agents import DQNAgent

model_path = Path('models/dqn_maze_best.pth')

if not model_path.exists():
    print('Model not found at', model_path)
else:
    agent = DQNAgent.load(model_path, device=torch.device('cpu'))
    reward = evaluate_policy(agent, episodes=5, max_steps=250)
    print('Eval reward average:', reward)
