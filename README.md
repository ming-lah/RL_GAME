# Maze Explorer DQN

A reinforcement learning project that trains a Deep Q-Network (DQN) agent to solve a top-down maze with a key-door mechanic, traps, and pygame-based visualisation. The maze is intentionally richer than classic control tasks (e.g. CartPole_v2) and includes coloured tile rendering plus replay scripts for interactive demos.

## Highlights
- Custom maze environment (`envs/maze_env.py`) with walls, traps, a collectible key, and a locked door guarding the goal.
- PyTorch DQN agent (`agents/dqn.py`) featuring replay memory, target network updates, gradient clipping, and configurable hyperparameters.
- Training harness (`train.py`) that saves checkpoints, logs metrics, and optionally evaluates the policy during training.
- Playback utility (`play.py`) to render trained policies with pygame.
- Metric plotting script (`plot_metrics.py`) that turns JSON logs into reward/loss/epsilon curves.

## Project Layout
```
project_rl/
|-- agents/            # DQN implementation and configuration dataclass
|-- envs/              # Custom maze environment with pygame rendering
|-- models/            # Default output directory for checkpoints & metrics
|-- utils/             # Shared utilities (e.g. replay buffer)
|-- play.py            # Run trained agent with live rendering
|-- plot_metrics.py    # Create training curve plot from logged JSON
|-- requirements.txt   # Suggested Python dependencies
`-- train.py           # Training entry point
```

## Setup
1. Activate your Conda environment :
   ```bash
   conda activate your_env
   ```
2. Install required packages (PyTorch build can be skipped if already available with GPU support):
   ```bash
   pip install -r requirements.txt
   ```

> **Note:** The maze environment auto-imports `gymnasium` first and falls back to `gym` if available. Install at least one of them (the requirement file targets `gymnasium`).

## Training
Launch training with default hyperparameters (800 episodes, 250 max steps) and GPU acceleration when available:
```bash
python train.py --device cuda
```
Key behaviours:
- Checkpoints land in `models/dqn_maze_epXXX.pth`; the best-performing model is stored as `models/dqn_maze_best.pth`.
- Episode statistics and evaluation scores are written to `models/training_metrics.json` for later analysis.
- Console logging reports episodic rewards, rolling averages, epsilon, and evaluation summaries.

Tuneable CLI flags include `--episodes`, `--max-steps`, `--eval-interval`, `--save-frequency`, and `--seed`.

## Visual Playthrough
After (or even during) training, open a pygame window to watch the agent:
```bash
python play.py --model-path models/dqn_maze_best.pth --episodes 3 --sleep 0.04
```
Flags:
- `--model-path`: path to any saved checkpoint (falls back to random actions if missing).
- `--sleep`: controls frame delay for smoother or faster playback.
- `--device`: `auto` (default), `cpu`, or `cuda`.

## Metric Plotting
Generate reward/loss/epsilon curves from the logged metrics JSON:
```bash
python plot_metrics.py --metrics-path models/training_metrics.json --output-path models/training_curve.png
```
The command writes a multi-panel PNG plot containing episodic reward traces (plus evaluation markers), mean training loss, and epsilon decay.

## Environment Details
- Grid size: 11x12 tiles with outer walls for containment.
- Tiles:
  - `S`: agent spawn point.
  - `K`: key (must be picked up to open the door).
  - `D`: locked door; impassable until the key is collected.
  - `T`: trap tiles with additional negative reward.
  - `G`: goal tile worth +1.0 reward.
  - `#`: walls; colliding yields a penalty.
- Observation: `[x_norm, y_norm, has_key]`, where position is normalised to [0,1] and the key flag is binary.
- Actions: up, down, left, right (discrete with self-contained bounds checks).
- Rewards: step penalty (-0.01), wall/door penalties, trap penalty, door unlock bonus, goal reward (+1.0).

## Suggested Experiments
1. Curriculum ideas - adjust rewards or traps in `envs/maze_env.py` to create harder mazes.
2. Network tweaks - modify `hidden_layers` within `DQNConfig` to explore deeper or narrower networks.
3. Experience replay variants - add prioritisation logic inside `utils/replay_buffer.py`.
4. Alternative policies - swap in duelling heads, Double DQN updates, or epsilon scheduling functions.

## Troubleshooting
- Missing `gymnasium`/`gym`: install via `pip install gymnasium` (or `gym`).
- No pygame window on headless systems: use `render_mode="rgb_array"` when instantiating the environment or disable playback.
- Performance: `--device cuda` will automatically leverage an RTX 4060 when PyTorch detects CUDA drivers.

Enjoy exploring the maze!
