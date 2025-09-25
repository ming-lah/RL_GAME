from typing import Dict, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover - pygame is optional for headless training
    pygame = None


class MazeKeyDoorEnv(gym.Env):
    """Maze environment with traps and a key-door mechanic."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 250):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode '{render_mode}'.")

        self.render_mode = render_mode
        self.max_steps = max_steps

        self.layout = [
            "############",
            "#S..#.....T#",
            "#.#.###.##.#",
            "#.#...#....#",
            "#.#.#.#.##.#",
            "#...#.#..#.#",
            "#.###.#K.#D#",
            "#.....##.#.#",
            "##T.#....#.#",
            "#..##.#.G..#",
            "############",
        ]
        self.height = len(self.layout)
        self.width = len(self.layout[0])
        self._parse_layout()

        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                                            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.np_random = None
        self.agent_pos: Tuple[int, int] = self.start_pos
        self.has_key = False
        self.step_count = 0

        self._window: Optional["pygame.Surface"] = None
        self._clock: Optional["pygame.time.Clock"] = None
        self._canvas: Optional["pygame.Surface"] = None
        self._cell_size = 48
        self._shaping_weight = 0.02  # guide exploration toward key/goal

    def _parse_layout(self) -> None:
        self.walls = set()
        self.traps = set()
        self.start_pos = None
        self.goal_pos = None
        self.key_pos = None
        self.door_pos = None

        for y, row in enumerate(self.layout):
            if len(row) != len(self.layout[0]):
                raise ValueError("All rows in the layout must have the same length.")
            for x, char in enumerate(row):
                if char == '#':
                    self.walls.add((x, y))
                elif char == 'S':
                    self.start_pos = (x, y)
                elif char == 'G':
                    self.goal_pos = (x, y)
                elif char == 'K':
                    self.key_pos = (x, y)
                elif char == 'D':
                    self.door_pos = (x, y)
                elif char == 'T':
                    self.traps.add((x, y))

        if self.start_pos is None or self.goal_pos is None or self.key_pos is None or self.door_pos is None:
            raise ValueError("Layout must contain start (S), goal (G), key (K), and door (D) tiles.")

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        elif self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random()

        self.agent_pos = self.start_pos
        self.has_key = False
        self.step_count = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}."
        self.step_count += 1
        reward = -0.01  # time penalty to encourage efficiency
        terminated = False
        truncated = False

        move_map = {
            0: (0, -1),  # up
            1: (0, 1),   # down
            2: (-1, 0),  # left
            3: (1, 0),   # right
        }
        dx, dy = move_map[action]
        candidate = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        stage_has_key = self.has_key
        potential_before = self._potential(self.agent_pos, stage_has_key)

        if not (0 <= candidate[0] < self.width and 0 <= candidate[1] < self.height):
            reward -= 0.25
        elif candidate in self.walls:
            reward -= 0.25
        elif candidate == self.door_pos and not self.has_key:
            reward -= 0.2  # bounced off the locked door
        else:
            self.agent_pos = candidate

            if self.agent_pos == self.key_pos and not self.has_key:
                self.has_key = True
                reward += 0.3

            if self.agent_pos in self.traps:
                reward -= 0.35

            if self.agent_pos == self.door_pos and self.has_key:
                reward += 0.15  # rewarding successful door unlock

            if self.agent_pos == self.goal_pos:
                terminated = True
                reward += 1.0

        if self.step_count >= self.max_steps:
            truncated = True

        potential_after = self._potential(self.agent_pos, self.has_key)
        reward += self._shaping_weight * (potential_after - potential_before)

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        x, y = self.agent_pos
        return np.array([
            x / (self.width - 1),
            y / (self.height - 1),
            1.0 if self.has_key else 0.0,
        ], dtype=np.float32)

    def _get_info(self) -> Dict:
        return {
            "position": self.agent_pos,
            "has_key": self.has_key,
            "steps": self.step_count,
        }

    def render(self):
        if self.render_mode is None:
            return None

        if pygame is None:
            raise RuntimeError("pygame is required for rendering but is not available.")

        if self._canvas is None:
            pygame.init()
            canvas_size = (self.width * self._cell_size, self.height * self._cell_size)
            self._canvas = pygame.Surface(canvas_size)
            if self.render_mode == "human":
                self._window = pygame.display.set_mode(canvas_size)
                pygame.display.set_caption("Maze Key-Door Environment")
            self._clock = pygame.time.Clock()

        self._canvas.fill((30, 30, 30))
        colors = {
            'floor': (220, 220, 220),
            'wall': (40, 40, 40),
            'start': (100, 200, 250),
            'goal': (120, 200, 120),
            'key': (240, 200, 60),
            'door_locked': (150, 80, 20),
            'door_open': (200, 150, 90),
            'trap': (200, 80, 80),
        }

        for y, row in enumerate(self.layout):
            for x, cell in enumerate(row):
                rect = pygame.Rect(x * self._cell_size, y * self._cell_size, self._cell_size, self._cell_size)
                base_color = colors['floor']
                if (x, y) in self.walls:
                    base_color = colors['wall']
                elif (x, y) == self.start_pos:
                    base_color = colors['start']
                elif (x, y) == self.goal_pos:
                    base_color = colors['goal']
                elif (x, y) == self.key_pos and not self.has_key:
                    base_color = colors['key']
                elif (x, y) == self.door_pos:
                    base_color = colors['door_open'] if self.has_key else colors['door_locked']
                elif (x, y) in self.traps:
                    base_color = colors['trap']

                pygame.draw.rect(self._canvas, base_color, rect)
                pygame.draw.rect(self._canvas, (0, 0, 0), rect, 1)

        ax = self.agent_pos[0] * self._cell_size + self._cell_size // 2
        ay = self.agent_pos[1] * self._cell_size + self._cell_size // 2
        radius = int(self._cell_size * 0.35)
        pygame.draw.circle(self._canvas, (50, 50, 200), (ax, ay), radius)
        if self.has_key:
            pygame.draw.circle(self._canvas, (255, 255, 0), (ax, ay), max(8, radius // 2), 2)

        if self.render_mode == "human":
            pygame.event.pump()
            assert self._window is not None
            self._window.blit(self._canvas, (0, 0))
            pygame.display.flip()
            if self._clock:
                self._clock.tick(self.metadata["render_fps"])
            return None

        frame = np.transpose(np.array(pygame.surfarray.pixels3d(self._canvas)), (1, 0, 2))
        return frame

    def close(self):
        if pygame is None:
            return
        if self._window is not None:
            pygame.display.quit()
            self._window = None
        if self._canvas is not None:
            pygame.quit()
            self._canvas = None
            self._clock = None

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _potential(self, position: Tuple[int, int], has_key_flag: bool) -> float:
        target = self.goal_pos if has_key_flag else self.key_pos
        return -float(self._manhattan(position, target))
