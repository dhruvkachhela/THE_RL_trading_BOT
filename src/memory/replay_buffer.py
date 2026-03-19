"""
Advanced Memory Module.

  PrioritizedReplayBuffer  — Prioritized Experience Replay with IS weights
  NStepBuffer              — accumulates N-step returns before pushing to PER
  SegmentTree              — efficient O(log N) PER using sum/min segment trees
"""

import numpy as np
from collections import deque
from typing import Tuple, List


# ──────────────────────────────────────────────────────────────
#  Segment Tree (for efficient O(log N) PER)
# ──────────────────────────────────────────────────────────────

class SumSegmentTree:
    """Binary sum-tree for O(log n) priority sum queries."""

    def __init__(self, capacity: int):
        self._cap = capacity
        self._tree = np.zeros(2 * capacity, dtype=np.float64)

    def _propagate(self, idx: int) -> None:
        parent = idx >> 1
        while parent >= 1:
            self._tree[parent] = self._tree[2 * parent] + self._tree[2 * parent + 1]
            parent >>= 1

    def update(self, idx: int, val: float) -> None:
        idx += self._cap
        self._tree[idx] = val
        self._propagate(idx)

    def total(self) -> float:
        return self._tree[1]

    def retrieve(self, s: float) -> int:
        """Find leaf index whose cumulative sum ≥ s."""
        idx = 1
        while idx < self._cap:
            left = 2 * idx
            if self._tree[left] > s:
                idx = left
            else:
                s -= self._tree[left]
                idx = left + 1
        return idx - self._cap


class MinSegmentTree:
    """Binary min-tree for O(log n) minimum priority queries."""

    def __init__(self, capacity: int):
        self._cap = capacity
        self._tree = np.full(2 * capacity, float("inf"), dtype=np.float64)

    def _propagate(self, idx: int) -> None:
        parent = idx >> 1
        while parent >= 1:
            self._tree[parent] = min(self._tree[2 * parent], self._tree[2 * parent + 1])
            parent >>= 1

    def update(self, idx: int, val: float) -> None:
        idx += self._cap
        self._tree[idx] = val
        self._propagate(idx)

    def minimum(self) -> float:
        return self._tree[1]


# ──────────────────────────────────────────────────────────────
#  Prioritized Replay Buffer
# ──────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay with segment-tree sampling.

    Improvements over list-based PER:
      - O(log N) push / sample via segment trees
      - Correct IS weight normalisation
      - Epsilon clamp on priorities (avoids zeros)
    """

    EPSILON = 1e-6

    def __init__(self, capacity: int, alpha: float = 0.6):
        # Capacity must be power of two for segment tree
        cap = 1
        while cap < capacity:
            cap <<= 1
        self._cap   = cap
        self._alpha = alpha
        self._pos   = 0
        self._size  = 0

        self._sum_tree = SumSegmentTree(cap)
        self._min_tree = MinSegmentTree(cap)
        self._max_priority = 1.0

        self._states      = [None] * cap
        self._actions     = np.zeros(cap, dtype=np.int64)
        self._rewards     = np.zeros(cap, dtype=np.float32)
        self._next_states = [None] * cap
        self._dones       = np.zeros(cap, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self._pos
        self._states[idx]      = state
        self._actions[idx]     = action
        self._rewards[idx]     = reward
        self._next_states[idx] = next_state
        self._dones[idx]       = float(done)

        priority = self._max_priority ** self._alpha
        self._sum_tree.update(idx, priority)
        self._min_tree.update(idx, priority)

        self._pos  = (self._pos + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = self._sample_indices(batch_size)

        states      = np.stack([self._states[i]      for i in indices])
        next_states = np.stack([self._next_states[i] for i in indices])
        actions     = self._actions[indices]
        rewards     = self._rewards[indices]
        dones       = self._dones[indices]

        # IS weights
        min_prob = self._min_tree.minimum() / self._sum_tree.total()
        max_weight = (min_prob * self._size) ** (-beta)

        probs   = np.array([self._sum_tree._tree[i + self._cap] for i in indices]) / self._sum_tree.total()
        weights = (probs * self._size) ** (-beta) / max_weight
        weights = np.clip(weights, 0.0, 1.0).astype(np.float32)

        return states, actions, rewards, next_states, dones, np.array(indices), weights

    def _sample_indices(self, batch_size: int) -> List[int]:
        total = self._sum_tree.total()
        segment = total / batch_size
        indices = []
        for i in range(batch_size):
            s = np.random.uniform(i * segment, (i + 1) * segment)
            idx = self._sum_tree.retrieve(s)
            idx = min(idx, self._size - 1)
            indices.append(idx)
        return indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, p in zip(indices, priorities):
            p = max(float(p) + self.EPSILON, self.EPSILON)
            self._max_priority = max(self._max_priority, p)
            pa = p ** self._alpha
            self._sum_tree.update(int(idx), pa)
            self._min_tree.update(int(idx), pa)

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def __len__(self) -> int:
        return self._size


# ──────────────────────────────────────────────────────────────
#  N-Step Return Buffer
# ──────────────────────────────────────────────────────────────

class NStepBuffer:
    """
    Collects N consecutive transitions and computes discounted n-step return.

    R_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{n-1}·r_{t+n-1}

    On flush, pushes (s_t, a_t, R_t, s_{t+n}, done) into the PER buffer.
    """

    def __init__(self, n: int, gamma: float, per_buffer: PrioritizedReplayBuffer):
        self.n      = n
        self.gamma  = gamma
        self.buffer = per_buffer
        self._buf: deque = deque(maxlen=n)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((state, action, reward, next_state, done))
        if len(self._buf) == self.n:
            self._flush_oldest()
        if done:
            # Flush remaining transitions
            while self._buf:
                self._flush_oldest()

    def _flush_oldest(self) -> None:
        if not self._buf:
            return
        state_0, action_0, _, _, _ = self._buf[0]

        # Compute discounted n-step reward
        R = 0.0
        gamma_k = 1.0
        last_done = False
        last_next = None
        for _, _, r, ns, d in self._buf:
            R += gamma_k * r
            gamma_k *= self.gamma
            last_next = ns
            last_done = d
            if d:
                break

        self.buffer.push(state_0, action_0, R, last_next, last_done)
        self._buf.popleft()
