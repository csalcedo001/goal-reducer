import queue
import threading
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union  # noqa

import numpy as np
import pathos.multiprocessing as mp
import tianshou as ts
import torch
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy.base import _nstep_return


def discrete_trj_remove_loop(
    pool,
    indicess: np.ndarray,
    buffer: ReplayBuffer,
):
    pass


class EpMemSamplingStrategy(Enum):
    random = 1  # Random sampling. The original approach by Chane et al.
    trajectory = 2  # Sample from past trajectories
    noloop = 3  # Noloop sampling
    prt_noloop = 4  # Quasi-metric noloop


class PolicyBase(ts.policy.BasePolicy):
    pool = mp.Pool(mp.cpu_count())
    eps = 1e-16

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def _setup_queue(self) -> None:
        self.qe = queue.Queue()
        self.thread = threading.Thread(target=self.worker)
        self.thread.start()

    def worker(self):
        while True:
            task = self.qe.get()
            if task is None:
                break
            task()
            self.qe.task_done()

    def after_train(self, epoch_passed: int) -> None:
        pass

    def finish_train(self) -> None:
        self.qe.join()
        # stop the worker
        self.qe.put(None)
        self.thread.join()

    def _setup_log(self) -> None:
        n_count = 0
        self.policy_dir = self.log_path / 'policy'
        self.fig_dir = self.log_path / 'fig'
        while self.policy_dir.exists() or self.fig_dir.exists():
            n_count += 1
            self.policy_dir = self.log_path / f'policy.{n_count}'
            self.fig_dir = self.log_path / f'fig.{n_count}'
        self.policy_dir.mkdir()
        self.fig_dir.mkdir()

    def train(self, mode: bool = True) -> "PolicyBase":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        b = np.array([False, True], dtype=np.bool_)
        i64 = np.array([[0, 1]], dtype=np.int64)
        _nstep_return(f64, b, f32.reshape(-1, 1), i64, 0.1, 1)

    def exploration_noise(
        self,
        act: Union[np.ndarray, Batch],
        batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act

    def compute_q_value(
        self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Compute the q value based on the network's raw output and action mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    def sample_subgoals_from_replay_buffer(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray):
        """Sample subgoals from the replay buffer and write it back to
        the batch.
        """
        bsz = len(batch)
        indicess = [indices]
        for _ in range(self.max_steps - 1):
            indicess.append(buffer.next(indicess[-1]))
        indicess = np.stack(indicess)
        strategy = self.sampling_strategy

        # random sampling strategy, from all possible paths
        if strategy is EpMemSamplingStrategy.random:
            terminal_indices = indicess.max(axis=0)
            subgoal_indices = buffer.sample_indices(bsz)

        elif strategy == EpMemSamplingStrategy.trajectory:
            last_indices = indicess.max(axis=0)
            terminal_indices = np.random.uniform(indicess[0], last_indices, size=bsz)
            subgoal_indices = np.random.uniform(indicess[0], terminal_indices, size=bsz)

        elif strategy == EpMemSamplingStrategy.noloop:
            subgoal_indices, terminal_indices, _ = discrete_sg_sampling_w_remove_loop(
                self.pool,
                indicess, buffer)

        elif strategy == EpMemSamplingStrategy.prt_noloop:
            subgoal_indices, terminal_indices, mask = discrete_sg_sampling_w_remove_loop(
                self.pool,
                indicess, buffer, prioritized=True)
            subgoal_indices = np.array(subgoal_indices)
            terminal_indices = np.array(terminal_indices)
            batch.subgal_weights = mask
        else:
            raise NotImplementedError

        batch.subgoal = buffer[subgoal_indices].obs.achieved_goal
        batch.final_reached_goal = buffer[terminal_indices].obs.achieved_goal
        return batch

    def sample_trjs_from_replay_buffer(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray):
        """Sample trajectories from the replay buffer and write it back to
        the batch.
        """
        indicess = [indices]
        for _ in range(self.max_steps - 1):
            indicess.append(buffer.next(indicess[-1]))
        indicess = np.stack(indicess)
        strategy = self.sampling_strategy

        if strategy == EpMemSamplingStrategy.noloop:
            trajectories = discrete_trj_remove_loop(self.pool, indicess, buffer)

        elif strategy == EpMemSamplingStrategy.prt_noloop:
            trajectories = discrete_trj_remove_loop(self.pool, indicess, buffer)
        else:
            raise NotImplementedError

        batch.trajectories = trajectories

        return batch

    def analyze(self,):
        raise NotImplementedError
