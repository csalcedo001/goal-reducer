"""
Compare DRL and GOLSAv2 w/ DRL for the girdworld task.

"""
import copy
import itertools
import platform
import pprint
import random
import time
from collections import defaultdict
from functools import partial

import click
import gymnasium as gym
import ipdb  # noqa: F401
import numpy as np
import tasks  # noqa: F401
import tianshou as ts
import torch
from models import VAEGoalReducer  # noqa
from neural_nets import GoalReducer, ImgObsAgentNet, WorldModel
from policy_w_RL import GOLSAv2Continuous
from tasks.minigrids.common import _get_valid_pos

# from policy_wo_RL import GOLSAv2woRL
from tianshou.data import Batch
from tianshou.utils.net.common import MLP

# from golsav2_noRL import GOLSAv2NoRL4GW
from utils import before_run, stdout_redirected

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GOLSAv2DDPG4RA(GOLSAv2Continuous):
    def analyze(
        self,
        env,
        s_combined: torch.Tensor,
        g_combined: torch.Tensor,
        dis_all: torch.Tensor,
        ep_passed: int,
    ):
        pass


def create_golsa_DDPG_policy(
    env,
    log_path,
    subgoal_on: bool,
    subgoal_planning: bool,
    sampling_strategy: int,
    state_rep_dim=32,
    qh_dim=512,
    lr=1e-4,
    d_kl_c=1.0,
    gamma=0.99,
    target_update_freq=320,
    device="cpu",
):
    from typing import Any, Dict, Optional, Tuple, Union
    from tianshou.utils.net.continuous import Actor, Critic

    critic_lr = 1e-3
    actor_lr = 1e-3
    action_shape = env.action_space.shape or env.action_space.n

    class Net(torch.nn.Module):
        def __init__(self, obs_shape, goal_shape, act_shape=0, hidden_sizes=256, device="cpu"):
            super().__init__()
            self.device = device
            self.model = torch.nn.Sequential(
                torch.nn.Linear(obs_shape + goal_shape + act_shape, hidden_sizes),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_sizes, hidden_sizes),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_sizes, hidden_sizes),
                torch.nn.ReLU(),
            )

            self.output_dim = hidden_sizes

        def forward(self, obs, state=None, info={}):
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            obs = obs.reshape(obs.shape[0], -1)
            return self.model(obs), state

    class SpcActor(Actor):
        min_scale: float = 1e-3
        LOG_SIG_MIN = -20
        LOG_SIG_MAX = 2

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            input_dim = getattr(self.preprocess, "output_dim")
            self.logstd_linear = MLP(
                input_dim,  # type: ignore
                self.output_dim,
                device=self.device,
            )

        def gen_act_dist(self, x, state: Any = None):
            logits, hidden = self.preprocess(x, state)
            mean = self.last(logits)
            log_std = self.logstd_linear(logits)
            std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX).exp()
            return (mean, std), hidden

        def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Any = None,
            info: Dict[str, Any] = {},
        ) -> Tuple[torch.Tensor, Any]:
            """Mapping: obs -> logits -> action."""
            state_inputs = torch.tensor(np.concatenate([obs["observation"], obs["desired_goal"]], axis=-1)).float().to(self.device)
            (mean, std), hidden = self.gen_act_dist(state_inputs, state)
            return (mean, std), hidden

    class SpcCritic(Critic):
        def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            act: Optional[Union[np.ndarray, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
        ) -> torch.Tensor:
            """Mapping: (s, a) -> logits -> Q(s, a)."""
            state_inputs = torch.tensor(np.concatenate([obs["observation"], obs["desired_goal"]], axis=-1)).float().to(self.device)
            if isinstance(act, np.ndarray):
                act = torch.tensor(act).float().to(self.device)

            obs = torch.cat([state_inputs, act], dim=1)
            logits, hidden = self.preprocess(obs)
            logits = self.last(logits)
            return logits

    obs_shape = 6
    goal_shape = 3
    net_a = Net(obs_shape, goal_shape, hidden_sizes=qh_dim, device=device)
    actor = SpcActor(net_a, action_shape, max_action=1, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)

    critic = SpcCritic(
        Net(
            obs_shape,
            goal_shape,
            act_shape=np.prod(action_shape),
            hidden_sizes=qh_dim,
            device=device,
        ),
        device=device,
    ).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    critic2 = SpcCritic(
        Net(
            obs_shape,
            goal_shape,
            act_shape=np.prod(action_shape),
            hidden_sizes=qh_dim,
            device=device,
        ),
        device=device,
    ).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    # goal reducer
    goal_reducer = GoalReducer(obs_shape, [64, 32], goal_dim=goal_shape)
    goal_reducer.to(device)
    goal_reducer_optim = torch.optim.Adam(
        [
            {"params": goal_reducer.parameters()},
        ],
        lr=lr,
    )

    policy = GOLSAv2DDPG4RA(
        env,
        actor,
        actor_optim,
        critic,
        critic_optim,
        critic2,
        critic2_optim,
        goal_reducer,
        goal_reducer_optim,
        subgoal_on,
        subgoal_planning,
        sampling_strategy,
        max_steps=env.max_steps,
        discount_factor=gamma,
        d_kl_c=d_kl_c,
        estimation_step=1,
        target_update_freq=target_update_freq,
        log_path=log_path,
        device=device,
    )
    return policy


@click.group()
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("--gpuid", default=0, type=int, help="GPU id.")
@click.pass_context
def cli(ctx, seed=None, gpuid=0):
    global device
    if seed is None:
        seed = int(time.time())

    print("seed is set to: ", seed)
    if torch.cuda.is_available():
        device = f"cuda:{gpuid}"

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ctx.obj = {
        "seed": seed,
        "machine": platform.node(),
        "proj": "GOLSAv2RobotArmFetch",
    }


def prepare4analysis(env_name, max_steps, init_noise_scale):
    with stdout_redirected():
        env = gym.make(
            env_name,
            max_steps=max_steps,
            init_noise_scale=init_noise_scale,
        )
    with torch.no_grad():
        robotarm_reach_data_all = torch.load("local_data/.robotarm_reach_data_all.pt").float().to(device)
        s_combined = robotarm_reach_data_all[:, :6]
        g_combined = robotarm_reach_data_all[:, 6:-1]
        dis_all = robotarm_reach_data_all[:, -1]

        good_indices = torch.where(dis_all > 0)[0]
        g_combined = g_combined[good_indices]
        s_combined = s_combined[good_indices]
        dis_all = dis_all[good_indices]

    return env, s_combined, g_combined, dis_all, good_indices


def train_policy(policy_name, lr,
                 env_name, max_steps, init_noise_scale,
                 training_num, test_num,
                 batch_size, epochs, stepperepoch, step_per_collect,
                 logger, log_path, run_analysis, seed):
    print("seed is set to ", seed)

    buffer_size = 20000
    gamma = 0.9
    sampling_strategy = 4
    d_kl_c = 0.5
    qh_dim = 128
    state_rep_dim = 32
    target_update_freq = 100

    env, s_combined, g_combined, dis_all, good_indices = prepare4analysis(env_name, max_steps, init_noise_scale)

    with stdout_redirected():
        train_envs = ts.env.SubprocVectorEnv(
            [
                partial(
                    gym.make,
                    env_name,
                    init_noise_scale=init_noise_scale,
                    max_steps=max_steps,
                    seed=seed + _env_idx * training_num + 1000,
                )
                for _env_idx in range(training_num)
            ]
        )
        test_envs = ts.env.SubprocVectorEnv(
            [
                partial(
                    gym.make,
                    env_name,
                    init_noise_scale=init_noise_scale,
                    max_steps=max_steps,
                    seed=seed + _env_idx * test_num + 2000,
                )
                for _env_idx in range(test_num)
            ]
        )
    subgoal_planning = True
    if policy_name == 'RL':
        subgoal_on = False
        policy = create_golsa_DDPG_policy(
            env,
            log_path,
            subgoal_on,
            subgoal_planning,
            sampling_strategy,
            state_rep_dim=state_rep_dim,
            qh_dim=qh_dim,
            lr=lr,
            d_kl_c=d_kl_c,
            gamma=gamma,
            target_update_freq=target_update_freq,
            device=device,
        )
        print("use RL")

    elif policy_name == 'GOLSAv2-w-RL':
        subgoal_on = True
        policy = create_golsa_DDPG_policy(
            env,
            log_path,
            subgoal_on,
            subgoal_planning,
            sampling_strategy,
            state_rep_dim=state_rep_dim,
            qh_dim=qh_dim,
            lr=lr,
            d_kl_c=d_kl_c,
            gamma=gamma,
            target_update_freq=target_update_freq,
            device=device,
        )
        print("use GOLSAv2-w-RL")

    elif policy_name == "NonRL":
        print("use NonRL")
        raise NotImplementedError("NonRL is not implemented yet")
    else:
        raise ValueError(f"Unknown policy name: {policy_name}")

    def compute_reward_fn(ag: np.ndarray, g: np.ndarray):
        return env.task.compute_reward(ag, g, {})

    vbuf = ts.data.HERVectorReplayBuffer(
        buffer_size,
        len(train_envs),
        compute_reward_fn=compute_reward_fn,
        horizon=max_steps,
        future_k=2,
    )
    train_collector = ts.data.Collector(policy, train_envs, vbuf, exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    def after_train(epoch, env_step):
        if run_analysis is True:
            policy.analyze(
                env,
                s_combined,
                g_combined,
                dis_all,
                epoch,
            )
        policy.train()

    policy.train()
    result = ts.trainer.offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=epochs,
        step_per_epoch=stepperepoch,
        step_per_collect=step_per_collect,
        update_per_step=0.3,
        episode_per_test=len(test_envs),
        batch_size=batch_size,
        # train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=after_train,
        stop_fn=lambda mean_rewards: mean_rewards >= env.stop_rew,
        test_in_train=False,
        logger=logger,
    )

    policy.finish_train()

    return result


@cli.command
@click.pass_context
@click.option('--env-name', '-e', help="Environment name")
@click.option('--policy', help="Policy algorithm name", default="DQL")
@click.option('--debug', default=True, help="Debug mode")
@click.option('--extra', default="-Subgoal", help="extra info written into log titles")
@click.option('--analyze', default=False, help="run analysis after each epoch")
@click.option('--maxsteps', default=140, help='Max interaction steps for a single episode in the environment.')
@click.option('--initnoisescale', default=0.05, help='Goal noise level')
@click.option('--trainingnum', default=10, help='Number of training environments.')
@click.option('--testnum', default=100, help='Number of test environments.')
@click.option('--batchsize', default=64, help='Batch size for training.')
@click.option('--epochs', default=100, help='Number of epochs to train.')
@click.option('--stepperepoch', default=10000, help='Number of interaction steps per collect in the environment.')
@click.option('--steppercollect', default=10, help='Number of interaction steps per epoch.')
@click.option('--lr', default=1e-3, help='Learning rate.')
def train(ctx, env_name, policy, debug, extra, analyze,
          maxsteps, initnoisescale, trainingnum, testnum,
          batchsize, epochs, stepperepoch, steppercollect, lr,
          ):

    seed = ctx.obj['seed']
    logger, log_path = before_run(ctx,
                                  env_name,
                                  policy,
                                  debug,
                                  extra)
    train_policy(policy, lr,
                 env_name, maxsteps, initnoisescale,
                 trainingnum, testnum,
                 batchsize, epochs, stepperepoch, steppercollect,
                 logger, log_path, run_analysis=analyze, seed=seed)
    print(f'log_path: {log_path}')


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
