"""
Compare DRL, GOLSAv2 w/ DRL, GOLSAv2 w/o RL for the girdworld task.

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
import tianshou as ts
import torch

import tasks  # noqa: F401
from models import VAEGoalReducer  # noqa
# from golsav2_noRL import GOLSAv2NoRL4GW
from utils import before_run
from tasks.minigrids.common import _get_valid_pos
from neural_nets import ImgObsAgentNet, GoalReducer, WorldModel
from policy_w_RL import GOLSAv2DQL
from policy_wo_RL import GOLSAv2woRL
from tianshou.data import Batch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_batch_goal(goal: Batch) -> np.ndarray:
    return goal['image']


class GOLSAv2DQL4GW(GOLSAv2DQL):

    def analyze(self,
                env,
                all_possible_img_inputs,
                shortest_distance_state_goal_pairs,
                all_possible_idx,
                qvs_ids,
                all_possible_idx_rev,
                random_subgoal_distance,
                random_subgoal_distance_err,
                ep_passed: int):
        """Run analysis for gridworld task after each epoch"""
        # raise NotImplementedError
        pass


class GOLSAv2woRL4GW(GOLSAv2woRL):

    def analyze(self,
                env,
                all_possible_img_inputs,
                shortest_distance_state_goal_pairs,
                all_possible_idx,
                qvs_ids,
                all_possible_idx_rev,
                random_subgoal_distance,
                random_subgoal_distance_err,
                ep_passed: int):
        """Run analysis for gridworld task after each epoch"""
        # raise NotImplementedError
        pass


def create_golsa_DQL_policy(env, log_path, subgoal_on: bool,
                            sampling_strategy: int,
                            subgoal_planning: bool = True,
                            state_rep_dim=32, qh_dim=512,
                            lr=1e-4, d_kl_c=1.0, gamma=0.99,
                            encoding_lr=None,
                            target_update_freq=320,
                            device='cpu'):
    action_shape = env.action_space.shape or env.action_space.n

    # q network
    net = ImgObsAgentNet(env.observation_space, action_shape, s_dim=state_rep_dim, qh_dim=qh_dim,
                         device=device)
    net.to(device)
    if encoding_lr is None:
        encoding_lr = lr / 10.  # for agv=13 size=15

    optim = torch.optim.Adam(
        [
            {'params': net.qnet.parameters()},
            {'params': net.encoding_layers.parameters(), 'lr': encoding_lr},
        ],
        lr=lr,
    )
    # goal reducer
    goal_reducer = GoalReducer(state_rep_dim, [64, 32])
    goal_reducer.to(device)
    goal_reducer_optim = torch.optim.Adam([
        {'params': goal_reducer.parameters()},
        {'params': net.encoding_layers.parameters(), 'lr': lr / 5.0}
    ], lr=lr)

    policy = GOLSAv2DQL4GW(net, optim,
                           goal_reducer, goal_reducer_optim,
                           subgoal_on,
                           subgoal_planning,
                           extract_batch_goal=extract_batch_goal,
                           sampling_strategy=sampling_strategy,
                           max_steps=env.max_steps,
                           discount_factor=gamma,
                           d_kl_c=d_kl_c,
                           estimation_step=1, target_update_freq=target_update_freq,
                           log_path=log_path,
                           device=device)
    return policy


def create_golsa_NonRL_policy4GW(env, log_path,
                                 sampling_strategy: int,
                                 state_rep_dim=32, qh_dim=512,
                                 lr=1e-4, d_kl_c=1.0, gamma=0.99,
                                 encoding_lr=None,
                                 device='cpu'):
    """Create GR+local policy for gridworld task"""
    action_shape = env.action_space.shape or env.action_space.n

    # q network
    net = ImgObsAgentNet(env.observation_space, action_shape, s_dim=state_rep_dim, qh_dim=qh_dim,
                         device=device, limited_output=True)
    net.to(device)
    if encoding_lr is None:
        encoding_lr = lr / 10.  # for agv=13 size=15
        # encoding_lr = lr / 2.  # for agv=13 size=19
        # encoding_lr = 2 * lr  # for agv=7

    optim = torch.optim.Adam(
        [
            {'params': net.qnet.parameters()},
            {'params': net.encoding_layers.parameters(), 'lr': encoding_lr},
        ],
        lr=lr,
    )

    # goal reducer
    # goal_reducer = GoalReducer(state_rep_dim, [64, 32])
    goal_reducer = VAEGoalReducer(
        state_rep_dim,
        hidden_dim=1024,
        latent_dim=32,
        KL_weight=6.0,
        device=device
    )

    goal_reducer.to(device)

    goal_reducer_optim = torch.optim.Adam([
        {'params': goal_reducer.parameters()},
        {'params': net.encoding_layers.parameters(), 'lr': encoding_lr}
    ], lr=lr)

    world_model = WorldModel(state_rep_dim, action_shape)
    world_model.to(device)
    world_model_optim = torch.optim.Adam([
        {'params': world_model.parameters()},
        {'params': net.encoding_layers.parameters(), 'lr': encoding_lr},
        {'params': net.decoding_layers.parameters(), 'lr': lr},
    ], lr=lr)

    policy = GOLSAv2woRL4GW(net, optim,
                            goal_reducer, goal_reducer_optim,
                            world_model, world_model_optim,
                            extract_batch_goal=extract_batch_goal,
                            sampling_strategy=sampling_strategy,
                            max_steps=env.unwrapped.max_steps,
                            discount_factor=gamma,
                            d_kl_c=d_kl_c,
                            estimation_step=1,
                            log_path=log_path,
                            device=device,
                            n_ep_explore=3)
    return policy


@click.group()
@click.option('--seed', default=None, type=int, help='Random seed.')
@click.pass_context
def cli(ctx, seed=None):
    if seed is None:
        seed = int(time.time())
        print('seed(None) is set to: ', seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ctx.obj = {
        'seed': seed,
        'machine': platform.node(),
        'proj': 'GOLSAv2GridWorld',
    }


def prepare4analysis(env_name, max_steps, agent_view_size):
    env = gym.make(env_name, max_steps=max_steps,
                   agent_view_size=agent_view_size)
    available_pos = _get_valid_pos(copy.deepcopy(env.unwrapped.walls))
    all_possible_obs = []
    all_possible_idx = {}  # (pos) -> idx

    for pos_idx, pos in enumerate(available_pos):
        goal_pos = available_pos[pos_idx - 1]
        env_specific = gym.make(env_name, max_steps=max_steps,
                                agent_pos=pos, goal_pos=goal_pos,
                                agent_view_size=agent_view_size)

        obs, info = env_specific.reset()
        all_possible_obs.append(obs['observation'].copy())
        all_possible_idx[info['agent_pos']] = len(all_possible_obs) - 1
    #  idx -> (pos, dir)
    all_possible_idx_rev = {v: k for k, v in all_possible_idx.items()}

    all_possible_obs = np.swapaxes(np.stack(all_possible_obs), -3, -1)
    all_possible_img_inputs = torch.tensor(all_possible_obs, dtype=torch.float).to(device)

    shortest_distance_state_goal_pairs = defaultdict(list)
    qvs_ids = {}  # ((s_pos, s_dir), (g_pos, g_dir)) -> q idx
    sidx = 0
    random_subgoal_distance = defaultdict(list)
    for obs_idx_s, obs_idx_g in itertools.permutations(np.arange(len(all_possible_img_inputs)), 2):
        distance = env.unwrapped.shortest_distance(all_possible_idx_rev[obs_idx_s], all_possible_idx_rev[obs_idx_g])
        shortest_distance_state_goal_pairs[distance].append(
            (all_possible_idx_rev[obs_idx_s],
                all_possible_idx_rev[obs_idx_g])
        )
        avg_subgoal_distance = []
        for obs_idx_sub_g in range(len(all_possible_img_inputs)):
            if obs_idx_sub_g == obs_idx_s or obs_idx_sub_g == obs_idx_g:
                continue
            avg_subgoal_distance.append(
                (env.unwrapped.shortest_distance(
                    all_possible_idx_rev[obs_idx_s],
                    all_possible_idx_rev[obs_idx_sub_g]) + env.unwrapped.shortest_distance(
                    all_possible_idx_rev[obs_idx_sub_g],
                    all_possible_idx_rev[obs_idx_g])
                 ) / distance
            )

        random_subgoal_distance[distance].append(np.mean(avg_subgoal_distance))

        qvs_ids[(all_possible_idx_rev[obs_idx_s],
                all_possible_idx_rev[obs_idx_g])] = sidx

        sidx += 1
        pass

    random_subgoal_distance = {k: np.mean(v) for k, v in random_subgoal_distance.items()}
    random_subgoal_distance_err = {k: np.std(v) for k, v in random_subgoal_distance.items()}

    return (env,
            all_possible_img_inputs,
            shortest_distance_state_goal_pairs,
            all_possible_idx,
            qvs_ids,
            all_possible_idx_rev,
            random_subgoal_distance,
            random_subgoal_distance_err)


def train_policy(
        policy_name, lr,
        env_name, max_steps, agent_view_size,
        training_num, test_num,
        batch_size, epochs, stepperepoch, step_per_collect,
        logger, log_path, run_analysis, seed
):
    # other parameters
    buffer_size = 20000
    gamma = 0.9
    d_kl_c = 0.05
    sampling_strategy = 4
    state_rep_dim = 32
    target_update_freq = 100
    qh_dim = 128

    # prepare for training
    (env,
     all_possible_img_inputs,
     shortest_distance_state_goal_pairs,
     all_possible_idx,
     qvs_ids,
     all_possible_idx_rev,
     random_subgoal_distance,
     random_subgoal_distance_err) = prepare4analysis(env_name, max_steps, agent_view_size)

    # get tasks ready

    train_envs = ts.env.SubprocVectorEnv(
        [partial(
            gym.make,
            env_name,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            seed=seed + _env_idx * training_num + 1000,

        ) for _env_idx in range(training_num)])
    test_envs = ts.env.SubprocVectorEnv(
        [partial(
            gym.make,
            env_name,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            seed=seed + _env_idx * test_num + 2000,

        ) for _env_idx in range(test_num)])
    vbuf = ts.data.VectorReplayBuffer(buffer_size, len(train_envs))

    # =============== get networks ready ===============

    if policy_name == 'RL':
        policy = create_golsa_DQL_policy(env, log_path, False,
                                         sampling_strategy,
                                         state_rep_dim=state_rep_dim,
                                         qh_dim=qh_dim,
                                         lr=lr, d_kl_c=d_kl_c, gamma=gamma,
                                         target_update_freq=target_update_freq,
                                         device=device)

    elif policy_name == 'GOLSAv2-w-RL':
        policy = create_golsa_DQL_policy(env, log_path, True,
                                         sampling_strategy,
                                         state_rep_dim=state_rep_dim,
                                         qh_dim=qh_dim,
                                         lr=lr, d_kl_c=d_kl_c, gamma=gamma,
                                         target_update_freq=target_update_freq,
                                         device=device)

    elif policy_name == 'GOLSAv2-wo-RL':
        policy = create_golsa_NonRL_policy4GW(env, log_path,
                                              sampling_strategy,
                                              state_rep_dim=state_rep_dim,
                                              qh_dim=qh_dim,
                                              lr=lr, d_kl_c=d_kl_c, gamma=gamma,
                                              device=device)
    else:
        raise ValueError(f'Unknown policy name: {policy_name}')

    train_collector = ts.data.Collector(policy, train_envs, vbuf, exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    def after_train(epoch, env_step):
        policy.set_eps(0.05)
        policy.after_train(epoch)
        if run_analysis is True:
            obs_reps = policy.model.encoding_layers(all_possible_img_inputs)
            torch.save(obs_reps.data.cpu(), log_path / f'obs_reps_{epoch}.pth')
            policy.analyze(
                env,
                all_possible_img_inputs,
                shortest_distance_state_goal_pairs,
                all_possible_idx,
                qvs_ids,
                all_possible_idx_rev,
                random_subgoal_distance,
                random_subgoal_distance_err,
                epoch
            )
            policy.train()

    policy.train()
    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=epochs, step_per_epoch=stepperepoch, step_per_collect=step_per_collect,
        update_per_step=0.1, episode_per_test=len(test_envs),
        batch_size=batch_size,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
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
@click.option('--agentviewsize', default=13, help='Agent view size, must be odd number.')
@click.option('--trainingnum', default=10, help='Number of training environments.')
@click.option('--testnum', default=100, help='Number of test environments.')
@click.option('--batchsize', default=64, help='Batch size for training.')
@click.option('--epochs', default=100, help='Number of epochs to train.')
@click.option('--stepperepoch', default=10000, help='Number of interaction steps per collect in the environment.')
@click.option('--steppercollect', default=10, help='Number of interaction steps per epoch.')
@click.option('--lr', default=1e-3, help='Learning rate.')
def train(ctx, env_name, policy, debug, extra, analyze,
          maxsteps, agentviewsize, trainingnum, testnum,
          batchsize, epochs, stepperepoch, steppercollect, lr,
          ):
    seed = ctx.obj['seed']
    logger, log_path = before_run(ctx,
                                  env_name,
                                  policy,
                                  debug,
                                  extra)
    train_policy(
        policy, lr,
        env_name, maxsteps, agentviewsize,
        trainingnum, testnum,
        batchsize, epochs, stepperepoch, steppercollect,
        logger, log_path, run_analysis=analyze, seed=seed
    )
    print(f'log_path: {log_path}')


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
