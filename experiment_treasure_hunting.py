import pathos.multiprocessing as mp
from scipy.io import loadmat, savemat
from rsatoolbox.util.searchlight import (
    evaluate_models_searchlight,
    get_searchlight_RDMs,
    get_volume_searchlight,
)
from rsatoolbox.rdm import RDMs
from rsatoolbox.model import ModelFixed
from rsatoolbox.inference import eval_fixed
from rsatoolbox.data import load_dataset
from nilearn.image import new_img_like, concat_imgs, mean_img
from nilearn.glm.second_level import SecondLevelModel
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors
from collections import defaultdict
import copy
from pathlib import Path
import shutil
import pandas as pd

import click
import gymnasium as gym
import platform
import time
import random
import numpy as np
import tasks  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from neural_nets import GoalReducer
from tasks.treasure_hunting.treasurehunting import TreasureHuntEnv
from utils import before_run, get_RDM, get_git_branch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        "proj": "GOLSAv2",
    }


def gen_all_s_g_sg_embs(env: TreasureHuntEnv):
    s_embs = []
    sg_embs = []
    g_embs = []
    acts = []
    for agent_id in env.loc_neighbors.keys():
        for key_id in env.loc_neighbors[agent_id]:
            for chest_id in env.loc_neighbors[key_id]:
                acts.append(
                    env.dir2act[env.optimal_1step_acts[(agent_id, key_id)]]
                )
                s_embs.append(
                    env.obs_info[agent_id]["emb"]
                )
                sg_embs.append(
                    env.ach_goal_info[(key_id, key_id)]["emb"]
                )
                g_embs.append(
                    env.ach_goal_info[(key_id, chest_id)]["emb"]
                )
                pass
    return np.array(acts), np.stack(s_embs), np.stack(sg_embs), np.stack(sg_embs)


class L2NormLayer(nn.Module):
    def __init__(self, eps=1e-10):
        super(L2NormLayer, self).__init__()
        self.eps = eps
        self.noise_level = 0.1

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x_normalized = 2. * x / norm
        x_normalized += self.noise_level * torch.randn_like(x_normalized)
        return x_normalized


class Enc(nn.Module):
    def __init__(self, obs_shap, h_dim, o_dim, output_unit=False):
        super().__init__()
        self.fc1 = nn.Linear(obs_shap, h_dim)
        self.fc2 = nn.Linear(h_dim, o_dim)
        if output_unit:
            self.last = L2NormLayer()
        else:
            self.last = nn.Identity()

    def forward(self, x, return_h=False):
        h = F.relu(self.fc1(x))
        o = self.last(F.tanh(self.fc2(h)))
        if return_h:
            return o, h
        else:
            return o


class Qnet(nn.Module):
    def __init__(self, state_dim, goal_dim, h_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim+goal_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, a_dim))

    def forward(self, x, return_h=False):
        h = F.relu(self.fc1(x))
        o = F.softmax(self.fc2(h), dim=-1)
        if return_h:
            return o, h
        else:
            return o


def create_policy(env: TreasureHuntEnv, state_dim, goal_dim, h_dim, lr, peak_p=0.7):
    encoding_lr = lr/10.
    obs_encoder = Enc(
        env.observation_space['observation'].shape[0],
        h_dim, state_dim, output_unit=True)
    obs_encoder.to(device)
    obs_decoder = nn.Sequential(
        nn.Linear(state_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, env.observation_space['observation'].shape[0]),
    )
    obs_decoder.to(device)

    obs_rep_opt = optim.Adam(
        [{'params': obs_encoder.parameters(), 'lr': encoding_lr},
         {'params': obs_decoder.parameters(), 'lr': encoding_lr},]
    )

    goal_encoder = Enc(
        env.observation_space['desired_goal'].shape[0],
        h_dim, goal_dim, output_unit=True)
    goal_encoder.to(device)
    goal_decoder = nn.Sequential(
        nn.Linear(goal_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, env.observation_space['desired_goal'].shape[0]),
    )
    goal_decoder.to(device)
    goal_rep_opt = optim.Adam(
        [{'params': goal_encoder.parameters(), 'lr': encoding_lr},
         {'params': goal_decoder.parameters(), 'lr': encoding_lr},]
    )

    qnet = Qnet(state_dim, goal_dim, h_dim, env.action_space.n)
    qnet.to(device)
    qnet_opt = optim.Adam(
        [
            {'params': qnet.parameters(), 'lr': lr},
        ]
    )

    goal_reducer = GoalReducer(goal_dim, [512, 256])
    goal_reducer.to(device)
    goal_reducer_opt = optim.Adam(
        [
            {'params': goal_reducer.parameters(), 'lr': lr},
        ]
    )

    act_threshold = torch.tensor([
        peak_p, (1-peak_p)/3., (1-peak_p)/3., (1-peak_p)/3.
    ]).to(device)
    entropy_threshold = (-act_threshold * torch.log(act_threshold)).sum()

    return (obs_encoder, obs_decoder, obs_rep_opt), (
        goal_encoder, goal_decoder, goal_rep_opt,
    ), (
        goal_reducer, goal_reducer_opt, qnet, qnet_opt,
    ), entropy_threshold


def analyze(log_path: Path, ep_passed: int, env: TreasureHuntEnv, obs_enc: nn.Module, goal_enc: nn.Module, qnet: nn.Module, gr: nn.Module):
    """Run analysis, plot the RDM
    """
    # specify the agent location, trial type
    akcs = []

    for agent_id in env.loc_neighbors.keys():
        # when it's at the beginning of two one-step paths.
        for key_id in env.loc_neighbors[agent_id]:
            kc = (key_id, key_id)
            akcs.append((agent_id, kc))

        # when it's at the beginning of 4 two-step paths
        for key_id in env.loc_neighbors[agent_id]:
            for chest_id in env.loc_neighbors[key_id]:
                kc = (key_id, chest_id)
                akcs.append((agent_id, kc))
        # when it's at the middle of two two-step paths
        for chest_id in env.loc_neighbors[agent_id]:
            kc = (agent_id, chest_id)
            akcs.append((agent_id, kc))

    trial_names = []
    for a, kc in akcs:
        start_state = env.loc_name_mapping[a].lower()
        k, c = kc
        if k == a:
            next_state = env.loc_name_mapping[c]
            final_state = 'None'
        else:
            next_state = env.loc_name_mapping[k]
            final_state = env.loc_name_mapping[c]
        trial_names.append(f'{start_state}{next_state}{final_state}')

    planning_reps = {
        's_enc.hidden': [],
        's_enc.out': [],
        'g_enc.hidden': [],
        'g_enc.out': [],
        'gr.hidden': [],
        'gr.out': [],
        'qnet.hidden1': [],
        'qnet.out1': [],
        'qnet.entropy': [],
    }
    acting_reps = {
        'qnet.hidden2': [],
        'qnet.out2': [],
    }
    with torch.no_grad():
        for agent_id, kc in akcs:
            # two phases
            s_emb = torch.tensor(env.obs_info[agent_id]["emb"]).to(device).float().unsqueeze(0)
            goal_emb = torch.tensor(env.ach_goal_info[kc]["emb"]).to(device).float().unsqueeze(0)
            # TODO now calculate the RDM for each neuron in the model.
            # planning

            # prompt phase
            s_rep, s_rep_h = obs_enc(s_emb, return_h=True)
            g_rep, g_rep_h = goal_enc(goal_emb, return_h=True)
            # planning phase
            sg_rep_dist, sg_h = gr(s_rep, g_rep, return_h=True)
            sg_rep = sg_rep_dist.rsample()
            act1, act1_h = qnet(torch.cat((s_rep, g_rep), dim=-1), return_h=True)
            act1 = act1.squeeze(0)
            entropy = (-act1 * torch.log(act1)).sum()
            act2, act2_h = qnet(torch.cat((s_rep, sg_rep), dim=-1), return_h=True)
            act2 = act2.squeeze(0)

            planning_reps['s_enc.hidden'].append(s_rep_h)
            planning_reps['s_enc.out'].append(s_rep)

            planning_reps['g_enc.hidden'].append(g_rep_h)
            planning_reps['g_enc.out'].append(g_rep)

            planning_reps['gr.hidden'].append(sg_h)
            planning_reps['gr.out'].append(sg_rep)

            planning_reps['qnet.hidden1'].append(act1_h)
            planning_reps['qnet.out1'].append(act1.unsqueeze(0))
            planning_reps['qnet.entropy'].append(entropy.unsqueeze(0).unsqueeze(0))

            acting_reps['qnet.hidden2'].append(act2_h)
            acting_reps['qnet.out2'].append(act2.unsqueeze(0))

        for module in planning_reps.keys():
            planning_reps[module] = torch.cat(planning_reps[module], dim=0)

        for module in acting_reps.keys():
            acting_reps[module] = torch.cat(acting_reps[module], dim=0)

        n_trials = len(akcs)
        total_RDM = torch.zeros(n_trials*3, n_trials*3).to(device)
        total_beta_names = [f'{tn}Start' for tn in trial_names] + [
            f'{tn}Prompt' for tn in trial_names] + [
                f'fb{tn}' for tn in trial_names
        ]
        noahzarr_betanames = []
        with open(Path('local_data/betanames.txt'), 'r') as f:
            for line in f:
                noahzarr_betanames.append(line.strip())
        assert len(set(total_beta_names)-set(noahzarr_betanames)) == 0
        beta_ids = []
        for bname in total_beta_names:
            beta_ids.append(noahzarr_betanames.index(bname))
        all_components = list(planning_reps.keys())+list(acting_reps.keys())
        component_RDMs = []
        for component in all_components:
            component_RDM = total_RDM.clone()
            if component in planning_reps.keys():
                component_RDM[:n_trials, :n_trials] = get_RDM(
                    planning_reps[component], planning_reps[component]
                )
            elif component in acting_reps.keys():
                component_RDM[n_trials:n_trials*2, n_trials:n_trials*2] = get_RDM(
                    acting_reps[component], acting_reps[component]
                )
            else:
                raise ValueError
            component_RDM = component_RDM[beta_ids][:, beta_ids]
            component_RDMs.append(component_RDM)
        component_RDMs = torch.stack(component_RDMs)
        RDM_info = {
            'component_RDMs': component_RDMs,
            'noahzarr_betanames': noahzarr_betanames,
            'all_components': all_components,
        }
        torch.save(RDM_info, log_path/f'RDM_info-epoch-{ep_passed}.pt')


@cli.command
@click.pass_context
@click.option('--debug', default=True, help="Debug mode")
def train(ctx, debug):
    # task part
    env_name = 'tasks.TreasureHunt'
    # model part
    uni_dim = 8
    state_dim = uni_dim
    goal_dim = uni_dim
    h_dim = 128
    lr = 1e-3
    max_T = 150000

    # stop_rew = 1.8
    stop_rew = 1.9

    replay_buffer_size = 1000
    ep_sample_size = 256
    exploration_eps = 100
    # noise_level = 0.1

    # ctx.obj[] TODO upate params

    logger, log_path = before_run(ctx,
                                  env_name,
                                  'GOLSAv2',
                                  debug,
                                  'Single')
    res_path = log_path / f"{ctx.obj['seed']}"
    if res_path.is_dir():
        shutil.rmtree(res_path)
    res_path.mkdir(exist_ok=True)

    env: TreasureHuntEnv = gym.make(
        env_name,
        seed=1,
        render_mode='vec',
    )

    (obs_encoder, obs_decoder, obs_rep_opt), (
        goal_encoder, goal_decoder, goal_rep_opt,
    ), (
        goal_reducer, goal_reducer_opt, qnet, qnet_opt,
    ), entropy_threshold = create_policy(env, state_dim, goal_dim,
                                         h_dim, lr, peak_p=0.999999)

    grep_training_on = 1
    srep_training_on = 1
    # rep_loss_threshold = 1e-4
    rep_loss_threshold = 1e-4

    replay_buffer = []

    reward_total = 0
    reward_totals = []
    act1st_is_corrects = []
    act2nd_is_corrects = []
    entropy_is_corrects = []
    episode = []
    episode_including_h = []

    episodes_all_conditions = defaultdict(list)

    analyze(res_path, len(reward_totals), env, obs_encoder, goal_encoder, qnet, goal_reducer)

    pbar = tqdm.tqdm(range(max_T))
    explore = True
    acc_bsz_until = 0
    episode_count = 0
    observation, info = env.reset()
    is1ststep = True
    # entropy_threshold = 1e-5  # don't change this
    for env_step in pbar:
        obs_encoder.eval()
        goal_encoder.eval()
        qnet.eval()
        goal_reducer.eval()
        if explore:
            action = np.random.choice(4, 1)[0]
        else:
            s_encoding, s_enc_h = obs_encoder(
                torch.tensor(observation['observation']).to(device).float().unsqueeze(0),
                return_h=True,
            )
            g_encoding, g_enc_h = goal_encoder(
                torch.tensor(observation['desired_goal']).to(device).float().unsqueeze(0),
                return_h=True,
            )
            act1, act1_h = qnet(
                torch.cat((s_encoding, g_encoding), dim=-1),
                return_h=True,
            )
            act1 = act1.squeeze(0)

            entropy = (-act1 * torch.log(act1)).sum()
            if entropy < entropy_threshold:
                action = torch.multinomial(act1, 1, replacement=False)[0].item()
                subgoal_encoding = torch.zeros_like(g_encoding)
                sg_h = torch.zeros((subgoal_encoding.shape[0], goal_reducer.layer_sizes[-1])
                                   ).to(subgoal_encoding.device)

                act2 = act1
            else:
                # subgoal_encoding = goal_reducer(s_encoding, g_encoding).mean
                subgoal_encoding_dis, sg_h = goal_reducer(s_encoding, g_encoding, return_h=True)
                subgoal_encoding = subgoal_encoding_dis.mean

                act2 = qnet(torch.cat((s_encoding, subgoal_encoding), dim=-1)).squeeze(0)

            action = torch.multinomial(act2, 1, replacement=False)[0].item()

            # tell if the action is correct

            if is1ststep:
                # see if it's a one-step sequence
                assert info['key'] != info['agent']
                if info['key'] == info['chest']:
                    # one-step
                    # this is good
                    entropy_is_corrects.append((entropy < entropy_threshold).float().item())
                    pass
                else:
                    # two-step
                    entropy_is_corrects.append((entropy >= entropy_threshold).float().item())
                    pass

                correct_act = env.dir2act[env.optimal_1step_acts[(info['agent'], info['key'])]]
                act1st_is_corrects.append(action == correct_act)

                # if not explore and entropy_is_corrects[-1] == False:
                #     import ipdb
                #     ipdb.set_trace()  # noqa

            else:
                # 2nd step
                entropy_is_corrects.append((entropy < entropy_threshold).float().item())
                if info['got_key'] is True:
                    correct_act = env.dir2act[env.optimal_1step_acts[(info['agent'], info['chest'])]]
                    act2nd_is_corrects.append(action == correct_act)

        next_observation, reward, terminated, truncated, info = env.step(
            action)
        reward_total += reward
        stop = terminated or truncated

        is1ststep = False

        episode.append((
            observation, action, next_observation, reward, copy.deepcopy(info), stop
        ))

        if not explore:
            episode_including_h.append(
                (observation, action, next_observation, reward, copy.deepcopy(info), stop,
                 # model variables.
                 s_encoding.squeeze(0).data.cpu().numpy(),
                 g_encoding.squeeze(0).data.cpu().numpy(),
                 act1.data.cpu().numpy(),
                 entropy.item(), action,
                 subgoal_encoding.squeeze(0).data.cpu().numpy(),
                 act2.data.cpu().numpy(),
                 # hidden layers
                 s_enc_h.squeeze(0).data.cpu().numpy(),
                 g_enc_h.squeeze(0).data.cpu().numpy(),
                 act1_h.data.cpu().numpy(),
                 sg_h.squeeze(0).data.cpu().numpy(),
                 )
            )
        observation = next_observation
        if stop:
            # collect the condition and see if it works
            if not explore and reward_total == 2:
                if len(episodes_all_conditions) > 0:
                    prev_min_length = min([len(episodes_all_conditions[k]) for k in episodes_all_conditions.keys()])
                else:
                    prev_min_length = 0
                episodes_all_conditions[info['akc']].append(copy.deepcopy(episode_including_h))
                after_min_length = min([len(episodes_all_conditions[k]) for k in episodes_all_conditions.keys()])
                if after_min_length > prev_min_length:
                    torch.save(episodes_all_conditions, res_path / 'episodes_all_conditions-2.pth')

            replay_buffer.append(copy.deepcopy(episode))
            replay_buffer = replay_buffer[:replay_buffer_size]
            reward_totals.append(reward_total)
            episode_count += 1

            episode = []
            episode_including_h = []
            reward_total = 0
            observation, info = env.reset()
            is1ststep = True

            if len(reward_totals) > exploration_eps:
                explore = False

            if len(reward_totals) % 3000 == 0:
                analyze(res_path, len(reward_totals), env, obs_encoder, goal_encoder, qnet, goal_reducer)

            # early stop
            if np.mean(reward_totals[-200:]) > stop_rew:
                print('training finished')
                break

        # training part
        # sample episodes
        if len(replay_buffer) < ep_sample_size:
            continue
        episode4learn = np.random.choice(np.asarray(replay_buffer, dtype=object), ep_sample_size, replace=False).tolist()

        obs_all = []
        goal_all = []
        act_all = []

        obs4gr = []
        goal4gr = []
        subgoal4gr = []
        act4gr = []

        for ep in episode4learn:
            for transition in ep:
                if (transition[0]['observation'] == transition[2]['observation']).all():
                    continue
                obs_all.append(transition[0]['observation'])
                goal_all.append(transition[2]['achieved_goal'])
                act_all.append(transition[1])

            if len(ep) == 2:
                transition0 = ep[0]
                if (transition0[0]['observation'] == transition0[2]['observation']).all():
                    # simulate loop removal
                    continue
                transition1 = ep[1]

                info0 = transition0[-2]
                info1 = transition1[-2]

                obs4gr.append(
                    transition0[0]['observation']
                )
                subgoal4gr.append(
                    transition0[2]['achieved_goal']
                )
                act4gr.append(
                    transition0[1]
                )
                goal4gr.append(
                    transition1[2]['achieved_goal']
                )
                # identity
                obs4gr.append(
                    transition0[0]['observation']
                )
                subgoal4gr.append(
                    transition0[2]['achieved_goal']
                )
                act4gr.append(
                    transition0[1]
                )
                goal4gr.append(
                    transition0[2]['achieved_goal']
                )

            pass
        obs_all = torch.tensor(np.array(obs_all)).to(device).float()
        goal_all = torch.tensor(np.array(goal_all)).to(device).float()
        act_all = torch.tensor(np.array(act_all)).to(device)

        # train obs and goal encoder/decoders

        obs_rep = obs_encoder(obs_all)
        obs_rec = obs_decoder(obs_rep)
        obs_loss = F.mse_loss(obs_rec, obs_all.data)
        obs_rep_opt.zero_grad()
        if obs_loss.item() > rep_loss_threshold:
            srep_training_on = 1
            obs_loss.backward()
            obs_rep_opt.step()
        else:
            srep_training_on = 0

        goal_rep = goal_encoder(goal_all)
        goal_rec = goal_decoder(goal_rep)
        goal_loss = F.mse_loss(goal_rec, goal_all.data)
        goal_rep_opt.zero_grad()
        if goal_loss.item() > rep_loss_threshold:
            grep_training_on = 1
            goal_loss.backward()
            goal_rep_opt.step()
        else:
            grep_training_on = 0

        # train local policy
        obs_goal = torch.cat([
            obs_encoder(obs_all)+0.2 * torch.randn_like(obs_rep),
            goal_encoder(goal_all)+0.2 * torch.randn_like(goal_rep)
        ], dim=1)
        act_pred = qnet(obs_goal)

        negobs_goal = torch.cat([
            obs_encoder(torch.randn_like(obs_all)),
            goal_encoder(torch.randn_like(goal_all))
        ], dim=1)
        act_pred_neg = qnet(negobs_goal)
        # entropy_loss = F.mse_loss(
        #     act_pred_neg,
        #     torch.ones_like(act_pred_neg)/4
        # )
        neg_act_entropy = (-act_pred_neg * torch.log(act_pred_neg+1e-18)).sum(dim=-1).mean()

        act_loss = F.cross_entropy(act_pred, act_all) - 0.01 * neg_act_entropy

        act_entropy = (-act_pred * torch.log(act_pred+1e-18)).sum(dim=-1).mean().item()

        act_correct = (torch.argmax(act_pred, dim=1) == act_all).float().mean()
        qnet_opt.zero_grad()
        act_loss.backward()
        qnet_opt.step()

        # train goal reducer
        if len(obs4gr) < 2:
            continue

        obs4gr = torch.tensor(np.array(obs4gr)).to(device).float()
        goal4gr = torch.tensor(np.array(goal4gr)).to(device).float()
        subgoal4gr = torch.tensor(np.array(subgoal4gr)).to(device).float()
        act4gr = torch.tensor(np.array(act4gr)).to(device)

        obs4gr_rep = obs_encoder(obs4gr)
        goal4gr_rep = goal_encoder(goal4gr)
        subgoal4gr_rep = goal_encoder(subgoal4gr).data

        subgoal_pred_rep = goal_reducer(obs4gr_rep,
                                        goal4gr_rep).rsample()

        bsz = subgoal_pred_rep.shape[0]
        subgoal_loss = F.mse_loss(subgoal_pred_rep, subgoal4gr_rep)
        acc_bsz_until += bsz
        subgoal_loss.backward()
        if acc_bsz_until > ep_sample_size:
            goal_reducer_opt.step()
            goal_reducer_opt.zero_grad()
            acc_bsz_until = 0

        # test the accuracy of two step reductions
        with torch.no_grad():
            obs_subgoal = torch.cat([
                obs4gr_rep,
                subgoal_pred_rep
            ], dim=1)

            act_subgoal_pred = qnet(obs_subgoal)
            # act_subgoal_pred = torch.softmax(v_subgoal_pred, dim=1)
            act4gr_correct = (torch.argmax(act_subgoal_pred, dim=1) == act4gr).float().mean()

        result = {
            "n/ep": episode_count,
            'rew': np.mean(reward_totals[-200:]),
            "len": len(replay_buffer[-1]),
            'explore': explore,
        }
        if env_step % 5 == 0:
            # pbar.set_postfix(result)
            logger.log_train_data(result, env_step)
            result.update({
                'rew': np.mean(reward_totals[-200:]),
                'exp': explore,
                'corr(1)': np.mean(act1st_is_corrects[-200:]),
                'corr(2)': np.mean(act2nd_is_corrects[-200:]),
                'H_corr': np.mean(entropy_is_corrects[-200:]),
                'cov': len(episodes_all_conditions) / len(env.akcs),
            }
            )
            pbar.set_postfix(result)
            logger.log_update_data(
                {'loss/obs_enc': obs_loss.item(),
                 'loss/goal_enc': goal_loss.item(),
                 'loss/local_policy': act_loss.item(),
                 'loss/goal_reducer': subgoal_loss.item(),
                 'correct/local_policy': act_correct.item(),
                 'correct/gr_policy': act4gr_correct.item(),
                 'grep_training_on': grep_training_on,
                 'srep_training_on': srep_training_on,
                 'local_entropy': act_entropy,
                 'nonlocal_entropy': neg_act_entropy.item(),
                 }, env_step)

    analyze(res_path, len(reward_totals), env, obs_encoder, goal_encoder, qnet, goal_reducer)


# for fMRI analysis


def RDMcolormapObject(direction=1):
    """
    Returns a matplotlib color map object for RSA and brain plotting
    """
    if direction == 0:
        cs = ['yellow', 'red', 'gray', 'turquoise', 'blue']
    elif direction == 1:
        cs = ['blue', 'turquoise', 'gray', 'red', 'yellow']
    else:
        raise ValueError('Direction needs to be 0 or 1')
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cs)
    return cmap
# from rsatoolbox.rdm.calc import compare


def upper_tri(RDM):
    """upper_tri returns the upper triangular index of an RDM

    Args:
        RDM 2Darray: squareform RDM

    Returns:
        1D array: upper triangular vector of the RDM
    """
    # returns the upper triangle
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]


def gen_model_RDMs(rdm_info_f: Path):
    rdm_info = torch.load(rdm_info_f)

    rdm_info['component_RDMs'] = rdm_info['component_RDMs'].data.cpu().numpy()
    new_betanames = []
    new_idx = []
    for bidx, betaname in enumerate(rdm_info['noahzarr_betanames']):
        if 'fb' in betaname:
            continue
        new_betanames.append(betaname)
        new_idx.append(bidx)

    rdm_info['component_RDMs_new'] = []
    for rdm_idx in range(rdm_info['component_RDMs'].shape[0]):
        rdm_info['component_RDMs_new'].append(
            rdm_info['component_RDMs'][rdm_idx][new_idx][:, new_idx].copy())

    # sides = np.ceil(np.sqrt(rdm_info['component_RDMs'].shape[0])).astype(int)
    # fig, axes = plt.subplots(sides, sides, figsize=[sides*1.5, sides*1.5])
    # for ax_id, ax in enumerate(axes.flatten()):
    #     ax.axis('off')
    #     if ax_id < rdm_info['component_RDMs'].shape[0]:
    #         ax.imshow(rdm_info['component_RDMs_new'][ax_id])
    #         ax.set_title(rdm_info['all_components'][ax_id])
    # fig.tight_layout()
    # plt.show()

    # arrays_to_save = {'array1': np.array([1, 2, 3]), 'array2': np.array([4, 5, 6])}
    rdm_dict = {}
    for idx, component in enumerate(rdm_info['all_components']):
        cn = component.replace('.', '_')
        rdm_dict[cn] = rdm_info['component_RDMs_new'][idx].copy()
        # print(cn)

    rdm2matlab = {
        # 'XpatternData': None,
        'rdm': rdm_dict,
        'rdmInfo': {
            'betaNames': np.array(new_betanames, dtype=object)
        },
    }
    return rdm2matlab


def load_subject_data(data_dir: Path, subject: int, new_betanames: list,
                      radius=5, threshold=0.5):
    subject_data_dir = data_dir / str(subject) / 'RESULTS/RSA2_nobc'
    subject_data_dir.is_dir()

    subject_spm = loadmat(subject_data_dir / 'SPM.mat')['SPM']['xX']
    all_beta_names = [x[0] for x in subject_spm.item()['name'].item()[0]]

    # used_beta_id = [for beta in all_beta_names if ]
    used_beta_ids = dict()
    for beta_name in new_betanames:
        for idx, full_beta_name in enumerate(all_beta_names):
            if beta_name in full_beta_name:
                used_beta_ids[beta_name] = idx
                break
    image_paths = []
    beta_list = []
    for beta_name, beta_id in used_beta_ids.items():
        beta_list.append(beta_name)
        bf = subject_data_dir / f'beta_0{str(beta_id).zfill(3)}.hdr'
        assert bf.is_file(), f'{beta_name} has no file {bf}'
        image_paths.append(str(bf.resolve()))

    # load one image to get the dimensions and make the mask
    tmp_img = nib.load(subject_data_dir / 'mask.hdr')
    # we infer the mask by looking at non-nan voxels
    mask = ~np.isnan(tmp_img.get_fdata())
    x, y, z = tmp_img.get_fdata().shape

    # loop over all images
    data = np.zeros((len(image_paths), x, y, z))
    for x, im in enumerate(image_paths):
        data[x] = nib.load(im).get_fdata()

    # only one pattern per image
    image_value = np.arange(len(image_paths))

    centers, neighbors = get_volume_searchlight(
        mask, radius=radius, threshold=threshold)
    # reshape data so we have n_observastions x n_voxels
    data_2d = data.reshape([data.shape[0], -1])
    data_2d = np.nan_to_num(data_2d)
    # Get RDMs
    SL_RDM = get_searchlight_RDMs(
        data_2d, centers, neighbors, image_value, method='correlation')
    return SL_RDM


def subject_RDM_worker(args):
    res_dir, tdata_dir, subject, betanames, radius, threshold = args
    subject_rdm = load_subject_data(
        tdata_dir, subject, betanames,
        radius=radius, threshold=threshold)
    torch.save(subject_rdm, res_dir / f'subj_{subject}_rdm.pth')
    print(f'{subject} RDM is finished')
    return subject_rdm


def parallel_subject_RDMs(subject_args, jobs):
    print('creating pool...')
    pool = mp.ProcessingPool(nodes=jobs)
    print('pool created')
    pool.map(subject_RDM_worker, subject_args)
    pool.close()
    pool.join()


def read_subj_rdms(subjects, res_dir):
    remaining_subjects = []
    subjects_RDMs = []
    for subj in subjects:
        subj_rdm_f = res_dir / f'subj_{subj}_rdm.pth'
        if subj_rdm_f.is_file():
            print(f'load {subj_rdm_f}')
            subjects_RDMs.append(torch.load(subj_rdm_f))
        else:
            remaining_subjects.append(subj)
    return subjects_RDMs, remaining_subjects


@cli.command
@click.pass_context
@click.option(
    '--data-dir',
    default='/data/hammer/space2/mvpaGoals/data/fmri/', help="fMRI data dir")
@click.option(
    '--model-rdmf',
    default='results.refactor/GOLSAv2-TreasureHunt-Single/GOLSAv2/1926/RDM_info-epoch-17239.pt',
    help="fMRI data dir")
def fmrianalyze(ctx, data_dir, model_rdmf):
    project_name = 'GOLSAv2-TreasureHunt-Single'
    proj_dir = Path(f"results.{get_git_branch()}/{project_name}")
    assert proj_dir.is_dir()
    res_dir = proj_dir / 'fMRI'
    res_dir.mkdir(exist_ok=True)

    model_RDMs = gen_model_RDMs(Path(model_rdmf))
    all_beta_names = model_RDMs['rdmInfo']['betaNames']

    data_dir = Path(data_dir)
    subjects = [201, 210, 212, 213, 220, 221, 227, ]
    # [228, 229, 230, 231, 233, 234, 235, 236, 238, 239, 240, 241, 242, 244, 245, 246, 247]

    # get mask
    tmpf = data_dir / str(subjects[0]) / 'RESULTS/RSA2_nobc' / 'beta_0211.img'
    tmp_img = (nib.load(tmpf))
    mask = ~np.isnan(tmp_img.get_fdata())
    subjects_RDMs, remaining_subjects = read_subj_rdms(subjects, res_dir)

    radius = 10
    threshold = 1.0
    subject_args = [
        (res_dir, data_dir, subject, all_beta_names, radius, threshold) for subject in remaining_subjects
    ]
    # subjects_RDMs_file = res_dir / 'subjects_RDMs.pth'
    n_jobs = 6

    if len(subject_args) > 0:
        parallel_subject_RDMs(subject_args, n_jobs)
        remaining_subjects_RDMs, _ = read_subj_rdms(remaining_subjects, res_dir)
        subjects_RDMs.extend(remaining_subjects_RDMs)

    # smoothing_fwhm = 8.0
    # p_threshold = 0.05
    # height_control = 'fdr'

    # rdm_dict = model_RDMs['rdm']
    # component = 'gr_out'

    # golsa_model = ModelFixed(f'{component} RDM', upper_tri(rdm_dict[component]))

    # subject_z_imgs = []

    # for subj_RDM in subjects_RDMs:
    #     eval_results = evaluate_models_searchlight(
    #         subj_RDM, golsa_model, eval_fixed, method='spearman', n_jobs=n_jobs)
    #     eval_score = np.array([float(e.evaluations) for e in eval_results])
    #     eval_score_z = np.arctanh(eval_score)
    #     eval_score_z[np.isinf(eval_score_z)] = 0
    #     x, y, z = mask.shape
    #     RDM_brain = np.zeros([x*y*z])
    #     RDM_brain[list(subj_RDM.rdm_descriptors['voxel_index'])] = eval_score_z
    #     RDM_brain = RDM_brain.reshape([x, y, z])
    #     subject_z_img = new_img_like(tmp_img, RDM_brain)
    #     subject_z_imgs.append(subject_z_img)

    # subject_z_imgs = concat_imgs(subject_z_imgs)
    # mean_z_img = mean_img(subject_z_imgs)
    # design_matrix = pd.DataFrame([1] * len(subjects_RDMs), columns=['intercept'])
    # second_level_model = SecondLevelModel(smoothing_fwhm=smoothing_fwhm)
    # second_level_model = second_level_model.fit(subject_z_imgs, design_matrix=design_matrix)
    # z_map = second_level_model.compute_contrast(output_type='z_score')
    # # now perform group level analysis

    # import ipdb
    # ipdb.set_trace()  # noqa

    pass
    # if not subjects_RDMs_file.is_file():
    #     parallel_subject_RDMs(subject_args, n_jobs)
    # else:
    #     subjects_RDMs = torch.load(subjects_RDMs_file)

    # now for a single component RDM,
    # we need to compare it with voxel RDMs in each subject.


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
