import torch
import numpy as np

from policy_wo_RL import GOLSAv2woRL
from tianshou.data import Batch, ReplayBuffer
from golsa_base import EpMemSamplingStrategy
from pathlib import Path
from utils import get_RDM
import matplotlib.pyplot as plt
from typing import Any, Dict
import torch.nn.functional as F


def remove_trailing(a):
    va = a[::-1]
    seen = set()
    seen.add(va[0])
    for i, v in enumerate(va[1:]):
        if v in seen:
            continue
        else:
            return a[:len(a)-i]
    return [va[0]]


def gr_distance_ratio(gr, model, env, s_embs, g_embs, locs, g_infos, device):
    with torch.no_grad():
        a_ids = []
        subgoal_ids = []
        goal_ids = []
        for akc in env.akcs:
            # kc_id = g_infos[akc[1:]]
            a_id = locs[akc[0]]
            if akc[1] != akc[2]:
                # distant goals
                a_ids.append(a_id)
                subgoal_ids.append(g_infos[(akc[1], akc[1])])
                goal_ids.append(g_infos[akc[1:]])
        # reduce goals
        s_reps = model.encoding_layers(s_embs)
        g_reps = model.g_encoding_layers(g_embs)

        s_combined_2step = s_reps[a_ids]
        g_combined_2step = g_reps[goal_ids]
        subgoal_combined = g_reps[subgoal_ids]
        subgoals = gr.gen_sg(s_combined_2step, g_combined_2step)  # rep level.

        # mean_goal_distance = torch.cdist(g_reps, g_reps).mean()
        # mean_goal_distance = torch.cdist(g_reps, subgoals).mean()
        mean_goal_distance = []
        for sg in subgoals:
            mean_goal_distance.append(torch.cdist(sg.unsqueeze(0), g_reps))
        mean_goal_distance = torch.cat(mean_goal_distance).mean()

        # predicted_goal_distance = torch.cdist(subgoal_combined, subgoals).mean()
        predicted_goal_distance = ((subgoal_combined - subgoals)**2).sum(dim=1).sqrt().mean()
        subgoal_distance_ratio = (predicted_goal_distance/mean_goal_distance).item()
        return subgoal_distance_ratio


def local_act_correct_ratio(model, env, s_embs, g_embs, locs, g_infos, device):
    with torch.no_grad():
        kc_ids = []
        a_ids = []
        for akc in env.akcs:
            kc_id = g_infos[akc[1:]]
            a_id = locs[akc[0]]
            kc_ids.append(kc_id)
            a_ids.append(a_id)

        s_reps = model.encoding_layers(s_embs)
        g_reps = model.g_encoding_layers(g_embs)
        # g_combined = g_reps[kc_ids]
        # s_combined = s_reps[a_ids]

        # subgoals = self.goal_reducer.gen_sg(s_combined, g_combined)  # rep level.

        # test one: can the agent find best option for one-step cases?
        # extract all one-step s and g. test the generaed
        # ONE-STEP sequences
        agent_locs = []
        one_step_goals = []
        correct_acts = []
        for agent_loc in env.loc_neighbors.keys():
            for neighbor_loc in env.loc_neighbors[agent_loc]:
                agent_locs.append(locs[agent_loc])
                one_step_goals.append(g_infos[(neighbor_loc, neighbor_loc)])
                correct_acts.append(

                    env.dir2act[env.optimal_1step_acts[(agent_loc, neighbor_loc)]]

                )
        correct_acts = torch.tensor(correct_acts).to(device)

        s_combined_1step = s_reps[agent_locs]
        g_combined_1step = g_reps[one_step_goals]
        act_dist_1step = model.qnet(torch.cat((s_combined_1step, g_combined_1step), dim=-1))
        act_1step = torch.argmax(act_dist_1step, dim=-1)
        correct_ratio_1step = (act_1step == correct_acts).float().mean().item()

    return correct_ratio_1step


class GOLSAv2NonRL4TH(GOLSAv2woRL):
    """
    NoRL for treasure hunting
    """
    _action_to_direction = {
        0: "A",
        1: "W",
        2: "D",
        3: "S",
    }
    dir2act = {v: k for k, v in _action_to_direction.items()}
    optimal_1step_acts = {
        # s, g -> a
        (0, 1): 'D',
        (0, 3): 'S',

        (1, 0): 'A',
        (1, 2): 'S',

        (2, 3): 'A',
        (2, 1): 'W',

        (3, 0): 'W',
        (3, 2): 'D',
    }

    def convert2tensor(self, x: np.ndarray):
        x_tensor = torch.tensor(x, dtype=torch.float).to(self.device)
        return x_tensor

    def sample_subgoals_frm_trjs(self, buffer, batch, max_w=100):
        """Freely adjustable subgoal learning."""
        s_buf_ids = []
        sg_buf_ids = []
        g_buf_ids = []
        trj_ws = []
        row_idxes = []
        trj_length_total = []
        clean_trjs = []
        for row_idx, row in enumerate(batch.trajectories):
            clean_trj = np.array(row)[~np.isnan(row)].astype(int).tolist()
            clean_trj = remove_trailing(clean_trj)
            trj_length = len(clean_trj)
            if trj_length < 2:
                continue
            elif trj_length == 2:
                # print(f'row_idx={row_idx}')
                s_buf_id = clean_trj[0]
                sg_buf_id = clean_trj[0]
                g_buf_id = clean_trj[1]
                trj_w = 1.0  # / max_w
            else:
                # print(f'>2 row_idx={row_idx} {trj_length}')
                # s_idid, sg_idid, g_idid = np.sort(np.random.choice(trj_length, 3, replace=False))
                s_idid = np.random.choice(trj_length-1, 1)[0]
                g_idid = np.random.choice(list(range(s_idid+1, trj_length)), 1)[0]
                if g_idid == s_idid + 1:
                    sg_idid = s_idid
                else:
                    sg_idid = np.random.choice(list(range(s_idid, g_idid-1)), 1)[0]

                s_buf_id = clean_trj[s_idid]
                sg_buf_id = clean_trj[sg_idid]
                g_buf_id = clean_trj[g_idid]
                trj_w = (g_idid - s_idid + 1.0)  # / max_w
            clean_trjs.append((row_idx, clean_trj))

            s_buf_ids.append(s_buf_id)
            sg_buf_ids.append(sg_buf_id)
            g_buf_ids.append(g_buf_id)
            trj_ws.append(trj_w)
            row_idxes.append(row_idx)
            trj_length_total.append(trj_length)

        if len(s_buf_ids) < 1:
            return None

        s_buf_ids = np.array(s_buf_ids)
        sg_buf_ids = np.array(sg_buf_ids)
        g_buf_ids = np.array(g_buf_ids)
        trj_ws = np.array(trj_ws)
        trj_length_total = np.array(trj_length_total)

        trj_info = {
            'trj_length_total_mean': trj_length_total.mean(),
            'trj_length_total_max': trj_length_total.max()
        }
        self.extra_learning_info.update(trj_info)

        trj_ws = trj_ws / trj_ws.sum()
        trj_ws = torch.from_numpy(np.array(trj_ws)).float().to(self.device)

        # eff_batch
        model = getattr(self, 'model')
        with torch.no_grad():
            subgoal_encoding = model.encode_g(
                self.extract_batch_goal(
                    buffer[sg_buf_ids].obs.achieved_goal
                )
            )

        # since we want to adjust the encoding layers too, here we need grads.
        state_encoding = model.encode_s(buffer[s_buf_ids].obs.observation)
        # final_reached_goal_encoding = model.g_encoding_layers(final_reached_goal_img_inputs)
        final_reached_goal_encoding = model.encode_g(
            self.extract_batch_goal(buffer[g_buf_ids].obs.achieved_goal)
        )

        batch = batch[row_idxes]
        batch.state_encoding = state_encoding
        batch.final_reached_goal_encoding = final_reached_goal_encoding
        batch.subgoal_encoding = subgoal_encoding.data
        batch.trj_ws = trj_ws
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

        hasrew = buffer[indicess[-1]].rew >= 2.0

        indicess = indicess.astype(float)
        for trj_idx in range(indicess.shape[1]):
            if hasrew[trj_idx] == 0:
                indicess[1:, trj_idx] = np.nan
        batch.trajectories = indicess.T
        return batch

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Process that batched data so that it can be used for Q learning.
        - Compute the n-step return for Q-learning targets.
        - Sample and train different subgoals from the replay buffer and train the goal reducer.
        """

        sg_loss = None
        act_loss = None
        batch = self.sample_trjs_from_replay_buffer(batch, buffer, indices)

        for _ in range(3):
            res = self.sample_subgoals_frm_trjs(buffer, batch)
            if res is None:
                continue
            batch = res
            sg_loss = self._optimize_gr(
                batch.state_encoding,
                batch.final_reached_goal_encoding,
                batch.subgoal_encoding,
                trj_ws=batch.trj_ws
            )

            # if sg_loss.item() >= 0.1:
            #     self.turn_on_subgoal_helper = 1.0

        # batch.negative_goals = buffer.sample(len(batch))[0].obs
        wm_loss = self.learn_worldmodel(batch, s_noise_level=0.1)

        self.extra_learning_info.update({
            "wm_l": wm_loss.item(),
        })
        if sg_loss is not None:
            self.extra_learning_info["sg_l"] = sg_loss.item()

        # if act_loss is not None:
        #     self.extra_learning_info["loss"] = act_loss.item()

        # test the correct ratio when the goal is just one step away.
        self.extra_learning_info['local_correct'] = local_act_correct_ratio(
            self.model,
            self.task_info['env'],
            self.task_info['s_embs'],
            self.task_info['g_embs'],
            self.task_info['locs'],
            self.task_info['g_infos'],
            self.device)
        # test if the model can successfully reduce all two-step goals into one step goals.
        self.extra_learning_info['sg_dis_ratio'] = gr_distance_ratio(
            self.goal_reducer, self.model,
            self.task_info['env'],
            self.task_info['s_embs'],
            self.task_info['g_embs'],
            self.task_info['locs'],
            self.task_info['g_infos'],
            self.device)

        return batch

    def analyze(self,
                env,
                ep_passed: int):
        """
        Things todo here:
        1. Compute RDM
        """
        self.eval()
        correct_ratio_1step = local_act_correct_ratio(
            self.model,
            self.task_info['env'],
            self.task_info['s_embs'],
            self.task_info['g_embs'],
            self.task_info['locs'],
            self.task_info['g_infos'],
            self.device)
        print(f'1step correct ratio: {correct_ratio_1step}')
        self.train()

    def analyze_(self,
                 env,
                 ep_passed: int):
        """
        Things todo here:
        1. Compute RDM
        """
        self.eval()
        with torch.no_grad():
            correct_ratio_1step = local_act_correct_ratio(
                self.model,
                self.task_info['env'],
                self.task_info['s_embs'],
                self.task_info['g_embs'],
                self.task_info['locs'],
                self.task_info['g_infos'],
                self.device)
            print(f'1step correct ratio: {correct_ratio_1step}')

            # s_reps = self.model.encoding_layers(s_embs)
            # g_reps = self.model.g_encoding_layers(g_embs)

            # kc_ids = []
            # a_ids = []
            # for akc in env.akcs:
            #     kc_id = g_infos[akc[1:]]
            #     a_id = locs[akc[0]]
            #     kc_ids.append(kc_id)
            #     a_ids.append(a_id)

            # g_combined = g_reps[kc_ids]
            # s_combined = s_reps[a_ids]

            # self.goal_reducer
            # subgoals = self.goal_reducer.gen_sg(s_combined, g_combined)  # rep level.

            # # test one: can the agent find best option for one-step cases?
            # # extract all one-step s and g. test the generaed
            # # ONE-STEP sequences
            # agent_locs = []
            # one_step_goals = []
            # correct_acts = []
            # for agent_loc in env.loc_neighbors.keys():
            #     for neighbor_loc in env.loc_neighbors[agent_loc]:
            #         agent_locs.append(locs[agent_loc])
            #         one_step_goals.append(g_infos[(neighbor_loc, neighbor_loc)])
            #         correct_acts.append(

            #             env.dir2act[env.optimal_1step_acts[(agent_loc, neighbor_loc)]]

            #         )
            # correct_acts = torch.tensor(correct_acts).to(self.device)

            # s_combined_1step = s_reps[agent_locs]
            # g_combined_1step = g_reps[one_step_goals]
            # act_dist_1step = self.model.qnet(torch.cat((s_combined_1step, g_combined_1step), dim=-1))
            # act_1step = torch.argmax(act_dist_1step, dim=-1)
            # correct_ratio_1step = (act_1step == correct_acts).float().mean().item()
            # print(f'1step correct ratio: {correct_ratio_1step}')

            # now let's focus on the goal reduction space.
            # In theory, we can calculate everything here used for RDM.
            # notice now G and S are not the same.
            # here we will visualize 8 (4x2) diagonal cases.
            diagonal_akcs = [
                (0, 1, 2),
                (0, 3, 2),

                (1, 2, 3),
                (1, 0, 3),

                (2, 3, 0),
                (2, 1, 0),

                (3, 0, 1),
                (3, 2, 1),
            ]
            # diagonal_akc_ids = [env.akcs.index(dakc) for dakc in diagonal_akcs]
            # diagonal_subgoal_reps = subgoals[diagonal_akc_ids]
            corner_goals = [
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 3),
            ]

            # corner_goal_ids = [g_infos[cg] for cg in corner_goals]
            # corner_g_reps = g_reps[corner_goal_ids]

            # now we need to calculate RDMs
            # components-> (condxcond)
            # env.akcs: 24 starting points.
            # also we need 8 midpoints
            startpoint_trials = env.akcs
            midpoint_trials = []
            for loc in env.loc_neighbors.keys():
                for neighbor_loc in env.loc_neighbors[loc]:
                    midpoint_trials.append((loc, loc, neighbor_loc))
            all_trials = startpoint_trials + midpoint_trials

            all_s_ids = []
            all_kc_ids = []
            trial_names = []
            # the format of all_trials is (agent, key, chest)
            # agent: the current location of the agent
            # key, chest: the configuration of the goal state.
            n_trials = len(all_trials)
            for trial in all_trials:
                all_s_ids.append(locs[trial[0]])
                all_kc_ids.append(g_infos[trial[1:]])

                start_state = env.loc_name_mapping[trial[0]].lower()

                if trial[0] == trial[1]:
                    # midpoint of a two-step path.
                    next_state = env.loc_name_mapping[trial[2]]
                    final_state = 'None'
                else:
                    next_state = env.loc_name_mapping[trial[1]]
                    final_state = env.loc_name_mapping[trial[2]]
                trial_names.append(f'{start_state}{next_state}{final_state}')
                # trial_names

            all_g_embs = g_embs[all_kc_ids]  # s embeddings
            all_s_embs = s_embs[all_s_ids]  # g embeddings

            # now let's calculate all inner variables.
            # the process is divided into two phases: planning and action.
            # for the planning part, both the goal reducer and the local policy work.

            s_combined = self.model.encoding_layers(all_s_embs)  # s representations
            g_combined = self.model.g_encoding_layers(all_g_embs)  # g representations
            sg_combined = self.goal_reducer.gen_sg(s_combined, g_combined)  # subgoal representations.
            act_g = self.model.qnet(torch.cat((s_combined, g_combined), dim=-1))
            act_sg = self.model.qnet(torch.cat((s_combined, sg_combined), dim=-1))
            act_all = torch.cat((act_g, act_sg), dim=-1)
            # components: g_encoder, s_encoder, goal_reducer, policy.

            s_RDM = get_RDM(s_combined, s_combined)
            g_RDM = get_RDM(g_combined, g_combined)
            gr_RDM = get_RDM(sg_combined, sg_combined)
            act_RDM = get_RDM(act_all, act_all)

            total_RDM = torch.zeros(n_trials*3, n_trials*3)
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

            total_s_RDM = total_RDM.clone()
            total_s_RDM[:n_trials, :n_trials] = s_RDM
            # re-order
            total_s_RDM = total_s_RDM[beta_ids][:, beta_ids]

            total_g_RDM = total_RDM.clone()
            total_g_RDM[:n_trials, :n_trials] = g_RDM
            # re-order
            total_g_RDM = total_g_RDM[beta_ids][:, beta_ids]

            total_gr_RDM = total_RDM.clone()
            total_gr_RDM[:n_trials, :n_trials] = gr_RDM
            # re-order
            total_gr_RDM = total_gr_RDM[beta_ids][:, beta_ids]

            total_act_RDM = total_RDM.clone()
            total_act_RDM[n_trials:n_trials*2, n_trials:n_trials*2] = act_RDM
            # re-order
            total_act_RDM = total_act_RDM[beta_ids][:, beta_ids]

            rdm_info = {
                'state': total_s_RDM.data.cpu().numpy(),
                'goal': total_g_RDM.data.cpu().numpy(),
                'subgoal': total_gr_RDM.data.cpu().numpy(),
                'action': total_act_RDM.data.cpu().numpy(),
            }
            torch.save(rdm_info, self.log_path/f'RDM_info-epoch-{ep_passed}.pt')

            fig, axes = plt.subplots(1, 4, sharex=True, sharey=True,
                                     figsize=[10, 3])
            for ax in axes:
                ax.axis('off')
            cmap = 'plasma'
            axes[0].imshow(rdm_info['state'], cmap=cmap)
            axes[0].set_title('State')

            axes[1].imshow(rdm_info['goal'], cmap=cmap)
            axes[1].set_title('Goal')

            axes[2].imshow(rdm_info['subgoal'], cmap=cmap)
            axes[2].set_title('Subgoal')

            axes[3].imshow(rdm_info['action'], cmap=cmap)
            axes[3].set_title('Action')

            fig.tight_layout()
            fig.suptitle(f'RDMs (epoch={ep_passed})')
            fig.savefig(self.fig_dir / f'RDM-epoch-{ep_passed}.png')

        self.train()
