"""GOLSAv2 built with unsupervised world model and GR learning.
"""
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from golsa_base import EpMemSamplingStrategy, PolicyBase, max_st_noise_scales
from policy_utils import policy_entropy
from tianshou.data import Batch, ReplayBuffer


class GOLSAv2woRL(PolicyBase):
    """
    policy without RL for gridworld task.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        goal_reducer: torch.nn.Module,
        goal_reducer_optim: torch.optim.Optimizer,
        world_model: torch.nn.Module,
        world_model_optim: torch.optim.Optimizer,
        extract_batch_goal: Callable = lambda x: x,
        sampling_strategy: int = 2,
        gr_steps: int = 3,
        max_steps: int = 100,
        discount_factor: float = 0.99,
        d_kl_c: float = 1.0,
        estimation_step: int = 1,
        reward_normalization: bool = False,
        is_double: bool = False,
        clip_loss_grad: bool = False,
        log_path: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        neg_act_kl_w: float = 1.0,
        n_ep_explore: int = 3,
        **kwargs: Any,
    ) -> None:
        """_summary_

        Notice in this version the subgoal is alway on.

        Args:
            model (torch.nn.Module): _description_
            optim (torch.optim.Optimizer): _description_
            goal_reducer (torch.nn.Module): _description_
            goal_reducer_optim (torch.optim.Optimizer): _description_
            world_model (torch.nn.Module): _description_
            world_model_optim (torch.optim.Optimizer): _description_
            sampling_strategy (int, optional): _description_. Defaults to 2.
            gr_steps (int, optional): number of max goal reduction steps, defaults to 3.
            max_steps (int, optional): _description_. Defaults to 100.
            discount_factor (float, optional): _description_. Defaults to 0.99.
            d_kl_c (float, optional): _description_. Defaults to 1.0.
            estimation_step (int, optional): _description_. Defaults to 1.
            reward_normalization (bool, optional): _description_. Defaults to False.
            is_double (bool, optional): _description_. Defaults to False.
            clip_loss_grad (bool, optional): _description_. Defaults to False.
            log_path (Optional[str], optional): _description_. Defaults to None.
            device (Union[str, torch.device], optional): _description_. Defaults to "cpu".
        """
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.goal_reducer = goal_reducer
        self.goal_reducer_optim = goal_reducer_optim

        self.world_model = world_model
        self.world_model_optim = world_model_optim

        self.extract_batch_goal = extract_batch_goal  # used to extract goal inputs from the batch.

        self.subgoal_planning = True
        self.sampling_strategy = EpMemSamplingStrategy(sampling_strategy)
        self.max_steps = max_steps
        self.device = device

        self.max_action_num = self.model.a_dim
        print('using sampling strategy: ', self.sampling_strategy)

        self.better_subgoal = False

        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self.d_kl_c = d_kl_c
        self._n_step = estimation_step
        self._iter = 0

        self._rew_norm = reward_normalization
        self._is_double = is_double
        self._clip_loss_grad = clip_loss_grad

        self.log_path = log_path

        self._setup_log()

        self.turn_on_subgoal_helper = .0
        self.extra_learning_info = {}

        # policy related settings
        self.gr_steps = gr_steps
        self.gr_n_tried = 12  # number of subgoals one may try at most at a time
        self.act_entropy_threshold = 0.8  # max entropy of action distribution in nats.

        self._setup_queue()

        # set up a dict for debugging to test the coverage of subgoals
        self.subgoal_dataset_coverage = defaultdict(Counter)

        self.random_exploration = True

        self.neg_act_kl_w = neg_act_kl_w
        self.n_ep_explore = n_ep_explore

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        gen_individually: bool = False,
        **kwargs: Any,
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs

        qvs, hidden, (state_encoding, goal_encoding) = model(obs_next, state=state, info=batch.info, gen_individually=gen_individually)
        act_dists = torch.softmax(qvs, dim=-1)
        final_act_dists = torch.ones_like(act_dists) / act_dists.shape[-1]

        H_pi = policy_entropy(act_dists)

        solution_found = H_pi < self.act_entropy_threshold
        if solution_found.any():
            sol_ids = torch.where(solution_found)[0]
            final_act_dists[sol_ids] = act_dists[sol_ids]

        repeated_state_encoding = state_encoding
        repeated_subgoal_encoding = goal_encoding

        for i in range(self.gr_steps):
            same_dim = [1 for _ in range(repeated_state_encoding.ndim)]

            repeated_state_encoding = repeated_state_encoding.unsqueeze(
                0).repeat(self.gr_n_tried, *same_dim)
            repeated_subgoal_encoding = repeated_subgoal_encoding.unsqueeze(
                0).repeat(self.gr_n_tried, *same_dim)

            repeated_subgoal_encoding = self.goal_reducer.gen_sg(
                repeated_state_encoding, repeated_subgoal_encoding)

            # run Q net
            repeated_qvs = model.qnet(torch.cat(
                (repeated_state_encoding,
                 repeated_subgoal_encoding),
                dim=-1))
            repeated_act_dists = torch.softmax(repeated_qvs, dim=-1)

            # for decision
            flatten_repeated_act_dists = repeated_act_dists.view(
                torch.tensor(repeated_act_dists.shape[:-2]).prod(),
                *repeated_act_dists.shape[-2:])  # (K, bsz, act_dim)

            H_repeated_act_dists = policy_entropy(flatten_repeated_act_dists)
            H_blow = H_repeated_act_dists < self.act_entropy_threshold

            tmp_solution_found = H_blow.any(dim=0)
            # we only update action distributions for the ones that are
            # not solved yet.
            new_solution_found = tmp_solution_found & ~solution_found
            if new_solution_found.any():
                sols_found = torch.where(new_solution_found)[0]

                for sc in sols_found:
                    srow = torch.where(H_blow[:, sc] == torch.tensor(True))[0]
                    # there're two ways to do it:
                    # 1. randomly select one (here we select the 1st but it
                    # does not matter.)
                    # final_act_dists[sc] = flatten_repeated_act_dists[srow[0], sc]
                    # 2. take the average of all plausible action distributions.
                    final_act_dists[sc] = flatten_repeated_act_dists[
                        srow, sc].mean(dim=0)

                # update solutiuon
                solution_found = solution_found | new_solution_found

        logits = final_act_dists
        # For solution_found case, we get the argmax
        # otherwiese we use random selection
        act_ids = torch.max(final_act_dists, dim=-1).indices
        rand_act_ids = torch.randint_like(act_ids, 0, logits.shape[-1])
        act = torch.where(solution_found, act_ids, rand_act_ids).data.cpu().numpy()

        if self.random_exploration:
            act = np.random.choice(self.max_action_num, len(act))

        return Batch(logits=logits, act=act, state=hidden)

    def _optimize_gr(self, state_encoding: torch.Tensor,
                     goal_encoding: torch.Tensor,
                     target_subgoal_encoding: torch.Tensor,
                     add_noise: bool = True,
                     trj_ws: Optional[torch.Tensor] = None):
        """Optimize goal-reducer.
        The learning of goal-reducer should be independent of the learning of the model.


        Args:
            state_encoding (torch.Tensor): Current state representation.
            goal_encoding (torch.Tensor): Goal representation.
            target_subgoal_encoding (torch.Tensor): Sampled subgoal representation.

        Returns:
            torch.Tensor: Subgoal loss.
        """
        state_dim = state_encoding.shape[-1]
        if add_noise:
            state_encoding = state_encoding + (
                torch.rand_like(state_encoding) - 0.5) * max_st_noise_scales[
                state_dim]

            goal_encoding = goal_encoding + (
                torch.rand_like(goal_encoding) - 0.5) * max_st_noise_scales[
                state_dim]

        if self.goal_reducer.gr == 'VAE':
            res = self.goal_reducer(
                state_encoding,
                goal_encoding
            )
            subgoal_encoding, mean_z, log_var_z = res
        elif self.goal_reducer.gr == 'Laplacian':
            subgoal_dist = self.goal_reducer(
                state_encoding,
                goal_encoding
            )
            subgoal_encoding = subgoal_dist.loc

        sz = subgoal_encoding.shape[0]
        if trj_ws is None:
            trj_ws = torch.ones_like(sz) / sz

        if add_noise:
            target_subgoal_encoding = target_subgoal_encoding + (
                torch.rand_like(subgoal_encoding) - 0.5
            ) * max_st_noise_scales[
                state_dim]

        if self.goal_reducer.gr == 'VAE':
            subgoal_loss = self.goal_reducer.loss_function(
                target_subgoal_encoding, subgoal_encoding, mean_z,
                log_var_z, self.goal_reducer.KL_weight, x_weights=trj_ws)
        elif self.goal_reducer.gr == 'Laplacian':
            log_prob = subgoal_dist.log_prob(target_subgoal_encoding).sum(dim=-1)
            # we need to increase the probability of the subgoals that have higher advantages.

            subgoal_loss = -(log_prob * trj_ws).mean() * 1.0
            subgoal_loss = torch.clamp(subgoal_loss, max=10.0)

        self.goal_reducer_optim.zero_grad()
        subgoal_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.goal_reducer.parameters(),
                                       max_norm=4.0,
                                       norm_type=2)
        self.goal_reducer_optim.step()
        return subgoal_loss

    def sample_subgoals_frm_trjs(self, buffer, batch, max_w=100):
        """Freely adjustable subgoal learning."""
        s_buf_ids = []
        sg_buf_ids = []
        g_buf_ids = []
        trj_ws = []
        row_idxes = []
        for row_idx, row in enumerate(batch.trajectories):
            clean_trj = np.array(row)[~np.isnan(row)].astype(int).tolist()
            trj_length = len(clean_trj)
            if trj_length < 2:
                continue
            elif trj_length == 2:
                s_buf_id = clean_trj[0]
                sg_buf_id = clean_trj[1]
                g_buf_id = clean_trj[1]
                trj_w = 1.0  # / max_w
            else:
                s_idid, sg_idid, g_idid = np.sort(np.random.choice(trj_length, 3, replace=False))
                s_buf_id = clean_trj[s_idid]
                sg_buf_id = clean_trj[sg_idid]
                g_buf_id = clean_trj[g_idid]
                trj_w = (g_idid - s_idid + 1.0)  # / max_w

            s_buf_ids.append(s_buf_id)
            sg_buf_ids.append(sg_buf_id)
            g_buf_ids.append(g_buf_id)
            trj_ws.append(trj_w)
            row_idxes.append(row_idx)

        if len(s_buf_ids) < 1:
            return None

        s_buf_ids = np.array(s_buf_ids)
        sg_buf_ids = np.array(sg_buf_ids)
        g_buf_ids = np.array(g_buf_ids)
        trj_ws = np.array(trj_ws)

        trj_info = {
            'trj_len_mean': trj_ws.mean(),
            'trj_len_min': trj_ws.min(),
            'trj_len_max': trj_ws.max(),
        }
        self.extra_learning_info.update(trj_info)

        trj_ws = trj_ws / trj_ws.sum()
        trj_ws = torch.from_numpy(np.array(trj_ws)).float().to(self.device)

        # eff_batch
        model = getattr(self, 'model')
        with torch.no_grad():
            # here we will also collect the coverage ratio
            buffer[s_buf_ids].obs
            s_pos = buffer[s_buf_ids].info['prev_agent_pos']
            g_pos = buffer[g_buf_ids].info['prev_agent_pos']
            sg_pos = buffer[sg_buf_ids].info['prev_agent_pos']
            # s, g, sg
            # s_g_pos = [(tuple(ts_pos), tuple(tg_pos), tuple(tsg_pos)) for ts_pos, tg_pos, tsg_pos in zip(s_pos, g_pos, sg_pos)]
            for ts_pos, tg_pos, tsg_pos in zip(s_pos, g_pos, sg_pos):
                self.subgoal_dataset_coverage[
                    (tuple(ts_pos), tuple(tg_pos))].update(tuple(tsg_pos))

                pass

            assert (self.extract_batch_goal(buffer[s_buf_ids].obs.achieved_goal) == self.extract_batch_goal(buffer[s_buf_ids].obs)).all()
            assert (self.extract_batch_goal(buffer[g_buf_ids].obs.achieved_goal) == self.extract_batch_goal(buffer[g_buf_ids].obs)).all()
            assert (self.extract_batch_goal(buffer[sg_buf_ids].obs.achieved_goal) == self.extract_batch_goal(buffer[sg_buf_ids].obs)).all()

            # compute s_g
            subgoal_encoding = model.encode_g(self.extract_batch_goal(buffer[sg_buf_ids].obs.achieved_goal))

        # since we want to adjust the encoding layers too, here we need grads.
        state_encoding = model.encode_s(self.extract_batch_goal(buffer[s_buf_ids].obs.achieved_goal))
        # this is the desired/predefined goal
        final_reached_goal_encoding = model.encode_g(self.extract_batch_goal(buffer[g_buf_ids].obs.achieved_goal))

        batch = batch[row_idxes]
        batch.state_encoding = state_encoding
        batch.final_reached_goal_encoding = final_reached_goal_encoding
        batch.subgoal_encoding = subgoal_encoding.data
        batch.trj_ws = trj_ws
        return batch

    def learn_worldmodel(self, batch, s_noise_level=1.0):
        """learn the world model, i.e., (s, a) -> s'
        """
        model = getattr(self, 'model')

        s_t = model.encode_s(batch.obs['observation'])
        a_t = torch.zeros((len(batch), self.max_action_num), dtype=torch.float).to(self.device)
        a_t[np.arange(len(batch)), batch.act] = 1.0

        s_nt_pred_by_wm = self.world_model(s_t + s_noise_level * torch.normal(torch.zeros_like(s_t), torch.ones_like(s_t)).to(self.device), a_t)
        o_nt_pred_wm = model.decoding_layers(s_nt_pred_by_wm)

        o_nt_pred_loss = F.mse_loss(o_nt_pred_wm,  model.imgarray2tensor(batch.obs_next['observation']))

        world_loss = o_nt_pred_loss + 1e-3 * (s_t**2).sum(dim=-1).mean()
        self.world_model_optim.zero_grad()
        world_loss.backward()
        self.world_model_optim.step()
        return world_loss

    def learn_act_dist(self, batch):
        model = getattr(self, 'model')

        s_t = model.encode_s(batch.obs['observation'])
        next_g_t = model.encode_g(self.extract_batch_goal(batch.obs_next['achieved_goal']))

        qvs = self.model.qnet(torch.cat((s_t, next_g_t), dim=-1))
        act_dists = torch.softmax(qvs, dim=-1)

        optimal_act_dists = torch.zeros(
            (len(batch), self.max_action_num),
            dtype=torch.float).to(self.device)

        optimal_act_dists[np.arange(len(batch)), batch.act] = 1.0
        # positive learning
        act_kl_divs = F.kl_div(
            torch.log(act_dists + 1e-18),
            optimal_act_dists,
            reduction='none').sum(dim=-1)

        loss = act_kl_divs.mean()
        # negative learning
        if self.neg_act_kl_w > 0:
            neg_next_g_t = model.encode_g(self.extract_batch_goal(batch.negative_goals))  # negative goal, for balance.
            qvs_neg = self.model.qnet(
                torch.cat((s_t, neg_next_g_t), dim=-1),
            )
            neg_act_dists = torch.softmax(qvs_neg, dim=-1)
            target_neg_act_dists = torch.ones_like(neg_act_dists) / self.max_action_num
            neg_act_kl_divs = F.kl_div(
                torch.log(neg_act_dists + 1e-18),
                target_neg_act_dists,
                reduction='none').sum(dim=-1)

            loss = loss + self.neg_act_kl_w * neg_act_kl_divs.mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Process that batched data so that it can be used for Q learning.
        - Compute the n-step return for Q-learning targets.
        - Sample and train different subgoals from the replay buffer and train the goal reducer.
        """
        if self.neg_act_kl_w > 0:
            batch.negative_goals = buffer.sample(len(batch))[0].obs.achieved_goal
        # indices
        # We put the subgoal learning here as
        # we may want to tune the subgoal learning frequency.
        # =============== Goal reducer learning start ===============

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

            if sg_loss.item() >= 0.1:
                self.turn_on_subgoal_helper = 1.0

        wm_loss = self.learn_worldmodel(batch)

        self.extra_learning_info.update({
            "wm_l": wm_loss.item(),
            'sg_coverage': len(self.subgoal_dataset_coverage) / 260**2
        })
        if sg_loss is not None:
            self.extra_learning_info["sg_l"] = sg_loss.item()

        if act_loss is not None:
            self.extra_learning_info["loss"] = act_loss.item()

        return batch

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        loss = None
        loss = self.learn_act_dist(batch)

        self._iter += 1
        if self._iter % 1000 == 0:
            torch.save(self.state_dict(), self.policy_dir / f'policy_{self._iter}.pth')
            torch.save(self.goal_reducer.state_dict(), self.policy_dir / f'gr_{self._iter}.pth')

        learning_res = {"iter": self._iter}
        if loss is not None:
            learning_res["loss"] = loss.item()
        # =============== Classical Q learning end ===============

        learning_res.update(self.extra_learning_info)

        return learning_res

    def after_train(self, epoch_passed: int) -> None:
        if epoch_passed > self.n_ep_explore:
            self.random_exploration = False
        else:
            self.random_exploration = True
