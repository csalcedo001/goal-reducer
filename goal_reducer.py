"""A set of utilities for subgoal planning.
"""

import numpy as np
import torch
from collections import defaultdict


def remove_loops_idx(a):
    clean_a = []
    clean_a_ids = []

    for aid, a_ele in enumerate(a):
        if a_ele not in clean_a:
            clean_a.append(a_ele)
            clean_a_ids.append(aid)
        else:
            a_ele_idx = clean_a.index(a_ele)
            clean_a = clean_a[:a_ele_idx + 1]

            clean_a_ids = clean_a_ids[:a_ele_idx + 1]
    return clean_a, clean_a_ids


def remove_loops(arr):
    """Remove loops in a trajectory.

    Args:
        arr (torch.Tensor): (T, neuron_dim)

    Returns:
        torch.Tensor: (T', neuron_dim)
    """
    arr_str = np.apply_along_axis(lambda x: ''.join(x), axis=1, arr=arr.astype(str))
    _, idx_list = remove_loops_idx(arr_str)
    # idx_list = []
    # assert arr.ndim == 2
    # for cur_idx in range(arr.shape[0]):
    #     if cur_idx == 0:
    #         idx_list.append(cur_idx)
    #         continue

    #     res = np.argwhere((arr[cur_idx] == arr[idx_list]).all(axis=1) == True)
    #     if len(res) > 0:
    #         repeat_idx = res[0][0]
    #         idx_list = idx_list[:repeat_idx]
    #     idx_list.append(cur_idx)
    # import ipdb; ipdb.set_trace() # fmt: off
    return np.array(idx_list)


def extract_subgoal_pos(traj_pos_list):
    """Extract subgoals from a trajectory list, randomly.

    Args:
        traj_pos_list (np.ndarray): 2D array of shape (traj_len, 2) containing
            the positions of the agent in the trajectory.

    Returns:
        tuple: (start_pos, subgoal_pos, goal_pos)
    """
    if len(traj_pos_list) <= 2:
        # if there is only one step transition, we just return the goal
        # position as the subgoal position.
        return traj_pos_list[0], traj_pos_list[-1], traj_pos_list[-1]
    else:
        # otherwise, we randomly sample a subgoal position between the start
        # and goal position but NOT the start or goal position.
        subgoal_id = np.random.choice(np.arange(1, len(traj_pos_list) - 1), 1)[0]
        assert (traj_pos_list[subgoal_id] != traj_pos_list[-1]).any()
        # try:
        #     assert (traj_pos_list[subgoal_id] != traj_pos_list[-1]).any()
        # except:
        #     import ipdb; ipdb.set_trace()
        #     pass
        return traj_pos_list[0], traj_pos_list[subgoal_id], traj_pos_list[-1]

    # subgoal_id = np.random.choice(np.arange(1, len(traj_pos_list)), 1)[0]
    # return traj_pos_list[0], traj_pos_list[subgoal_id], traj_pos_list[-1]


# now we have a list of trajectories, we need to extract subgoals.
def sample_subgoals(
        all_possible_idx,
        all_obs_reps,
        buffer,
        batch_size,
        max_steps=100,
        remove_loop=True,
):
    """Sample subgoals from the replay buffer.
    Args:
        all_possible_idx (dict): dict mapping state id to state index.
        all_obs_reps (torch.Tensor): tensor of shape (num_possible_states, 32)
            containing the representation of all possible states.
        buffer (ts.data.VectorReplayBuffer): replay buffer.
        batch_size (int): number of subgoals to sample.
        max_steps (int): maximum number of steps in a trajectory to consider.
        remove_loop (bool): whether to remove closed loop in trajectories.
    Returns:
        tuple: (s_pos_list, sg_pos_list, g_pos_list,
                s_reps, sg_reps, g_reps): The first three lists contain the
                start, subgoal, and goal positions of the sampled trajectories.
                The last three lists contain the representations of the start,
                subgoal, and goal states.
    """
    s_pos_list = []
    sg_pos_list = []
    g_pos_list = []

    s_rep_idx_list = []
    sg_rep_idx_list = []
    g_rep_idx_list = []

    while len(s_pos_list) < batch_size:
        all_indices = buffer.sample_indices(0)
        idx_list = np.random.choice(all_indices, batch_size, replace=False)
        indicess = [idx_list]
        for _ in range(max_steps - 1):
            indicess.append(buffer.next(indicess[-1]))
        indicess = np.stack(indicess)  # shape (T, batch_size)

        bs_unique_idx_list = []
        bs_traj_agent_pos_list = []
        bs_traj_metainfo_list = []
        bs_traj_agent_pos_noloop_list = []  # remove closed loop in trajectories
        for col_idx in range(batch_size):
            unique_idx_list = []
            # single trajectory
            for idx in indicess[:, col_idx]:
                if idx in unique_idx_list:
                    break
                else:
                    unique_idx_list.append(idx)
            traj_idx = np.array(unique_idx_list)
            bs_unique_idx_list.append(traj_idx)
            bs_traj_agent_pos_list.append(
                buffer[traj_idx].info.agent_pos
            )
            noloop_idx_list = remove_loops(buffer[traj_idx].info.agent_pos)
            bs_traj_agent_pos_noloop_list.append(
                buffer[traj_idx].info.agent_pos[noloop_idx_list]
            )
            bs_traj_metainfo_list.append(
                {
                    'agent_pos': buffer[traj_idx].info.agent_pos[0],
                    'goal_pos': buffer[traj_idx].info.goal_pos[0],
                }
            )

        full_traj_pos_list = [{
            'meta_data': bs_traj_metainfo_list[traj_idx],
            'agent_pos':bs_traj_agent_pos_list[traj_idx],
        } for traj_idx, poslist in enumerate(bs_traj_agent_pos_list)
            if len(poslist) > 3]

        noloop_traj_pos_list = [{
            'meta_data': bs_traj_metainfo_list[traj_idx],
            'agent_pos':bs_traj_agent_pos_noloop_list[traj_idx],
        } for traj_idx, poslist in enumerate(bs_traj_agent_pos_noloop_list)
            if len(poslist) > 3]

        if remove_loop:
            pos_list = noloop_traj_pos_list
        else:
            pos_list = full_traj_pos_list

        for traj in pos_list:
            s, sg, g = extract_subgoal_pos(traj['agent_pos'])
            # print(s, sg, g)
            s_pos_list.append(s)
            sg_pos_list.append(sg)
            g_pos_list.append(g)

            s_rep_idx = all_possible_idx[(s[0], s[1])]
            sg_rep_idx = all_possible_idx[(sg[0], sg[1])]
            g_rep_idx = all_possible_idx[(g[0], g[1])]
            s_rep_idx_list.append(s_rep_idx)
            sg_rep_idx_list.append(sg_rep_idx)
            g_rep_idx_list.append(g_rep_idx)

            # s_rep = all_obs_reps[s_rep_idx]
            # sg_rep = all_obs_reps[sg_rep_idx]
            # g_rep = all_obs_reps[g_rep_idx]

    s_reps = all_obs_reps[s_rep_idx_list][:batch_size]
    sg_reps = all_obs_reps[sg_rep_idx_list][:batch_size]
    g_reps = all_obs_reps[g_rep_idx_list][:batch_size]

    s_pos_list = np.stack(s_pos_list)[:batch_size]
    sg_pos_list = np.stack(sg_pos_list)[:batch_size]
    g_pos_list = np.stack(g_pos_list)[:batch_size]
    return (
        s_pos_list, sg_pos_list, g_pos_list,
        s_reps, sg_reps, g_reps,
    )


def subgoal_quality_metric(s_pos_list, g_pos_list, sg_pos_list, sg_rep_pred,
                           env, all_obs_reps, all_possible_idx_rev
                           ):
    sg_pred_indices = torch.cdist(sg_rep_pred.data, all_obs_reps).min(dim=1).indices.cpu().numpy()
    sg_pred_pos_list = []
    for sg_pred_idx in sg_pred_indices:
        sg_pred_pos_list.append(
            all_possible_idx_rev[sg_pred_idx]
        )

    idx_in_batch = 0
    quality_ratios = defaultdict(list)
    bias_ratios = defaultdict(list)
    for idx_in_batch in range(len(s_pos_list)):
        dist_sg_total = env.shortest_distance(tuple(s_pos_list[idx_in_batch]), tuple(g_pos_list[idx_in_batch]))
        # dist_sg_given = env.shortest_distance(tuple(s_pos_list[idx_in_batch]), tuple(sg_pos_list[idx_in_batch])
        #                                       ) + env.shortest_distance(
        #     tuple(sg_pos_list[idx_in_batch]), tuple(g_pos_list[idx_in_batch])
        # )
        dist_s2sg = env.shortest_distance(tuple(s_pos_list[idx_in_batch]), sg_pred_pos_list[idx_in_batch]
                                          )
        dist_sg2g = env.shortest_distance(
            sg_pred_pos_list[idx_in_batch], tuple(g_pos_list[idx_in_batch])
        )
        dist_sg_pred = dist_s2sg + dist_sg2g

        # quality_ratios[dist_sg_total].append(dist_sg_pred / dist_sg_given)
        if dist_sg_total >1 and dist_s2sg ==dist_sg_pred:
            quality_ratios[dist_sg_total].append(10.)
        else:
            quality_ratios[dist_sg_total].append(dist_sg_pred / dist_sg_total)
        bias_ratios[dist_sg_total].append(dist_s2sg / dist_sg_pred)

    subgoal_quality = {k: {
        'mean_sq': np.mean(quality_ratios[k]),
        'std_sq': np.std(quality_ratios[k]),
        'mean_b': np.mean(bias_ratios[k]),
        'std_b': np.std(bias_ratios[k]),
    } for k in quality_ratios.keys()}

    return subgoal_quality


# now we have a list of trajectories, we need to extract subgoals.
def sample_subgoals_with_priority(
        all_possible_idx,
        all_obs_reps,
        buffer,
        batch_size,
        max_steps=100,
        remove_loop=True,
):
    """Sample subgoals from the replay buffer with priority.
    The idea is simple:
        we randomly select a batch of starting points, and then we run
        max_steps steps of rollout. Based on the rollout, we randomly select
        the ending point between the start and the goal as the "goal" (notice
        that this is different from the goal in the environment).
        Then we perform the same operation between the start and the goal to
        get the subgoal.
        Lastly, we assign the subgoal a priority based on the distance between
        the starting point and the goal we chose.

    Args:
        all_possible_idx (dict): dict mapping state id to state index.
        all_obs_reps (torch.Tensor): tensor of shape (num_possible_states, 32)
            containing the representation of all possible states.
        buffer (ts.data.VectorReplayBuffer): replay buffer.
        batch_size (int): number of subgoals to sample.
        max_steps (int): maximum number of steps in a trajectory to consider.
        remove_loop (bool): whether to remove closed loop in trajectories.
    Returns:
        tuple: (s_pos_list, sg_pos_list, g_pos_list,
                s_reps, sg_reps, g_reps): The first three lists contain the
                start, subgoal, and goal positions of the sampled trajectories.
                The last three lists contain the representations of the start,
                subgoal, and goal states.
    """
    s_pos_list = []
    sg_pos_list = []
    g_pos_list = []

    s_rep_idx_list = []
    sg_rep_idx_list = []
    g_rep_idx_list = []
    priorities = []
    while len(s_pos_list) < batch_size:
        all_indices = buffer.sample_indices(0)
        idx_list = np.random.choice(all_indices, batch_size, replace=False)
        indicess = [idx_list]
        for _ in range(max_steps - 1):
            indicess.append(buffer.next(indicess[-1]))
        indicess = np.stack(indicess)  # shape (T, batch_size)

        bs_unique_idx_list = []
        bs_traj_agent_pos_list = []
        bs_traj_metainfo_list = []
        bs_traj_agent_pos_noloop_list = []  # remove closed loop in trajectories
        for col_idx in range(batch_size):
            unique_idx_list = []
            # single trajectory
            for idx in indicess[:, col_idx]:
                if idx in unique_idx_list:
                    break
                else:
                    unique_idx_list.append(idx)
            traj_idx = np.array(unique_idx_list)
            bs_unique_idx_list.append(traj_idx)
            bs_traj_agent_pos_list.append(
                buffer[traj_idx].info.agent_pos
            )
            noloop_idx_list = remove_loops(buffer[traj_idx].info.agent_pos)
            bs_traj_agent_pos_noloop_list.append(
                buffer[traj_idx].info.agent_pos[noloop_idx_list]
            )
            bs_traj_metainfo_list.append(
                {
                    'agent_pos': buffer[traj_idx].info.agent_pos[0],
                    'goal_pos': buffer[traj_idx].info.goal_pos[0],
                }
            )

        full_traj_pos_list = [{
            'meta_data': bs_traj_metainfo_list[traj_idx],
            'agent_pos':bs_traj_agent_pos_list[traj_idx],
        } for traj_idx, poslist_ in enumerate(bs_traj_agent_pos_list)
            if len(poslist_) > 1]

        noloop_traj_pos_list = [{
            'meta_data': bs_traj_metainfo_list[traj_idx],
            'agent_pos':bs_traj_agent_pos_noloop_list[traj_idx],
        } for traj_idx, poslist_ in enumerate(bs_traj_agent_pos_noloop_list)
            if len(poslist_) > 1]

        if remove_loop:
            pos_list = noloop_traj_pos_list
        else:
            pos_list = full_traj_pos_list

        for traj in pos_list:
            s, sg, g = extract_subgoal_pos(traj['agent_pos'])
            # priority = 1. / traj['agent_pos'].shape[0]
            priority = traj['agent_pos'].shape[0]
            # priority = 1. / np.unique(traj['agent_pos'], axis=0).shape[0]
            # if we can remove all loops, we can do 1./np.unique(traj['agent_pos'], axis=0).shape[0]
            # import ipdb; ipdb.set_trace() # fmt: off
            # print(s, sg, g)
            s_pos_list.append(s)
            sg_pos_list.append(sg)
            g_pos_list.append(g)
            priorities.append(priority) 

            s_rep_idx = all_possible_idx[(s[0], s[1])]
            sg_rep_idx = all_possible_idx[(sg[0], sg[1])]
            g_rep_idx = all_possible_idx[(g[0], g[1])]
            s_rep_idx_list.append(s_rep_idx)
            sg_rep_idx_list.append(sg_rep_idx)
            g_rep_idx_list.append(g_rep_idx)

            # s_rep = all_obs_reps[s_rep_idx]
            # sg_rep = all_obs_reps[sg_rep_idx]
            # g_rep = all_obs_reps[g_rep_idx]

    s_reps = all_obs_reps[s_rep_idx_list][:batch_size]
    sg_reps = all_obs_reps[sg_rep_idx_list][:batch_size]
    g_reps = all_obs_reps[g_rep_idx_list][:batch_size]
    priorities = torch.Tensor(priorities)[:batch_size].to(s_reps.device)

    s_pos_list = np.stack(s_pos_list)[:batch_size]
    sg_pos_list = np.stack(sg_pos_list)[:batch_size]
    g_pos_list = np.stack(g_pos_list)[:batch_size]
    return (
        s_pos_list, sg_pos_list, g_pos_list,
        s_reps, sg_reps, g_reps,
        priorities
    )
