"""
This file contains the state graphs used in the paper, including
- RandomSG: a random state graph
- FourRoomSG: a four-room state graph
"""
# import multiprocessing as mp
import copy
import time
from collections import defaultdict
from itertools import permutations
from typing import List, Optional, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
import pathos.multiprocessing as mp
import torch
from scipy.sparse import csr_matrix
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import dijkstra
import matplotlib.colors as mcolors
from numba import njit
import numba as nb
import tasks  # noqa
from enum import Enum


class SamplingStrategy(Enum):
    RANDOM = "Random"
    TRAJECTORY = "Trajectory"
    LOOP_REMOVAL = "Loop-removal"


def remove_loops(arr, arr_embs, rho=0.01):

    clean_a: List[int] = []
    clean_a_ids: List[int] = []
    clean_a_embs: List[torch.Tensor] = []

    for aid, (a_ele, a_emb) in enumerate(zip(arr, arr_embs)):
        if len(clean_a) == 0:
            clean_a.append(a_ele)
            clean_a_ids.append(aid)
            clean_a_embs.append(a_emb.clone())
        else:
            dists = torch.cdist(a_emb.unsqueeze(0), torch.stack(clean_a_embs))[0]
            min_v, min_idx = torch.min(dists, dim=0)
            min_idx = min_idx.item()
            min_v = min_v.item()
            if min_v >= rho:
            # if a_ele not in clean_a:
                # torch.cdist(a_emb, torch.stack(clean_a_embs))
                # import ipdb; ipdb.set_trace() # fmt: off
                clean_a.append(a_ele)
                clean_a_ids.append(aid)
                clean_a_embs.append(a_emb.clone())
            else:
                # dists = torch.cdist(a_emb.unsqueeze(0), torch.stack(clean_a_embs))[0]
                # import ipdb; ipdb.set_trace()  # noqa
                # a_ele_idx = min_idx # clean_a.index(a_ele)
                a_ele_idx = clean_a.index(a_ele)
                clean_a = clean_a[: a_ele_idx + 1]
                clean_a_ids = clean_a_ids[: a_ele_idx + 1]
                clean_a_embs = clean_a_embs[: a_ele_idx + 1]

    return clean_a, clean_a_ids


@njit
def remove_loops_j(a: np.ndarray) -> Tuple[float, int, int]:
    clean_a = nb.typed.List.empty_list(nb.int64)
    clean_a_ids = nb.typed.List.empty_list(nb.int64)

    for aid, a_ele in enumerate(a):
        if a_ele not in clean_a:
            clean_a.append(a_ele)
            clean_a_ids.append(aid)
        else:
            a_ele_idx = clean_a.index(a_ele)
            clean_a = clean_a[: a_ele_idx + 1]
            clean_a_ids = clean_a_ids[: a_ele_idx + 1]
    # return clean_a, clean_a_ids

    assert clean_a_ids[0] == 0
    if len(clean_a_ids) < 2:
        w_trj = 0.0
        # mask[idx_in_batch] = 0
        # invalid
        terminal_index = a[0]
        subgoal_index = a[0]

        # subgoal_indices.append(indices_single[0])
        # terminal_indices.append(indices_single[0])
    else:
        noloop_length = len(clean_a_ids)
        w_trj = 1.0 / (noloop_length + 5.0)
        id_nids = np.arange(noloop_length)
        terminal_idid = np.random.choice(id_nids[1:], 1).item()  # at least 2
        if len(id_nids[1:terminal_idid]) == 0:
            sg_idid = terminal_idid
        else:
            sg_idid = np.random.choice(id_nids[1 : terminal_idid + 1], 1).item()

        terminal_index = a[terminal_idid]
        subgoal_index = a[sg_idid]

        # subgoal_indices.append(subgoal_index)
        # terminal_indices.append(terminal_index)
    return w_trj, subgoal_index, terminal_index


class BaseStageGraph:
    """Base class for state graphs."""

    conn_matrix: np.ndarray
    s_enc: torch.nn.Module
    dist_matrix: np.ndarray
    s_embs: torch.Tensor
    max_steps: int
    # emb_scale: float

    def __init__(self, state_dim: int, emb_scale=1.0, *args, **kwargs):
        """Initialize a state graph.

        Args:
            state_dim (int): dimension of the state embedding.
            device (str, optional): device to assign embeddings.
                Defaults to 'cpu'.
        """
        self.state_dim = state_dim
        self.emb_scale = emb_scale
        self.init_graph(*args, **kwargs)

        # self.get_sg_quality()
        self._subgoal_quality = None

    def __str__(self) -> str:
        return type(self).__name__

    def init_graph(
        self,
    ):
        raise FileNotFoundError

    def get_sg_quality(
        self,
    ):
        # print('Calculating subgoal quality')
        # get subgoal quality dicts

        sid_ids = np.arange(self.dist_matrix.shape[0])
        # prior_subgoal_quality = defaultdict(dict)
        prior_subgoal_quality = defaultdict(lambda: np.zeros(self.dist_matrix.shape[0]))

        for s_idid, g_idid in permutations(range(self.dist_matrix.shape[0]), 2):
            dis_shortest = self.shortest_path(s_idid, g_idid)
            assert dis_shortest > 0
            for sg_idid in sid_ids:
                if sg_idid == s_idid or sg_idid == g_idid:
                    continue
                dis_sum = self.shortest_path(s_idid, sg_idid) + self.shortest_path(sg_idid, g_idid)
                assert dis_sum > 0
                prior_subgoal_quality[(s_idid, g_idid)][sg_idid] = float(dis_shortest) / float(dis_sum)

        self._subgoal_quality = prior_subgoal_quality

    @property
    def subgoal_quality(self):
        if self._subgoal_quality is None:
            self.get_sg_quality()
        return self._subgoal_quality

    @staticmethod
    def walk(conn: np.ndarray, max_steps: int, start: Optional[int] = None, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        if start is None:
            start = np.random.randint(conn.shape[0])
        path = [start]
        for _ in range(max_steps):
            # completely random walk
            ids = np.where(conn[path[-1]] > 0)[0]
            path.append(np.random.choice(ids, 1).item())

        return copy.deepcopy(path)

    # @abstractmethod
    def shortest_path(self, starts, ends):
        return self.dist_matrix[starts, ends]

    # @abstractmethod
    def visualize_s_g_sg(self, states: np.ndarray) -> Figure:
        """Gien a 2d numpy array of shape (num_states, state_dim) or
        a 3d numpy array of shape (batch, num_states, state_dim) visualize
        the path.
        If the input is 3d, reduce the color opacity for each single trajectory.

        Args:
            states (_type_): _description_
        """
        raise NotImplementedError

    def visualize_sg_tree(self, s: int, gtree_levels: List[List[int]]) -> Figure:
        raise NotImplementedError


class RandomSG(BaseStageGraph):
    def __init__(self, state_dim: int, num_vertices: int, *args, **kwargs):
        self.num_vertices = num_vertices
        super().__init__(state_dim, *args, **kwargs)
        self.s_enc = torch.nn.Sequential(
            torch.nn.Embedding(self.num_vertices, state_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(state_dim, state_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(state_dim, state_dim),
        )
        self.s_enc.eval()
        with torch.no_grad():
            self.s_embs = self.s_enc(torch.arange(self.num_vertices)).data * self.emb_scale
        dists =  torch.cdist(self.s_embs, self.s_embs).flatten()
        self.rho = dists[dists!=0].min().item()
        self.max_steps = 100

    def init_graph(self, seed=1926):
        # G = nx.gnm_random_graph(self.num_vertices, self.num_vertices * 3, seed=seed)
        G = nx.random_geometric_graph(self.num_vertices, 0.2, seed=896803)
        # G = nx.gnm_random_graph(self.num_vertices, self.num_vertices * 2,
        #                         seed=seed)
        # Add random weights to the edges
        for u, v in G.edges():
            G.edges[u, v]["weight"] = 1.0
        adj_matrix = nx.adjacency_matrix(G).todense()

        # x = np.random.rand(num_nodes) > 0.2
        # y = np.random.rand(num_nodes) > 0.5
        # adj_matrix = adj_matrix+ np.outer(x, y)

        # print(adj_matrix.max())
        np.fill_diagonal(adj_matrix, 1.0)
        self.conn_matrix = adj_matrix
        # import ipdb; ipdb.set_trace() # fmt: off
        dist_matrix, predecessors = dijkstra(csgraph=adj_matrix, directed=True, return_predecessors=True)
        self.dist_matrix = dist_matrix

    def visualize_s_g_sg(self, states: np.ndarray):
        """Gien a 2d numpy array of shape (num_states, state_dim) or
        a 3d numpy array of shape (batch, num_states, state_dim) visualize
        the path.
        If the input is 3d, reduce the color opacity for each single trajectory.

        Args:
            states (_type_): _description_
        """
        G = nx.from_numpy_array(self.conn_matrix)
        fig, ax = plt.subplots(figsize=(10, 10))
        # pos = {i: self.pos[i] for i in range(len(self.pos))}
        # pos = nx.spectral_layout(G)
        pos = nx.spring_layout(G)
        s = states[0]
        g = states[-1]
        sgs = states[1:-1].tolist()

        nx.draw(
            G,
            pos=pos,
            with_labels=False,
            ax=ax,
            node_color="black",
            node_size=50,
        )
        nx.draw_networkx_nodes(G, pos, nodelist=[s], node_shape="^", node_color=["blue"])
        nx.draw_networkx_nodes(G, pos, nodelist=sgs, node_shape="*", node_color="red")
        nx.draw_networkx_nodes(G, pos, nodelist=[g], node_shape="*", node_color=["salmon"])
        return fig


class FourRoomSG(BaseStageGraph):
    def __init__(self, state_dim: int, env_name: Optional[str] = None, env=None, *args, **kwargs):
        if env is None:
            assert env_name is not None

        self.env_name = env_name
        self.env = env
        self.max_steps = 340
        if env_name is not None:
            if "19" in self.env_name:
                # self.max_steps = 1000
                self.max_steps = 1000

        super().__init__(state_dim, *args, **kwargs)
        self.s_enc = torch.nn.Sequential(
            torch.nn.Embedding(self.num_vertices, state_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(state_dim, state_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(state_dim, state_dim),
        )
        with torch.no_grad():
            self.s_embs = self.s_enc(torch.arange(self.num_vertices)).data * self.emb_scale
        dists =  torch.cdist(self.s_embs, self.s_embs).flatten()
        self.rho = dists[dists!=0].min().item()
        self.max_steps = 50

    def __repr__(self) -> str:
        return f"FourRoomSG({self.env_name})"

    def __str__(self) -> str:
        return self.__repr__()

    def init_graph(self):
        if not self.env:
            env = gym.make(self.env_name, max_steps=self.max_steps, agent_view_size=13)
        else:
            env = self.env
        env.reset()
        self.conn_matrix = env.unwrapped.conn_matrix
        self.dist_matrix = env.unwrapped.id_dist_matrix
        self.pos = env.unwrapped.available_coords

        self.num_vertices = env.unwrapped.conn_matrix.shape[0]

    def visualize_s_g_sg(self, states: np.ndarray) -> Figure:
        """Gien a 2d numpy array of shape (num_states, state_dim) or
        a 3d numpy array of shape (batch, num_states, state_dim) visualize
        the path.
        If the input is 3d, reduce the color opacity for each single trajectory.

        Args:
            states (_type_): _description_
        """
        G = nx.from_numpy_array(self.conn_matrix)
        fig, ax = plt.subplots(figsize=(4, 4))
        pos = {i: self.pos[i] for i in range(len(self.pos))}
        s = states[0]
        g = states[-1]
        sgs = states[1:-1].tolist()

        nx.draw(
            G,
            pos=pos,
            with_labels=False,
            ax=ax,
            node_color="black",
            node_size=50,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[s],
            node_color=["blue"],
            node_size=40,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=sgs,
            node_color="salmon",
            node_size=40,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[g],
            node_color=["red"],
            node_size=40,
        )
        return fig

    def visualize_sg_tree(self, s: int, gtree_levels: List[List[int]], node_size: int = 50, g_size: int = 20, ax=None):
        # -> Figure:
        """Visualize the subgoal tree.

        Args:
            states (_type_): _description_
        """
        G = nx.from_numpy_array(self.conn_matrix)
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))

        pos = {i: self.pos[i] for i in range(len(self.pos))}

        nx.draw(
            G,
            pos=pos,
            with_labels=False,
            ax=ax,
            node_color="white",
            edge_color="white",
            node_size=node_size,
        )
        # current node

        # Create a color gradient from red to white
        red = mcolors.CSS4_COLORS["green"]
        white = mcolors.CSS4_COLORS["lightgreen"]
        colors = mcolors.LinearSegmentedColormap.from_list("red_to_white", [red, white])

        for gidx, gv in enumerate(gtree_levels):
            node_color = colors(gidx / len(gtree_levels))
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=list(set(gv)),
                node_color=[node_color],
                node_size=g_size,
                ax=ax,
            )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[s],
            node_color=["red"],
            node_size=g_size,
            ax=ax,
        )
        # fig = ax.get_figure()
        # return fig

    def visualize_sgtree_groups(self, s_init: int, g_ultimate: int, gtree_levels: List[Tuple[int, List[int]]], node_size: int = 50, g_size: int = 10, ax=None):
        # -> Figure:
        """Visualize the subgoal tree.

        Args:
            states (_type_): _description_
        """
        G = nx.from_numpy_array(self.conn_matrix)
        if ax is None:
            # _, ax = plt.subplots(figsize=(4, 4))
            ax = plt.gca()

        pos = {i: self.pos[i] for i in range(len(self.pos))}

        nx.draw(
            G,
            pos=pos,
            with_labels=False,
            ax=ax,
            node_color="white",
            edge_color="white",
            node_size=node_size,
        )
        # current node

        # Create a color gradient from red to white
        red = mcolors.CSS4_COLORS["green"]
        white = mcolors.CSS4_COLORS["lightgreen"]
        colors = mcolors.LinearSegmentedColormap.from_list("red_to_white", [red, white])

        for gidx, (s, gv) in enumerate(gtree_levels):
            node_color = colors(gidx / len(gtree_levels))
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=list(set(gv)),
                node_color=[node_color],
                # node_size=g_size*3,
                node_size=node_size * 3,
                alpha=0.1,
                ax=ax,
            )

        for gidx, (s, gv) in enumerate(gtree_levels):
            node_color = colors(gidx / len(gtree_levels))
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[s],
                node_color=["red"],
                node_size=g_size,
                # alpha=0.1,
                ax=ax,
            )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[s_init],
            node_color=["red"],
            node_size=g_size,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[g_ultimate],
            node_color=["green"],
            node_size=g_size,
            ax=ax,
        )
        fig = ax.get_figure()
        # ax.set_title(f'')
        return fig


def walk_on_state_graph(graph: BaseStageGraph, num_runs: int, max_steps: int, n_cores: int = 2, seed_offset: int = 0):
    """Random walk on state graphs to collect transitions.

    Args:
        graph (BaseStageGraph): The graph to run
        num_runs (int): The number of walks to perform.
        max_steps (int): The maximum number of steps in a single run.
        n_cores (int, optional): The number of cores to use. Defaults to 2.
    Returns:
        np.ndarray: The result of the random walks. Shape (num_runs, max_steps)
    """



    def _walk_on_graph(run_id):
        # trj_ids = graph.walk(graph.conn_matrix, max_steps, seed=run_id)
        seed = run_id
        if seed is not None:
            np.random.seed(seed)
        # if start is None:
        conn = graph.conn_matrix.copy()
        start = np.random.randint(conn.shape[0])
        # path = [start]
        path = list()
        path.append(start)
        for _ in range(max_steps):
            # completely random walk
            ids = np.where(conn[path[-1]] > 0)[0]
            path.append(np.random.choice(ids, 1).item())
        trj_ids = copy.deepcopy(path)
        # import ipdb; ipdb.set_trace()  # noqa
        trj_embs = graph.s_embs[trj_ids].data.clone()
        # trj_embs = None
        _, clean_a_ids = remove_loops(trj_ids, trj_embs, graph.rho)
        return copy.deepcopy(trj_ids), copy.deepcopy(clean_a_ids)

    path_indices = []
    noloop_path_indices = []
    # run_id_list_base = int(time.time() - 1e9)
    tried = 0

    # pool = mp.Pool(processes=n_cores)
    while len(path_indices) < num_runs:
        tried += 1
        # run_id_list = np.arange(num_runs - len(path_indices)) * 100 + seed_offset * 1000 + int(time.time())
        run_id_list = int(time.time()) + np.random.randint(100000000, size=num_runs - len(path_indices))
        # run_id_list = [None for _ in range(num_runs - len(path_indices))]
        # result = pool.map(_walk_on_graph, run_id_list)
        result = [_walk_on_graph(r) for r in run_id_list]

        c_res = 0
        for res in result:
            if res[0][0] == res[0][-1] or res[1][0] == res[1][-1]:
                # print('tried', tried, 'hitted', res[0][0] == res[0][-1], res[1][0] == res[1][-1])
                if tried > 3:
                    # print(run_id_list)
                    # import ipdb; ipdb.set_trace() # fmt: off
                    pass
                continue
            c_res += 1
            path_indices.append(res[0])
            noloop_path_indices.append(res[1])

        # run_id_list_base += c_res

    return np.array(path_indices)[:num_runs], noloop_path_indices[:num_runs]


if __name__ == "__main__":
    graph = RandomSG(10, 100)
    result = walk_on_state_graph(graph, 100, 10)
    print(result)
    print(result.shape)
