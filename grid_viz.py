import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tasks
from tianshou.data import ReplayBuffer
from pathlib import Path
import matplotlib.animation as animation


def traj_viz_minigrid(buf: ReplayBuffer, name_predix=''):
    max_t = len(buf)
    rgb_tf = np.array([[0.2989, 0.5870, 0.1140]])
    single_size = 2
    cols = 2
    has_goal = hasattr(buf.info, 'goal_frame')
    if has_goal:
        cols = 4

    fig, axes = plt.subplots(
        max_t,
        cols,
        figsize=[cols * single_size + 0.5, max_t * single_size],
    )
    axes[0, 0].set_title('agent view')
    axes[0, 1].set_title('top view')
    if has_goal:
        axes[0, 2].set_title('goal top view')
        axes[0, 3].set_title('goal agent view')

    for t in range(max_t):
        obs_next_image = np.dot(buf.obs_next[t]['image'].astype(float),
                                rgb_tf.T)[:, :, 0]
        obs_next_normalized = (obs_next_image -
                               obs_next_image.min()) / obs_next_image.max()
        axes[t, 0].pcolormesh(
            obs_next_normalized,
            edgecolors='gray',
            linewidth=1,
            vmin=0,
            vmax=1,
        )
        axes[t, 0].set_aspect('equal')
        axes[t, 0].scatter(x=obs_next_normalized.shape[1] - 0.5,
                           y=obs_next_normalized.shape[0] // 2 + 1 - 0.5,
                           s=100,
                           marker="<",
                           c='red')
        axes[t, 0].set_ylabel(f't={t+1}', rotation=0, labelpad=20)

        axes[t, 1].set_xlabel(
            f"Act(prev)={tasks.minigrids.common.Actions(buf.act[t]).name}\n"
            f"Loc={buf.info['agent_pos'][t]}")
        axes[t, 0].set_xticks([])
        axes[t, 0].set_yticks([])

        axes[t, 1].set_xticks([])
        axes[t, 1].set_yticks([])
        axes[t, 1].imshow(buf.info['frame'][t])

        if has_goal:
            axes[t, 2].set_xticks([])
            axes[t, 2].set_yticks([])
            axes[t, 2].imshow(buf.info['goal_frame'][t])

            axes[t, 3].set_xticks([])
            axes[t, 3].set_yticks([])
            goal_image = np.dot(buf.obs_next[t]['goal_image'].astype(float),
                                rgb_tf.T)[:, :, 0]
            goal_normalized = (goal_image -
                               goal_image.min()) / goal_image.max()
            axes[t, 3].pcolormesh(
                goal_normalized,
                edgecolors='gray',
                linewidth=1,
                vmin=0,
                vmax=1,
            )
            axes[t, 3].set_aspect('equal')
            axes[t, 3].scatter(x=goal_normalized.shape[1] - 0.5,
                               y=goal_normalized.shape[0] // 2 + 1 - 0.5,
                               s=50,
                               marker="<",
                               alpha=0.2,
                               c='red')

    plt.tight_layout()
    fn = f'figs/{name_predix}{max_t}.png'
    print(f"save to {fn}")
    plt.savefig(fn)


def traj_viz_video(frames: np.ndarray, fn: str, fig_dir: str = 'figs'):
    if frames.ndim < 3:
        return
    if frames.shape[0] < 2:
        return
    if type(fig_dir) is not Path:
        fig_dir = Path(fig_dir)
    fig_dir.mkdir(exist_ok=True)
    width, height = matplotlib.rcParams["figure.figsize"]
    size = min(width, height)
    plt.clf()
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    def data_gen(t=0):
        for t, f in enumerate(frames):
            yield t, f

    arr1 = [None]

    def update(data):
        tw, phase = data
        ax.set_title("t = {}".format(tw))
        ax.set_xticks([])
        ax.set_yticks([])
        if arr1[0]:
            arr1[0].remove()
        arr1[0] = ax.imshow(phase)

    ani = animation.FuncAnimation(fig,
                                  update,
                                  data_gen,
                                  interval=200,
                                  blit=False,
                                  repeat=False)
    ani.save(fig_dir / f"{fn}.mp4")


# =============== tmp function ==============
def _draw_im(view_new, ax):
    rgb_tf = np.array([[0.2989, 0.5870, 0.1140]])
    im_new = np.dot(view_new.astype(float), rgb_tf.T)[:, :, 0]
    im_new = (im_new - im_new.min()) / im_new.max()
    ax.pcolormesh(
        im_new,
        edgecolors='gray',
        linewidth=1,
        vmin=0,
        vmax=1,
    )
    ax.set_aspect('equal')
    ax.scatter(x=im_new.shape[1] - 0.5,
               y=im_new.shape[0] // 2 + 1 - 0.5,
               s=100,
               marker="<",
               c='red')
    ax.set_xticks([])
    ax.set_yticks([])


def _view_test(inputs, ix, info, idx):
    view_new = inputs[idx].data.permute(1, 2, 0).cpu().numpy()
    view_old = ix[idx].data.permute(1, 2, 0).cpu().numpy()
    view_god = info.frame[idx]
    fig, axes = plt.subplots(1, 3)
    _draw_im(view_new, axes[0])
    _draw_im(view_old, axes[1])
    axes[2].imshow(view_god)
    plt.savefig(f'figs/xxx_{idx}.png')
