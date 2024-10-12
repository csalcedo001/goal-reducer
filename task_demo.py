"""
Manually control different tasks.
"""
import os
import time

import click
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import tasks  # noqa

os.environ["DISPLAY"] = ":1"


@click.group()
def cli():
    pass


@cli.command
@click.option("--env-name", "-e", type=str)
@click.option("--tile-size", type=int, default=32)
@click.option("--agent-view", type=bool, default=False)
def minigrids(env_name, tile_size, agent_view):
    from minigrid.minigrid_env import MiniGridEnv
    from minigrid.utils.window import Window
    from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

    class ManualControlMiniGrids:
        def __init__(
            self,
            env_name: str,
            env: MiniGridEnv,
            agent_view: bool = False,
            window: Window = None,
            seed=None,
        ) -> None:
            self.env = env
            self.agent_view = agent_view
            self.seed = seed

            if window is None:
                window = Window("minigrid - " + str(env_name))
            self.window = window
            self.window.reg_key_handler(self.key_handler)

        def start(self):
            """Start the window display with blocking event loop"""
            self.reset(self.seed)
            self.window.show(block=True)

        def step(self, action: MiniGridEnv.Actions):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.agent_view:
                print(
                    f"step={self.env.step_count}, reward={reward:.2f}, info {info}s")
            else:
                print(
                    f"step={self.env.step_count}, reward={reward:.2f}, info {info}\n"
                    # f"achieved_goal={obs['achieved_goal']}, desired_goal={obs['desired_goal']}"
                )

            if terminated:
                print("terminated!")
                self.reset(self.seed)
            elif truncated:
                print("truncated!")
                self.reset(self.seed)
            else:
                self.redraw()

        def redraw(self):
            frame = self.env.get_frame(agent_pov=self.agent_view)
            self.window.show_img(frame)

        def reset(self, seed=None):
            self.env.reset(seed=seed)

            if hasattr(self.env, "mission"):
                print("Mission: %s" % self.env.mission)
                self.window.set_caption(self.env.mission)

            self.redraw()

        def key_handler(self, event):
            key: str = event.key
            print("pressed", key)

            if key == "escape":
                self.window.close()
                return
            if key == "backspace":
                self.reset()
                return

            key_to_action = {
                "left": MiniGridEnv.Actions.left,
                "right": MiniGridEnv.Actions.right,
                "up": MiniGridEnv.Actions.forward,
                " ": MiniGridEnv.Actions.toggle,
                "pageup": MiniGridEnv.Actions.pickup,
                "pagedown": MiniGridEnv.Actions.drop,
                "enter": MiniGridEnv.Actions.done,
            }

            action = key_to_action[key]
            self.step(action)

    env_name
    env: MiniGridEnv = gym.make(
        env_name, tile_size=tile_size, agent_view_size=13)
    if agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, env.tile_size)
        env = ImgObsWrapper(env)

    seed = 12
    manual_control = ManualControlMiniGrids(
        env_name, env, agent_view=agent_view, seed=seed)
    manual_control.start()


@cli.command
@click.option("--env-name", "-e", type=str)
@click.option("--tile-size", type=int, default=32)
@click.option("--agent-view", type=bool, default=False)
def tvminigrids(env_name, tile_size, agent_view):
    """Topview minigrids"""
    assert "tasks.TVMGFR" in env_name, "Only works for MGFR"
    from minigrid.minigrid_env import MiniGridEnv
    from minigrid.utils.window import Window
    from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

    from tasks.minigrids.topview_fourrooms import TopViewGcFourRoomsEnv

    class ManualControlMiniGrids:
        def __init__(
            self,
            env_name: str,
            env: TopViewGcFourRoomsEnv,
            agent_view: bool = False,
            window: Window = None,
            seed=None,
        ) -> None:
            self.env = env
            self.agent_view = agent_view
            self.seed = seed

            if window is None:
                window = Window("minigrid - " + str(env_name))
            self.window = window
            self.window.reg_key_handler(self.key_handler)

        def start(self):
            """Start the window display with blocking event loop"""
            self.reset(self.seed)
            self.window.show(block=True)

        def step(self, action: TopViewGcFourRoomsEnv.Actions):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.agent_view:
                print(
                    f"step={self.env.step_count}, reward={reward:.2f}, info {info}s")
            else:
                print(
                    f"step={self.env.step_count}, reward={reward:.2f}, info {info}\n"
                    # f"achieved_goal={obs['achieved_goal']}, desired_goal={obs['desired_goal']}"
                )

            if terminated:
                print("terminated!")
                self.reset(self.seed)
            elif truncated:
                print("truncated!")
                self.reset(self.seed)
            else:
                self.redraw()

        def redraw(self):
            frame = self.env.get_frame(agent_pov=self.agent_view)
            self.window.show_img(frame)

        def reset(self, seed=None):
            self.env.reset(seed=seed)

            if hasattr(self.env, "mission"):
                print("Mission: %s" % self.env.mission)
                self.window.set_caption(self.env.mission)

            self.redraw()

        def key_handler(self, event):
            key: str = event.key
            print("pressed", key)

            if key == "escape":
                self.window.close()
                return
            if key == "backspace":
                self.reset()
                return

            key_to_action = {
                "left": TopViewGcFourRoomsEnv.Actions.left,
                "right": TopViewGcFourRoomsEnv.Actions.right,
                "up": TopViewGcFourRoomsEnv.Actions.up,
                "down": TopViewGcFourRoomsEnv.Actions.down,
            }

            action = key_to_action[key]
            self.step(action)

    env: TopViewGcFourRoomsEnv = gym.make(
        env_name, tile_size=tile_size, agent_view_size=13)
    if agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, env.tile_size)
        env = ImgObsWrapper(env)

    seed = 12
    manual_control = ManualControlMiniGrids(
        env_name, env, agent_view=agent_view, seed=seed)
    manual_control.start()


@cli.command
@click.option("--env", "-e", type=int, default=0)
@click.option("--top-view", type=bool, default=False)
def miniworlds(env, top_view):
    import math

    import gym_miniworld
    import pyglet
    from pyglet.window import key

    env_ids = [
        "tasks/MiniWorldsOneRoom-10x10-RandAgent-RandGoal-v0",
        gym_miniworld.envs.env_ids[0],
    ]
    view_mode = "top" if top_view else "agent"
    env = gym.make(env_ids[env], view=view_mode, render_mode="human")

    no_time_limit = True
    domain_rand = True
    if no_time_limit:
        env.max_episode_steps = math.inf
    if domain_rand:
        env.domain_rand = True

    print("============")
    print("Instructions")
    print("============")
    print("move: arrow keys\npickup: P\ndrop: D\ndone: ENTER\nquit: ESC")
    print("============")
    env.reset()

    # Create the display window
    env.render()

    def step(action):
        print("step {}/{}: {}".format(env.step_count + 1,
              env.max_episode_steps, env.actions(action).name))

        obs, reward, termination, truncation, info = env.step(action)

        if reward > 0:
            print(f"reward={reward:.2f}")

        if termination or truncation:
            print("done!")
            env.reset()

        env.render()

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        """
        This handler processes keyboard commands that
        control the simulation
        """

        if symbol == key.BACKSPACE or symbol == key.SLASH:
            print("RESET")
            env.reset()
            env.render()
            return

        if symbol == key.ESCAPE:
            env.close()
            # sys.exit(0)

        if symbol == key.UP:
            step(env.actions.move_forward)
        elif symbol == key.DOWN:
            step(env.actions.move_back)

        elif symbol == key.LEFT:
            step(env.actions.turn_left)
        elif symbol == key.RIGHT:
            step(env.actions.turn_right)

        elif symbol == key.PAGEUP or symbol == key.P:
            step(env.actions.pickup)
        elif symbol == key.PAGEDOWN or symbol == key.D:
            step(env.actions.drop)

        elif symbol == key.ENTER:
            step(env.actions.done)

    @env.unwrapped.window.event
    def on_key_release(symbol, modifiers):
        pass

    @env.unwrapped.window.event
    def on_draw():
        env.render()

    @env.unwrapped.window.event
    def on_close():
        pyglet.app.exit()

    # Enter main event loop
    pyglet.app.run()

    env.close()


@cli.command
@click.option("--render", type=str, default="rgb_array")
@click.option("--T", type=int, default=100)
@click.option("--noise", type=float, default=0.01)
@click.option("--ep", type=int, default=400)
@click.option("--plot", type=bool, default=False)
def reach(render, t, noise, ep, plot):
    """Robot arm reach task"""
    from tasks.robotarm.reach import ReachEnv

    env_name = "tasks.RobotArmReach-RARG-GI"
    assert render in ("rgb_array", "human")
    render_mode = render
    env: ReachEnv = gym.make(
        env_name,
        seed=1,
        render_width=480,
        render_height=480,
        max_steps=100,
        init_noise_scale=noise,
        render_mode=render_mode,
    )

    max_T = t
    g_t = 0
    reward_total = 0
    goals = []
    current_positions = []
    reward_totals = []
    dis_all = []
    observation, info = env.reset()
    noise_amp = 0.05
    act_th = 1.0
    while True:
        g_t += 1
        # action = env.action_space.sample()  # random action

        current_position = observation["observation"][0:3] + \
            np.random.randn(3) * noise_amp
        goal = observation["desired_goal"][0:3] + np.random.randn(3) * noise_amp

        # oracle_subgoal = current_position + 0.5 * (goal - current_position)
        action = 5.0 * (goal - current_position)

        action = np.clip(action, -act_th, act_th)

        # action = env.action_space.sample()  # random action

        next_observation, reward, terminated, truncated, info = env.step(action)
        reward_total += reward
        if render_mode == "human":
            time.sleep(0.1)
        observation = next_observation
        if g_t == max_T:
            terminated = True

            # reset
        if terminated or truncated:
            reward_totals.append(reward_total)
            mean_reward = np.mean(reward_totals)
            std_reward = np.std(reward_totals)
            observation, info = env.reset()
            goal = observation["desired_goal"][0:3]
            goals.append(goal)

            current_position = observation["observation"][0:3]
            current_positions.append(current_position)

            dis = np.linalg.norm(goal - current_position)
            dis_all.append(dis)

            goal_cat = np.stack(goals, axis=0)
            current_position_cat = np.stack(current_positions, axis=0)

            print(
                f"init pos range: {np.min(current_position_cat, axis=0)} - {np.max(current_position_cat, axis=0)}\n"
                f"goal range: {np.min(goal_cat, axis=0)} - {np.max(goal_cat, axis=0)}\n"
                f"dis={np.mean(dis_all):.3f}, max_dis={np.max(dis_all):.3f}, min_dis={np.min(dis_all):.3f}, "
                f"mean_reward={mean_reward:.3f}, std_reward={std_reward:.3f}"
            )
            reward_totals = reward_totals[-2000:]

            g_t = 0
            reward_total = 0

            if len(dis_all) > ep:
                break
    if plot:
        plt.hist(dis_all, bins=30)
        plt.show(block=True)


def update_canvas(ax, loc_coords, loc_labels, info, reward=0, act_k=None):
    ax.cla()

    ax.scatter(loc_coords[:, 0], loc_coords[:, 1], color='lightgrey',s=100)
    for i, txt in enumerate(loc_labels):
        ax.annotate(txt, (loc_coords[i, 0], loc_coords[i, 1]+0.1),
                    ha='center', va='center')
    ax.set_title(f'Reward: {reward}')
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)

    ax.scatter(
        [loc_coords[info['agent'], 0]],
        [loc_coords[info['agent'], 1]],
        color='red',
        s=200,
        alpha=0.8,
        zorder=2000
    )
    ax.scatter(
        [loc_coords[info['akc'][1], 0]],
        [loc_coords[info['akc'][1], 1]],
        color='orange',
        marker='P',
        s=400,
        alpha=0.5,
        zorder=1001
    )
    ax.scatter(
        [loc_coords[info['akc'][2], 0]],
        [loc_coords[info['akc'][2], 1]],
        color='green',
        marker='D',
        s=400,
        alpha=0.5,
        zorder=1000
    )

    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()
    pass


@cli.command
@click.option("--render", type=str, default="vec")
@click.option("--T", type=int, default=3)
@click.option("--ep", type=int, default=400)
@click.option("--plot", type=bool, default=False)
def treasurehunt(render, t, ep, plot):
    """Robot arm reach task"""
    from tasks.treasure_hunting.treasurehunting import TreasureHuntEnv
    import matplotlib.pyplot as plt

    env_name = "tasks.TreasureHunt"
    assert render in ("vec", "human")
    render_mode = render
    env: TreasureHuntEnv = gym.make(
        env_name,
        seed=1,
        render_mode=render_mode,
    )

    reward_total = 0
    loc_labels = [
        'Cow Field',
        'Scarecrow',
        'Axe Stump',
        'Farm House',
    ]
    loc_coords = np.array([
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 0],
    ])

    observation, info = env.reset()

    rkd = {'down': 'S', 'up': 'W', 'left': 'A', 'right': 'D'}

    def on_press(event):
        nonlocal reward_total
        print('press', event.key)
        # sys.stdout.flush()
        if event.key in rkd.keys():
            act_k = rkd[event.key]  # event.key.capitalize()
            print(f'your ation is {act_k}')
            action = env.dir2act[act_k]
            next_observation, reward, terminated, truncated, info = env.step(
                action)
            reward_total += reward
            observation = next_observation

            update_canvas(ax, loc_coords, loc_labels,
                          info, reward=reward, act_k=act_k)
            if terminated or truncated:
                time.sleep(0.5)
                ax.cla()
                ax.annotate(
                    f'You get {reward_total} point(s)!',
                    (0.5, 0.5), textcoords='axes fraction', va='center',
                    ha='center',
                )
                ax.figure.canvas.draw()
                ax.figure.canvas.flush_events()
                time.sleep(1)

                reward_total = 0
                observation, info = env.reset()

                update_canvas(ax, loc_coords, loc_labels, info)

    if render_mode == 'human':
        plt.ion()
        fig, ax = plt.subplots()
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        fig.canvas.mpl_connect('key_press_event', on_press)
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)

        update_canvas(ax, loc_coords, loc_labels, info, )
        plt.show(block=True)


if __name__ == "__main__":
    cli()
