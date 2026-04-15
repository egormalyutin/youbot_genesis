from os.path import dirname, join
from typing import Any, cast

import genesis as gs
import torch

gs.init(backend=gs.gpu)

DIR = dirname(__file__)

YOUBOT_DESCRIPTION = join(DIR, "../subprojects/youbot_description")

JOINTS = [
    "wheel_joint_fl",
    "wheel_joint_fr",
    "wheel_joint_bl",
    "wheel_joint_br",
    "arm_joint_1",
    "arm_joint_2",
    "arm_joint_3",
    "arm_joint_4",
    "arm_joint_5",
    "gripper_finger_joint_l",
    "gripper_finger_joint_r",
]

joint2id = {k: i for i, k in enumerate(JOINTS)}

LINKS = [
    "arm_link_5",
]

link2id = {k: i for i, k in enumerate(LINKS)}

BATCH_RENDERER = False


class VecEnv:
    def __init__(self, n_envs: int):
        self.scene = gs.Scene(
            show_viewer=False,
            viewer_options=gs.options.ViewerOptions(max_FPS=120),
            vis_options=gs.options.VisOptions(env_separate_rigid=True),
        )
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        self.bot = self.scene.add_entity(
            gs.morphs.URDF(
                file=join(YOUBOT_DESCRIPTION, "robots/youbot.urdf.xacro"),
                pos=(0, 0, 0.2),
            ),
        )

        self.box = self.scene.add_entity(
            gs.morphs.Box(pos=(0.5, 0, 0), size=(0.1, 0.1, 0.1))
        )

        self.bot_dofs_idx = []
        for name in JOINTS:
            dofs = self.bot.get_joint(name).dofs_idx_local
            self.bot_dofs_idx.append(dofs[0])

        self.bot_links_idx = []
        for name in LINKS:
            link = self.bot.get_link(name)
            self.bot_links_idx.append(link.idx_local)

        if BATCH_RENDERER:
            renderer_opts = gs.sensors.BatchRendererCameraOptions(
                res=(640, 480),
                pos=(0.1, 0.0, 0.05),  # Offset from link frame
                lookat=(0.2, 0.0, 0.0),  # Look direction
                entity_idx=self.bot.idx,  # Attach to robot
                link_idx_local=self.bot_links_idx[0],  # End-effector link
                lights=[
                    {
                        "pos": (2.0, 2.0, 5.0),
                        "color": (1.0, 1.0, 1.0),
                        "intensity": 1.0,
                        "directional": True,
                        "castshadow": True,
                    }
                ],
            )

        else:
            renderer_opts = gs.sensors.RasterizerCameraOptions(
                res=(640, 480),
                pos=(0.1, 0.0, 0.05),  # Offset from link frame
                lookat=(0.2, 0.0, 0.0),  # Look direction
                entity_idx=self.bot.idx,  # Attach to robot
                link_idx_local=self.bot_links_idx[0],  # End-effector link
                # use_rasterizer=True,
            )

        self.camera = self.scene.add_sensor(cast(Any, renderer_opts))

        self.scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    def step(self):
        self.scene.step()

        bots_links_pos = self.bot.get_links_pos(self.bot_links_idx)
        boxes_pos = self.box.get_links_pos()[:, 0]

        hands_pos = bots_links_pos[:, link2id["arm_link_5"]]

        dists = torch.sum((hands_pos - boxes_pos) ** 2, -1) ** 0.5
        images = self.camera.read()

        print(dists)


env = VecEnv(8)

for i in range(1000):
    env.step()
