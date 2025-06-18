# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to add and simulate on-board sensors for a robot.

We add the following sensors on the quadruped robot, ANYmal-C (ANYbotics):

* USD-Camera: This is a camera sensor that is attached to the robot's base.
* Height Scanner: This is a height scanner sensor that is attached to the robot's base.
* Contact Sensor: This is a contact sensor that is attached to the robot's feet.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/add_sensors_on_robot.py --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from pxr import Usd, Vt
import omni


import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

##
# Pre-defined configs
##
#from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip

# spot robot config 
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip
# spot with arm robot config
#from isaaclab_assets.robots.spot import SPOT_ARM_CFG  # isort: skip

def design_scene():
    """Spawn each object at a random position with its specific scale."""

    # List of (USD asset path, scale) tuples
    usd_objects = [
        (f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/011_banana.usd", (1.0, 1.0, 1.0)),  # meters
        (f"{ISAAC_NUCLEUS_DIR}/Props/Mugs/SM_Mug_C1.usd", (0.02, 0.02, 0.02)),              # meters
        (f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/props/SM_Crate_A07_Yellow_01/SM_Crate_A07_Yellow_01.usd", (0.001, 0.01, 0.005)),  # centimeters
        (f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/props/sm_whitecorrugatedbox_b/sm_whitecorrugatedbox_b19_brown_01.usd", (0.001, 0.001, 0.001)),  # centimeters
        (f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01_physics.usd", (0.001, 0.001, 0.01)),  # centimeters
        (f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/037_scissors.usd", (1.0, 1.0, 1.0)),  # meters
        (f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd", (0.01, 0.01, 0.01)),  # centimeters
    ]

    for i, (usd_path, scale) in enumerate(usd_objects):
        x = random.uniform(-1.5, 1.5)
        y = random.uniform(-1.5, 1.5)
        z = 1.05  # Adjust as needed for your scene

        prim_path = f"/World/Objects/Object_{i}"

        cfg = sim_utils.UsdFileCfg(usd_path=usd_path, scale=scale)
        cfg.func(prim_path, cfg, translation=(x, y, z))

def import_scene():
    usd_path = os.path.abspath("assets/Articulate3D_samples/0a76e06478.usda")

    omni.usd.get_context().open_stage(usd_path)

@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # objects
    # design_scene()
    # import_scene
    import_scene()    

    # robot
    robot: ArticulationCfg = SPOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    #with arm
    #robot: ArticulationCfg = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    #height_scanner = RayCasterCfg(
    #    prim_path="{ENV_REGEX_NS}/Robot/base",
    #    update_period=0.02,
    #    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #    attach_yaw_only=True,
    #    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #    debug_vis=True,
    #    mesh_prim_paths=["/World/defaultGroundPlane"],
    #)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot", update_period=0.0, history_length=6, debug_vis=True
    )



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Set the output directory for the camera
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] = torch.tensor([3.7, 2.0, 0.7])
            # Set orientation: 180° around Z axis (quaternion [x, y, z, w])
            root_state[:, 3:7] = torch.tensor([1.2, 0.0, 0.0, 0.0])
    
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        print("-------------------------------")
        print(scene["camera"])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")
        #print(scene["height_scanner"])
        #print("Received max height value: ", torch.max(scene["height_scanner"].data.ray_hits_w[..., -1]).item())
        print("-------------------------------")
        print(scene["contact_forces"])
        print("Received max contact force of: ", torch.max(scene["contact_forces"].data.net_forces_w).item())

        #save every 10th image (for visualization purposes only)
        # note: saving images will slow down the simulation
        if count % 10 == 0:
            rgb_image = scene["camera"].data.output["rgb"][0, ..., :3]  # (H, W, 3)
            # Create output directory for images if it doesn't exist
            rgb_dir = os.path.join(output_dir, "rgb")
            os.makedirs(rgb_dir, exist_ok=True)
            save_images_grid(
                [rgb_image],
                subtitles=["Camera"],
                title=f"RGB Image: Step {count}",
                filename=os.path.join(rgb_dir, f"{count:04d}.jpg"),
            )

def save_images_grid(
    images: list[torch.Tensor],
    cmap: str | None = None,
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save images in a grid with optional subtitles and title.

    Args:
        images: A list of images to be plotted. Shape of each image should be (H, W, C).
        cmap: Colormap to be used for plotting. Defaults to None, in which case the default colormap is used.
        nrows: Number of rows in the grid. Defaults to 1.
        subtitles: A list of subtitles for each image. Defaults to None, in which case no subtitles are shown.
        title: Title of the grid. Defaults to None, in which case no title is shown.
        filename: Path to save the figure. Defaults to None, in which case the figure is not saved.
    """
    # show images in a grid
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    # Ensure axes is always a 1D array
    if nrow * ncol == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # plot images
    for idx, (img, ax) in enumerate(zip(images, axes)):
        img = img.detach().cpu().numpy()
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        if subtitles:
            ax.set_title(subtitles[idx])
    # remove extra axes if any
    for ax in axes[n_images:]:
        fig.delaxes(ax)
    # set title
    if title:
        plt.suptitle(title)

    # adjust layout to fit the title
    plt.tight_layout()
    # save the figure
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    # close the figure
    plt.close()


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
