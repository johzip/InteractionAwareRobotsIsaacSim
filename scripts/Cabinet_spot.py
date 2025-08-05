from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import os
from pathlib import Path
import omni.appwindow
import cv2
import time

from pxr import UsdGeom, UsdPhysics, Sdf

from isaacsim.core.utils.prims import create_prim, define_prim
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from spot_policy import SpotFlatTerrainPolicy, SpotArmFlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.extensions import enable_extension

# Comment out ROS2 bridge to avoid startup issues
# enable_extension('isaacsim.ros2.bridge')


class SpotCabinetRunner(object):
    def __init__(self, physics_dt, render_dt) -> None:
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        BASE_DIR = Path(__file__).resolve().parent.parent
        
        # Cabinet environment parameters
        self.episode_length_s = 8.3333
        self.decimation = 2
        self.num_envs = 1
        self.device = "cpu"
        self.needs_cabinet_reset = False
        self.episode_length_buf = 0
        self.max_episode_length = int(self.episode_length_s / (physics_dt * self.decimation))
        
        # Setup cabinet scene
        self._setup_cabinet_scene()
        
        # Setup Spot robot parameters
        self.policy_path = os.path.join(BASE_DIR, "Assets/spot_robots/policies/spot_arm/models", "spot_arm_policy.pt")
        self.policy_params_path = os.path.join(BASE_DIR, "Assets/spot_robots/policies/spot_arm/params", "env.yaml")
        self.usd_path = os.path.join(BASE_DIR, "Assets/spot_robots", "spot_arm.usd")
        self.spot_position = np.array([1.0, -1.0, 0.8])  # Position away from cabinet
        
        self._spot = None
        self._cabinet = None

        # Spot control parameters
        self._base_command = np.zeros(3)
        self._input_keyboard_mapping = {
            "NUMPAD_8": [1.0, 0.0, 0.0], "UP": [1.0, 0.0, 0.0],
            "NUMPAD_2": [-1.0, 0.0, 0.0], "DOWN": [-1.0, 0.0, 0.0],
            "NUMPAD_6": [0.0, -1.0, 0.0], "RIGHT": [0.0, -1.0, 0.0],
            "NUMPAD_4": [0.0, 1.0, 0.0], "LEFT": [0.0, 1.0, 0.0],
            "NUMPAD_7": [0.0, 0.0, 1.0], "N": [0.0, 0.0, 1.0],
            "NUMPAD_9": [0.0, 0.0, -1.0], "M": [0.0, 0.0, -1.0],
        }

        self.needs_spot_reset = False
        self.first_step = True

        # Camera setup
        self.camera_prim_pathRight = "/World/Spot/body/frontright_fisheye"
        self.camera_prim_pathLeft = "/World/Spot/body/frontleft_fisheye" 
        self.cameraRight = None
        self.cameraLeft = None
        
        self.picfreq = 2
        self.IDcounter = 0
        self.last_capture_time = time.time()
        self.output_dir = Path(os.path.join(BASE_DIR, "output/", str(int(time.time()))))

    def _setup_cabinet_scene(self):
        """Setup the cabinet scene"""
        # Add default ground plane
        self._world.scene.add_default_ground_plane()

        # Load cabinet USD
        cabinet_usd = "/home/zipfelj/data/zipfel/Articulate_3D/full_scene_sim_ready/model_scene_video.usda"
        add_reference_to_stage(usd_path=cabinet_usd, prim_path="/World/Cabinet")
        
        # Create cabinet articulation
        self._cabinet = Articulation(prim_paths_expr="/World/Cabinet", name="cabinet")
        self._world.scene.add(self._cabinet)

        # Set cabinet position and orientation
        cabinet_position = np.array([[0.0, 0.0, 0.39146906]])
        cabinet_orientation = np.array([[0.0673854, 0, 0, -0.997727]])  # w, x, y, z
        self._cabinet.set_world_poses(positions=cabinet_position, orientations=cabinet_orientation)
        self._cabinet.set_joint_positions({"drawer_joint": 0.0})

        # Add collision meshes to cabinet (optional - comment out if causing issues)
        # self._add_collision_to_cabinet()
        
    def _add_collision_to_cabinet(self):
        """Add collision to cabinet meshes"""
        stage = self._world.stage

        def add_triangle_mesh_colliders(prim):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                attr = prim.GetAttribute("collision:approximation")
                if not attr.IsValid():
                    attr = prim.CreateAttribute("collision:approximation", Sdf.ValueTypeNames.Token)
                attr.Set("triangleMesh")
            
            for child in prim.GetChildren():
                add_triangle_mesh_colliders(child)
        
        cabinet_prim = stage.GetPrimAtPath("/World/Cabinet")
        if cabinet_prim:
            add_triangle_mesh_colliders(cabinet_prim)

    def setup(self) -> None:
        # Reset world first
        self._world.reset()
        
        # Initialize cabinet
        if self._cabinet:
            self._cabinet.initialize()
        
        # Create Spot robot
        print("Creating Spot robot...")
        self._spot = SpotArmFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            usd_path=self.usd_path,
            policy_path=self.policy_path,
            policy_params_path=self.policy_params_path,
            position=self.spot_position,
        )
        print("Spot robot created")

        # Initialize cameras
        try:
            self.cameraRight = Camera(self.camera_prim_pathRight)
            self.cameraLeft = Camera(self.camera_prim_pathLeft)
            print("Cameras initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize cameras: {e}")
            self.cameraRight = None
            self.cameraLeft = None
        
        # Setup input handling
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._sub_keyboard_event
        )

        # Add unified physics callback
        self._world.add_physics_callback("spot_cabinet_forward", callback_fn=self.on_physics_step)

    def on_physics_step(self, step_size) -> None:
        # Handle Spot robot
        if self._spot is None:
            return
            
        if self.first_step:
            print("Initializing Spot robot...")
            self._spot.initialize()
            self.first_step = False
            print("Spot robot initialized and ready to move")
        elif self.needs_spot_reset:
            print("Resetting Spot...")
            self._world.reset(True)
            self.needs_spot_reset = False
            self.first_step = True
        else:
            # Handle Spot movement
            if np.any(self._base_command != 0):
                print(f"Spot command: {self._base_command}")
            self._spot.forward(step_size, self._base_command)

        # Handle cabinet physics (optional)
        self._handle_cabinet_physics(step_size)

    def _handle_cabinet_physics(self, step_size):
        """Handle cabinet physics and interaction"""
        if not self._cabinet or not self._cabinet._is_initialized:
            return
            
        # Update episode counter
        self.episode_length_buf += 1
        
        # Check for cabinet reset conditions
        if self._check_cabinet_termination() or self.episode_length_buf >= self.max_episode_length:
            self.needs_cabinet_reset = True
            return
        
        # Print cabinet debug info occasionally
        if self.episode_length_buf % 120 == 0:  # Every 2 seconds
            try:
                cabinet_pos = self._cabinet.get_joint_positions()
                if cabinet_pos is not None and len(cabinet_pos) > 0:
                    if isinstance(cabinet_pos, np.ndarray):
                        drawer_opening = cabinet_pos[0].item() if len(cabinet_pos) > 0 else 0.0
                    else:
                        drawer_opening = cabinet_pos.get("drawer_joint", 0.0)
                    print(f"Cabinet episode step: {self.episode_length_buf}, Drawer opening: {drawer_opening:.4f}")
            except:
                pass

    def _check_cabinet_termination(self):
        """Check if cabinet episode should terminate"""
        if not self._cabinet:
            return False
            
        try:
            cabinet_pos = self._cabinet.get_joint_positions()
            if cabinet_pos is not None:
                if isinstance(cabinet_pos, np.ndarray):
                    drawer_opening = cabinet_pos[0].item() if len(cabinet_pos) > 0 else 0.0
                else:
                    drawer_opening = cabinet_pos.get("drawer_joint", 0.0)
                return drawer_opening > 0.39
        except:
            pass
        return False
    
    def _reset_cabinet_environment(self):
        """Reset the cabinet environment"""
        print(f"Resetting cabinet environment after {self.episode_length_buf} steps")
        
        self.episode_length_buf = 0
        
        # Reset cabinet joint positions
        if self._cabinet:
            try:
                self._cabinet.set_joint_positions({"drawer_joint": 0.0})
            except:
                try:
                    self._cabinet.set_joint_positions(np.array([0.0]))
                except:
                    pass
        
        self.needs_cabinet_reset = False

    def run(self) -> None:
        print("Starting simulation - use arrow keys or numpad to move Spot")
        print("Cabinet drawer will reset automatically when opened too far")
        
        while simulation_app.is_running():
            # Handle cabinet environment resets
            if self.needs_cabinet_reset:
                self._reset_cabinet_environment()
            
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_spot_reset = True
        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._input_keyboard_mapping:
                print(f"Key pressed: {event.input.name}")
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
                print(f"New command: {self._base_command}")
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                print(f"Key released: {event.input.name}")
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
                print(f"New command: {self._base_command}")
        return True

    def get_cabinet(self):
        """Get the cabinet articulation object"""
        return self._cabinet
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self._world.remove_physics_callback("spot_cabinet_forward")
        except:
            pass


def main():
    physics_dt = 1 / 200.0
    render_dt = 1 / 60.0

    runner = SpotCabinetRunner(physics_dt=physics_dt, render_dt=render_dt)
    simulation_app.update()
    
    runner.setup()
    simulation_app.update()
    
    # Run the simulation
    runner.run()
    
    # Cleanup
    runner.cleanup()
    simulation_app.close()


if __name__ == "__main__":
    main()