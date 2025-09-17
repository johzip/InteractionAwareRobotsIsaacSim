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

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from PIL import Image
    import torch
    OPENVLA_AVAILABLE = True
    print("✅ OpenVLA dependencies available")
except Exception as e:
    print(f"⚠️ OpenVLA not available: {e}")
    print("Running in manual control mode only")
    OPENVLA_AVAILABLE = False


from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from spot_policy import SpotFlatTerrainPolicy, SpotArmFlatTerrainPolicy
from isaacsim.core.utils.extensions import enable_extension

class SpotCabinetRunner(object):

### first code block init
    def __init__(self, physics_dt, render_dt) -> None:
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
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
        self._init_cabinet_scene()

        # Setup Spot robot parameters
        self.policy_path = os.path.join(BASE_DIR, "Assets/spot_robots/policies/spot_arm/models", "spot_arm_policy.pt")
        self.policy_params_path = os.path.join(BASE_DIR, "Assets/spot_robots/policies/spot_arm/params", "env.yaml")
        self.usd_path = os.path.join(BASE_DIR, "Assets/spot_robots", "spot_arm.usd")
        self.spot_position = np.array([1.0, -1.0, 0.8])

        self._spot = None
        self._cabinet = None

        self._init_control()

        self.camera_prim_pathRight = "/World/Spot/body/frontright_fisheye"
        self.camera_prim_pathLeft = "/World/Spot/body/frontleft_fisheye" 

        self.first_step = 5

    def _init_control(self):
        """Initialize Spot control parameters"""
        # Spot control parameters
        self._base_command = np.zeros(3)
        self._input_keyboard_mapping = {
            "UP": [1.0, 0.0, 0.0],
            "DOWN": [-1.0, 0.0, 0.0],
            "RIGHT": [0.0, -1.0, 0.0],
            "LEFT": [0.0, 1.0, 0.0],
            "N": [0.0, 0.0, 1.0],
            "M": [0.0, 0.0, -1.0],
        }

        self.arm_joint_indices = []
        self.arm_joint_names = []

        self._arm_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._arm_keyboard_mapping = {
            "NUMPAD_1": [0.09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # arm0_sh1 (index 0) - positive
            "NUMPAD_2": [-0.27, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # arm0_sh1 (index 0) - negative
            "U": [0.0, 0.09, 0.0, 0.0, 0.0, 0.0, 0.0],   # arm0_sh0 (index 1) - positive
            "J": [0.0, -0.09, 0.0, 0.0, 0.0, 0.0, 0.0],  # arm0_sh0 (index 1) - negative
            "I": [0.0, 0.0, 0.09, 0.0, 0.0, 0.0, 0.0],   # arm0_el0 (index 2) - positive
            "K": [0.0, 0.0, -0.09, 0.0, 0.0, 0.0, 0.0],  # arm0_el0 (index 2) - negative
            "P": [0.0, 0.0, 0.0, 0.09, 0.0, 0.0, 0.0],   # arm0_el1 (index 7) - positive
            "L": [0.0, 0.0, 0.0, -0.09, 0.0, 0.0, 0.0],  # arm0_el1 (index 7) - negative
            "NUMPAD_7": [0.0, 0.0, 0.0, 0.0, 0.09, 0.0, 0.0],  # arm0_wr0 (index 12) - positive
            "NUMPAD_4": [0.0, 0.0, 0.0, 0.0, -0.09, 0.0, 0.0], # arm0_wr0 (index 12) - negative
            "NUMPAD_8": [0.0, 0.0, 0.0, 0.0, 0.0, 0.09, 0.0],  # arm0_wr1 (index 17) - positive
            "NUMPAD_5": [0.0, 0.0, 0.0, 0.0, 0.0, -0.09, 0.0], # arm0_wr1 (index 17) - negative
            "NUMPAD_9": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09],  # arm0_f1x (index 18) - positive - gripper open close
            "NUMPAD_6": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.09], # arm0_f1x (index 18) - negative 
        }

        self.keyPressActive = False

    def _init_cabinet_scene(self):
        """Setup the cabinet scene"""
        # Add default ground plane
        self._world.scene.add_default_ground_plane()

        # Load cabinet USD
        cabinet_usd = "/home/zipfelj/workspace/Articulate3D/full_scene_sim_ready/model_scene_video.usda"
        add_reference_to_stage(usd_path=cabinet_usd, prim_path="/World/Cabinet")
        
        # Create cabinet articulation
        self._cabinet = Articulation(prim_paths_expr="/World/Cabinet", name="cabinet")
        self._world.scene.add(self._cabinet)

        # Set cabinet position and orientation
        cabinet_position = np.array([[0.0, 0.0, 0.39146906]])
        cabinet_orientation = np.array([[0.0673854, 0, 0, -0.997727]]) # w, x, y, z
        self._cabinet.set_world_poses(positions=cabinet_position, orientations=cabinet_orientation)
        self._cabinet.set_joint_positions({"drawer_joint": 0.0})


### second code block setup
    def setup(self) -> None:
        self._world.reset()

        if self._cabinet:
            self._cabinet.initialize()

        self._spot = SpotArmFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            usd_path=self.usd_path,
            policy_path=self.policy_path,
            policy_params_path=self.policy_params_path,
            position=self.spot_position,
        )
        print("Spot robot created")

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

        self._world.add_physics_callback("spot_cabinet_forward", callback_fn=self.on_physics_step)

        self.setup_camera()

        self._spot.initialize()

    def setup_camera(self):
        try:
            if self.cameraRight and self.cameraLeft:
                self.cameraRight.initialize()
                self.cameraLeft.initialize()
                self.cameras_ready = True
                print("Cameras are now ready for image capture")
        except Exception as e:
            print(f"Camera initialization error: {e}")

    def findInitArmJoints(self) -> np.ndarray:
        all_joints = self._spot.robot.dof_names
        actual_arm_indices = []
        actual_arm_names = []
        for i, name in enumerate(all_joints):
            if 'arm0_' in name:
                actual_arm_indices.append(i)
                actual_arm_names.append(name)
        
        print(f"ACTUAL ARM JOINT INDICES: {actual_arm_indices}")
        print(f"ACTUAL ARM JOINT NAMES: {actual_arm_names}")
        
        # Update the indices
        self.arm_joint_indices = actual_arm_indices
        arm_joint_names = actual_arm_names

        if self.arm_joint_indices:
            initial_arm_positions = np.array([
                -6.8275726e-01,   # arm0_sh1
                6.4476831e-03,    # arm0_sh0
                1.1585672e+00,    # arm0_el0
                1.0420896e-01,    # arm0_el1
                -4.2314461e-01,   # arm0_wr0
                -1.3542861e-02,   # arm0_wr1
                -1.5841520e-07,   # arm0_f1x
            ])
            
            # Trim to match actual number of discovered joints
            initial_arm_positions = initial_arm_positions[:len(self.arm_joint_indices)]
            return initial_arm_positions
        

### third and cyclic code block
    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.keyPressActive = True
            if event.input.name in self._arm_keyboard_mapping:
                # Only respond to manual controls if not in AI mode
                joint_movement = np.array(self._arm_keyboard_mapping[event.input.name])
                print(f"Arm key pressed: {event.input.name}")
                print(f"  Joint movement: {joint_movement}")
                self._arm_command += joint_movement
                print(f"  New arm command: {self._arm_command}")
                return True
                
            # Handle base movement
            if event.input.name in self._input_keyboard_mapping:
                print(f"Base key pressed: {event.input.name}")
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
                print(f"New base command: {self._base_command}")
                return True
                    
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.keyPressActive = False
            if event.input.name in self._arm_keyboard_mapping:
                joint_movement = np.array(self._arm_keyboard_mapping[event.input.name])
                print(f"Arm key released: {event.input.name}")
                self._arm_command -= joint_movement
                print(f"  New arm command: {self._arm_command}")
                return True
                
            if event.input.name in self._input_keyboard_mapping:
                print(f"Base key released: {event.input.name}")
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
                print(f"New base command: {self._base_command}")
                return True
        return False

    def on_physics_step(self, step_size) -> None:
        if self._spot is None:
            return
        if self.first_step > 0:
            self.first_step = self.first_step - 1
            initial_arm_positions = self.findInitArmJoints()
            
            self._spot.forward(
                step_size, 
                self._base_command,
                manual_arm_control=False,
            )
            
            
        self._handle_manual_control(step_size)

    def _handle_manual_control(self, step_size):
        """Handle manual keyboard control"""
        # Check if arm keys are currently pressed
        self.manual_arm_mode = np.any(self._arm_command != 0)
        
        # Get current arm movement command
        arm_movement = self._get_arm_movement_command()
        
        # Active arm movement
        if np.any(self._base_command != 0):
            print(f"Spot base command: {self._base_command}")

        if self.keyPressActive:   
            self._spot.forward(
                step_size, 
                self._base_command,
                manual_arm_control=True,
                arm_changes=arm_movement
            )
        else:
            zero_changes = np.zeros(len(self.arm_joint_indices))
            self._spot.forward(
                step_size, 
                self._base_command,
                manual_arm_control=True,  
                arm_changes=zero_changes   
            )
    
    def _get_arm_movement_command(self):
        """Get arm movement commands from keyboard - simplified version"""
        if hasattr(self, '_arm_command') and np.any(self._arm_command != 0):
            return self._arm_command  # Return raw command without scaling
        return None

    def run(self) -> None:
        print("Starting simulation")
        print("🎮 Manual Controls:")
        print("  Base: Arrow keys or numpad to move Spot")
        print("  Arm: Q/A, W/S, E/D, R/F, T/G, Y/H, U/J for joints")
        print("💡 Arm moves while you hold keys, stops when you release!")
        while simulation_app.is_running():
            # Handle cabinet environment resets
            if self.needs_cabinet_reset:
                self._reset_cabinet_environment()
            
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_spot_reset = True
        return

### cleanup
    def cleanup(self):
        try:
            self._world.remove_physics_callback("spot_cabinet_forward")
        except:
            pass

### Main
def main():
    physics_dt = 1 / 200.0
    render_dt = 1 / 60.0

    runner = SpotCabinetRunner(physics_dt=physics_dt, render_dt=render_dt)
    simulation_app.update()
    
    runner.setup()
    simulation_app.update()

    runner.run()
    
    runner.cleanup()
    simulation_app.close()

if __name__ == "__main__":
    main()