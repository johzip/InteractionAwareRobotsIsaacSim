from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import warnings
# Suppress warnings to clean up console output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    AutoModelForVision2Seq = None
    AutoProcessor = None
    Image = None
    torch = None

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from spot_policy import SpotFlatTerrainPolicy, SpotArmFlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.extensions import enable_extension

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
            "UP": [1.0, 0.0, 0.0],
            "DOWN": [-1.0, 0.0, 0.0],
            "RIGHT": [0.0, -1.0, 0.0],
            "LEFT": [0.0, 1.0, 0.0],
            "N": [0.0, 0.0, 1.0],
            "M": [0.0, 0.0, -1.0],
        }

        # Arm control parameters - based on your joint indices
        self.arm_joint_indices = [0, 1, 2, 7, 12, 17]  # From your output
        self.arm_joint_names = ['arm0_sh1', 'arm0_sh0', 'arm0_el0', 'arm0_el1', 'arm0_wr0', 'arm0_wr1']
        
        self._arm_command = np.zeros(6)  # 6 DOF arm + the buggy arm0_f1x
        self._arm_keyboard_mapping = {
            "Z": [1, 0.0, 0.0, 0.0, 0.0, 0.0],   # arm0_sh1 (index 0) - positive
            "H": [-1, 0.0, 0.0, 0.0, 0.0, 0.0],  # arm0_sh1 (index 0) - negative
            "U": [0.0, 1, 0.0, 0.0, 0.0, 0.0],   # arm0_sh0 (index 1) - positive
            "J": [0.0, -1, 0.0, 0.0, 0.0, 0.0],  # arm0_sh0 (index 1) - negative
            "I": [0.0, 0.0, 1, 0.0, 0.0, 0.0],   # arm0_el0 (index 2) - positive
            "K": [0.0, 0.0, -1, 0.0, 0.0, 0.0],  # arm0_el0 (index 2) - negative
            "O": [0.0, 0.0, 0.0, 1, 0.0, 0.0],   # arm0_el1 (index 7) - positive
            "L": [0.0, 0.0, 0.0, -1, 0.0, 0.0],  # arm0_el1 (index 7) - negative
            "NUMPAD_7": [0.0, 0.0, 0.0, 0.0, 1, 0.0],   # arm0_wr0 (index 12) - positive
            "NUMPAD_4": [0.0, 0.0, 0.0, 0.0, -1, 0.0],
            "NUMPAD_8": [0.0, 0.0, 0.0, 0.0, 0.0, 1],  # arm0_wr1 (index 17)
            "NUMPAD_5": [0.0, 0.0, 0.0, 0.0, 0.0, -1],
           # "NUMPAD_9": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1],  # arm0_f1x (index 18)
           # "NUMPAD_6": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1],
        }
        
        # Simplified arm control state - no more target positions!
        self.manual_arm_mode = False
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
        
        # OpenVLA parameters
        self.openvla_ready = False
        self.processor = None
        self.vla = None

    def _setup_cabinet_scene(self):
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
        cabinet_orientation = np.array([[0.0673854, 0, 0, -0.997727]])  # w, x, y, z
        self._cabinet.set_world_poses(positions=cabinet_position, orientations=cabinet_orientation)
        self._cabinet.set_joint_positions({"drawer_joint": 0.0})

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

        print(f"Using pre-defined arm joint indices: {self.arm_joint_indices}")
        print(f"Using pre-defined arm joint names: {self.arm_joint_names}")

        
        if OPENVLA_AVAILABLE:
            try:
                print("Loading OpenVLA processor...")
                self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
                
                print("Loading OpenVLA model (avoiding flash attention)...")
                model_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                    # Explicitly disable flash attention
                    "attn_implementation": "eager",  # Use standard PyTorch attention
                }
                
                self.vla = AutoModelForVision2Seq.from_pretrained(
                    "openvla/openvla-7b", 
                    **model_kwargs
                ).to("cuda:0")
                
                print("✅ OpenVLA loaded successfully with standard attention")
                print("Press 'O' to toggle AI mode")
                self.openvla_ready = True
                
            except Exception as e:
                print(f"❌ Failed to load OpenVLA: {e}")
                print("🎮 Falling back to manual control only")
                self.processor = None
                self.vla = None
                self.openvla_ready = False
        else:
            print("🎮 Manual control mode - OpenVLA dependencies not available")
            self.processor = None
            self.vla = None
            self.openvla_ready = False

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
            # Check if arm keys are currently pressed
            self.manual_arm_mode = np.any(self._arm_command != 0)
            
            # Get current arm movement command
            arm_movement = self._get_arm_movement_command()
            print(f"manual_arm_mode: {self.manual_arm_mode}; Arm movement command: {arm_movement}")
            
            # FIX: ALWAYS use manual arm control mode to prevent arm reset
            if self.manual_arm_mode and arm_movement is not None:
                # Active arm movement
                if np.any(self._base_command != 0):
                    print(f"Spot base command: {self._base_command}")
                    
                self._spot.forward(
                    step_size, 
                    self._base_command,
                    manual_arm_control=True,
                    arm_changes=arm_movement
                )
            else:
                # NO arm movement, but still use manual control to prevent reset
                if np.any(self._base_command != 0):
                    print(f"Spot command (full policy): {self._base_command}")
                    
                # FIX: Pass manual_arm_control=True with zero changes to maintain position
                self._spot.forward(
                    step_size, 
                    self._base_command,
                    manual_arm_control=True,  # ← ALWAYS TRUE to prevent policy control
                    arm_changes=np.zeros(6)   # ← Zero changes = maintain current position
                )
        # Handle cabinet physics
        self._handle_cabinet_physics(step_size)

    def _get_arm_movement_command(self):
        """Get arm movement commands from keyboard - simplified version"""
        if hasattr(self, '_arm_command') and np.any(self._arm_command != 0):
            return self._arm_command  # Return raw command without scaling
        return None

    def _handle_cabinet_physics(self, step_size):
        """Handle cabinet physics and interaction"""
        if not self._cabinet or not self._cabinet._is_initialized:
            return
            
        self.episode_length_buf += 1
        
        if self._check_cabinet_termination() or self.episode_length_buf >= self.max_episode_length:
            self.needs_cabinet_reset = True
            return

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
        print("Starting simulation")
        print("Base controls: Arrow keys or numpad to move Spot")
        print("Arm controls (individual joints - direct movement):")
        print("  Q/A: arm0_sh1, W/S: arm0_sh0, E/D: arm0_el0")
        print("  R/F: arm0_el1, T/G: arm0_wr0, Y/H: arm0_wr1, U/J: arm0_f1x")
        print("💡 Arm moves while you hold keys, stops when you release!")
        print("💡 No position memory - direct real-time control")
        print("Cabinet drawer will reset automatically when opened too far")
        if self.openvla_ready:
            print("🤖 Press 'O' to toggle AI control mode")
        else:
            print("🎮 Manual control only")
        
        cameras_ready = False

        while simulation_app.is_running():
            # Handle cabinet environment resets
            if self.needs_cabinet_reset:
                self._reset_cabinet_environment()
            
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_spot_reset = True

            # Initialize cameras after a few steps
            if not cameras_ready and self._world.current_time_step_index > 5:
                try:
                    if self.cameraRight and self.cameraLeft:
                        self.cameraRight.initialize()
                        self.cameraLeft.initialize()
                        cameras_ready = True
                        print("Cameras are now ready for image capture")
                except Exception as e:
                    print(f"Camera initialization error: {e}")

            # Capture images from cameras
            if cameras_ready and self.cameraRight and self.cameraLeft:
                current_time = time.time()
                if current_time - self.last_capture_time >= 1.0 / self.picfreq:
                    try:
                        self.last_capture_time = current_time
                        right_image = self.cameraRight.get_rgb()
                        left_image = self.cameraLeft.get_rgb()
                        
                        if right_image is not None and left_image is not None:
                            spot_images_dir = Path("/home/zipfelj/workspace/IsaacRobotics/visualData")
                            spot_images_dir.mkdir(parents=True, exist_ok=True)
                            
                            right_image_path = spot_images_dir / f"right_{self.IDcounter:04d}.png"
                            left_image_path = spot_images_dir / f"left_{self.IDcounter:04d}.png"
                            
                            if right_image.shape[2] == 4:  # RGBA
                                right_image = cv2.cvtColor(right_image, cv2.COLOR_RGBA2RGB)
                            if left_image.shape[2] == 4:  # RGBA
                                left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
                            
                            cv2.imwrite(str(right_image_path), cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(str(left_image_path), cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))
                            
                            print(f"Captured images: {right_image_path}, {left_image_path}")
                            self.IDcounter += 1
                        else:
                            print("Warning: Could not capture images - cameras returned None")
                            
                    except Exception as e:
                        print(f"Error capturing images: {e}")

        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Toggle AI mode with 'O' key
            if event.input.name == "O":
                if self.openvla_ready:
                    ai_mode = not getattr(self, '_ai_mode', False)
                    self._ai_mode = ai_mode
                    mode = "AI" if ai_mode else "Manual"
                    print(f"Switched to {mode} control mode")
                else:
                    print("OpenVLA not available")
                return True
            
            # Handle base movement
            if event.input.name in self._input_keyboard_mapping:
                print(f"Base key pressed: {event.input.name}")
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
                print(f"New base command: {self._base_command}")
                
            # Handle arm movement
            if event.input.name in self._arm_keyboard_mapping:
                joint_movement = np.array(self._arm_keyboard_mapping[event.input.name])
                print(f"Arm key pressed: {event.input.name}")
                print(f"  Joint movement: {joint_movement}")
                self._arm_command += joint_movement
                print(f"  New arm command: {self._arm_command}")
                
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                print(f"Base key released: {event.input.name}")
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
                print(f"New base command: {self._base_command}")
                
            # Handle arm key release
            if event.input.name in self._arm_keyboard_mapping:
                joint_movement = np.array(self._arm_keyboard_mapping[event.input.name])
                print(f"Arm key released: {event.input.name}")
                self._arm_command -= joint_movement
                print(f"  New arm command: {self._arm_command}")
                
        return True
    
    def get_cabinet(self):
        return self._cabinet
    
    def cleanup(self):
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
    
    runner.run()
    
    runner.cleanup()
    simulation_app.close()

if __name__ == "__main__":
    main()