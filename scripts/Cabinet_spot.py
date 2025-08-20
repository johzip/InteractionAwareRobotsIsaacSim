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


# FIX: Make OpenVLA imports completely optional
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
            "NUMPAD_8": [1.0, 0.0, 0.0], "UP": [1.0, 0.0, 0.0],
            "NUMPAD_2": [-1.0, 0.0, 0.0], "DOWN": [-1.0, 0.0, 0.0],
            "NUMPAD_6": [0.0, -1.0, 0.0], "RIGHT": [0.0, -1.0, 0.0],
            "NUMPAD_4": [0.0, 1.0, 0.0], "LEFT": [0.0, 1.0, 0.0],
            "NUMPAD_7": [0.0, 0.0, 1.0], "N": [0.0, 0.0, 1.0],
            "NUMPAD_9": [0.0, 0.0, -1.0], "M": [0.0, 0.0, -1.0],
        }

        # Arm control parameters - based on your joint indices
        self.arm_joint_indices = [0, 1, 2, 7, 12, 17, 18]  # From your output
        self.arm_joint_names = ['arm0_sh1', 'arm0_sh0', 'arm0_el0', 'arm0_el1', 'arm0_wr0', 'arm0_wr1', 'arm0_f1x']
        
        self._arm_command = np.zeros(7)  # 7 DOF arm
        self._arm_keyboard_mapping = {
            "Q": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # arm0_sh1 (index 0)
            "A": [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "W": [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # arm0_sh0 (index 1)
            "S": [0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            "E": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],  # arm0_el0 (index 2)
            "D": [0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0],
            "R": [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],  # arm0_el1 (index 7)
            "F": [0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0],
            "T": [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],  # arm0_wr0 (index 12)
            "G": [0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0],
            "Y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],  # arm0_wr1 (index 17)
            "H": [0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0],
            "U": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],  # arm0_f1x (index 18)
            "J": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1],
        }
        
        # Arm control state
        self.manual_arm_mode = False
        self.arm_target_positions = None

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


        # FIX: Only try to load OpenVLA if dependencies are available
        if OPENVLA_AVAILABLE:
            try:
                print("Loading OpenVLA processor...")
                self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
                
                print("Loading OpenVLA model (avoiding flash attention)...")
                # FIX: Force disable flash attention and use standard attention
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
            self._setup_arm_control()
            self.first_step = False
            print("Spot robot initialized and ready to move")
        elif self.needs_spot_reset:
            print("Resetting Spot...")
            self._world.reset(True)
            self.needs_spot_reset = False
            self.first_step = True
        else:
            # Determine if we're in manual arm control mode
            self.manual_arm_mode = np.any(self._arm_command != 0)
            
            if self.manual_arm_mode:
                # Update arm targets based on keyboard input
                self._update_arm_targets()
                
                # Forward with manual arm control
                if np.any(self._base_command != 0):
                    print(f"Spot base command: {self._base_command}")
                    
                print(f"Spot movement Command params \n armPos: {self.arm_target_positions}  \n baseCommand: {self._base_command}")

                self._spot.forward(
                    step_size, 
                    self._base_command,
                    manual_arm_control=True,
                    arm_targets=self.arm_target_positions
                )
            else:
                # Normal policy control for everything
                if np.any(self._base_command != 0):
                    print(f"Spot command (full policy): {self._base_command}")
                self._spot.forward(step_size, self._base_command)

        # Handle cabinet physics
        self._handle_cabinet_physics(step_size)

    def _update_arm_targets(self):
        """Update arm target positions based on keyboard input"""
        if self.arm_target_positions is None:
            return
            
        arm_movement = self._get_arm_movement_command()
        if arm_movement is not None:
            print(f"Updating arm targets: {arm_movement}")
            
            # Update target positions
            for i, movement in enumerate(arm_movement):
                if i < len(self.arm_target_positions):
                    self.arm_target_positions[i] += movement
                    
            # Apply joint limits
            self.arm_target_positions = np.clip(self.arm_target_positions, -3.14, 3.14)
            print(f"New arm targets: {self.arm_target_positions}")

    def _override_policy_arm_actions(self):
        """Override policy arm actions with manual commands"""
        if not self.arm_joint_names or not hasattr(self, 'arm_target_positions'):
            return
            
        try:
            # Check if we have active arm commands
            if not (hasattr(self, '_arm_command') and np.any(self._arm_command != 0)):
                return  # No active arm commands, let policy control
            
            # Get current positions and override arm joints
            current_positions = self._spot.robot.get_joint_positions()
            
            # Create new action with manual arm positions
            full_action = current_positions.copy()
            for i, target_pos in zip(self.arm_joint_indices, self.arm_target_positions):
                full_action[i] = target_pos
            
            # Apply immediately
            from isaacsim.core.utils.types import ArticulationAction
            action = ArticulationAction(joint_positions=np.array(full_action))
            self._spot.robot.apply_action(action)
            
            print("Overrode policy arm actions with manual control")
            
        except Exception as e:
            print(f"Error overriding arm actions: {e}")

    def _setup_arm_control(self):
        """Setup arm control with known joint indices"""
        try:
            all_joint_names = self._spot.robot.dof_names
            print(f"All robot joints: {all_joint_names}")
            print(f"Pre-defined arm joint indices: {self.arm_joint_indices}")
            print(f"Pre-defined arm joint names: {self.arm_joint_names}")
            
            # Initialize arm target positions from current positions
            current_positions = self._spot.robot.get_joint_positions()
            print(f"Current joint positions: {current_positions}")
            
            # Extract initial arm positions
            self.arm_target_positions = current_positions[self.arm_joint_indices].copy()
            self.arm_initial_positions = self.arm_target_positions.copy()
            
            print(f"✅ Arm control setup complete with {len(self.arm_joint_names)} joints.")
            print(f"Initial arm positions: {self.arm_target_positions}")
            
        except Exception as e:
            print(f"❌ Error setting up arm control: {e}")
            self.arm_joint_names = []
            self.arm_joint_indices = []
            self.arm_target_positions = None


    def _handle_arm_control(self):
        """Handle direct arm motor control"""
        if not self.arm_joint_names:
            return
            
        try:
            # Get current arm positions
            current_positions = self._spot.robot.get_joint_positions()
            
            # Get arm movement commands
            arm_movement = self._get_arm_movement_command()
            
            if arm_movement is not None:
                print(f"Applying arm movement: {arm_movement}")
                # FIX: Apply movement to specific arm joints
                for i, movement in enumerate(arm_movement):
                    if i < len(self.arm_target_positions):
                        self.arm_target_positions[i] += movement
                        
                # Apply joint limits
                self.arm_target_positions = np.clip(self.arm_target_positions, -3.14, 3.14)
                
                # FIX: Apply only to arm joints, not all joints
                current_arm_positions = [current_positions[i] for i in self.arm_joint_indices]
                
                # Create action for arm joints only
                full_action = current_positions.copy()  # Start with current positions
                for i, target_pos in zip(self.arm_joint_indices, self.arm_target_positions):
                    full_action[i] = target_pos
                
                # Apply the action
                from isaacsim.core.utils.types import ArticulationAction
                action = ArticulationAction(joint_positions=np.array(full_action))
                self._spot.robot.apply_action(action)
                
                print(f"Updated arm target positions: {self.arm_target_positions}")
                
        except Exception as e:
            print(f"Error in arm control: {e}")

    def _get_arm_movement_command(self):
        """Get arm movement commands from keyboard"""
        if hasattr(self, '_arm_command') and np.any(self._arm_command != 0):
            return self._arm_command * 0.01  # Scale for smooth movement
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
        print("Arm controls (individual joints):")
        print("  Q/A: arm0_sh1, W/S: arm0_sh0, E/D: arm0_el0")
        print("  R/F: arm0_el1, T/G: arm0_wr0, Y/H: arm0_wr1, U/J: arm0_f1x")
        print("  P: Reset arm to initial position")
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
                            spot_images_dir = Path("/home/zipfelj/data/zipfel/spot_images/")
                            spot_images_dir.mkdir(parents=True, exist_ok=True)
                            
                            right_image_path = spot_images_dir / f"right_{self.IDcounter:04d}.png"
                            left_image_path = spot_images_dir / f"left_{self.IDcounter:04d}.png"
                            
                            if right_image.shape[2] == 4:  # RGBA
                                right_image = cv2.cvtColor(right_image, cv2.COLOR_RGBA2RGB)
                            if left_image.shape[2] == 4:  # RGBA
                                left_image = cv2.cvtColor(left_image, cv2.COLOR_RGBA2RGB)
                            
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
            
            # Reset arm to initial position with 'P' key
            if event.input.name == "P":
                if self.arm_target_positions is not None:
                    self.arm_target_positions = self.arm_initial_positions.copy()
                    print(f"Reset arm to initial position: {self.arm_target_positions}")
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