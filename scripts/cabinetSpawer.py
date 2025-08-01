import numpy as np
import torch
import os
from pathlib import Path
from pxr import UsdGeom, UsdPhysics, Sdf
import carb
import omni

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.stage import add_reference_to_stage
import isaacsim.core.utils.numpy.rotations as rot_utils


class FrankaCabinetEnv:
    def __init__(self, world=None, physics_dt=1/120, render_dt=1/60):
        # Use provided world or create new one
        if world is not None:
            self._world = world
            self._owns_world = False  # Don't close world we don't own
        else:
            self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
            self._owns_world = True
        
        # Environment parameters
        self.episode_length_s = 8.3333
        self.decimation = 2
        self.num_envs = 1
        self.device = "cpu"
        
        self.needs_reset = False
        self._is_setup = False

        # Initialize environment
        self._setup_scene()
        
        # Episode tracking
        self.episode_length_buf = 0
        self.max_episode_length = int(self.episode_length_s / (physics_dt * self.decimation))
        
    def _setup_scene(self):
        """Setup the cabinet scene"""
        # Only add ground plane if we own the world
        if self._owns_world:
            self._world.scene.add_default_ground_plane()

        cabinet_usd = "/home/zipfelj/data/zipfel/Articulate_3D/full_scene_sim_ready/model_scene_video.usda"
        add_reference_to_stage(usd_path=cabinet_usd, prim_path="/World/Cabinet")
        self._cabinet = Articulation(prim_paths_expr="/World/Cabinet", name="cabinet")
        
        self._world.scene.add(self._cabinet)

        cabinet_position = np.array([[0.0, 0.0, 0.39146906]])
        cabinet_orientation = np.array([[0.0673854, 0, 0, -0.997727]])  # w, x, y, z
        self._cabinet.set_world_poses(positions=cabinet_position, orientations=cabinet_orientation)

        self._cabinet.set_joint_positions({"drawer_joint": 0.0})

        # Add triangle mesh colliders
        self._add_collision_to_scene()
        
    def _add_collision_to_scene(self):
        """Add collision to all meshes in the scene"""
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
        
        # Apply to cabinet
        cabinet_prim = stage.GetPrimAtPath("/World/Cabinet")
        
        if cabinet_prim:
            add_triangle_mesh_colliders(cabinet_prim)
    
    def setup(self):
        """Setup physics callback - call this after world.reset()"""
        if self._is_setup:
            return
            
        # Initialize the cabinet
        self._cabinet.initialize()
        
        # Add physics callback
        self._world.add_physics_callback("franka_cabinet_forward", callback_fn=self.on_physics_step)
        
        self._is_setup = True

    def on_physics_step(self, step_size):
        """Physics step callback"""
        
        # Check if first step - ensure everything is initialized
        if self.episode_length_buf == 0:
            if not self._cabinet._is_initialized:
                self._cabinet.initialize()
        
        # Update episode counter
        self.episode_length_buf += 1
        
        # Check for reset conditions
        if self._check_termination() or self.episode_length_buf >= self.max_episode_length:
            self.needs_reset = True
            return
        
        # Print some debug info
        if self.episode_length_buf % 60 == 0:  # Every second
            cabinet_pos = self._cabinet.get_joint_positions()
            if cabinet_pos is not None and len(cabinet_pos) > 0:
                if isinstance(cabinet_pos, np.ndarray):
                    drawer_opening = cabinet_pos[0].item() if len(cabinet_pos) > 0 else 0.0
                else:
                    drawer_opening = cabinet_pos.get("drawer_joint", 0.0)
                print(f"Episode step: {self.episode_length_buf}, Drawer opening: {drawer_opening:.4f}")

    def _check_termination(self):
        """Check if episode should terminate"""
        cabinet_pos = self._cabinet.get_joint_positions()
        if cabinet_pos is not None:
            if isinstance(cabinet_pos, np.ndarray):
                drawer_opening = cabinet_pos[0].item() if len(cabinet_pos) > 0 else 0.0
            else:
                drawer_opening = cabinet_pos.get("drawer_joint", 0.0)
            return drawer_opening > 0.39
        return False
    
    def _reset_environment(self):
        """Reset the cabinet environment"""
        print(f"Resetting cabinet environment after {self.episode_length_buf} steps")
        
        # Reset episode counter
        self.episode_length_buf = 0
        
        # Reset cabinet joint positions
        try:
            self._cabinet.set_joint_positions({"drawer_joint": 0.0})
        except:
            self._cabinet.set_joint_positions(np.array([0.0]))
        
        # Re-initialize cabinet if needed
        if not self._cabinet._is_initialized:
            self._cabinet.initialize()
        
        # Clear reset flag
        self.needs_reset = False

    def get_cabinet(self):
        """Get the cabinet articulation object"""
        return self._cabinet
    
    def get_world(self):
        """Get the world object"""
        return self._world
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, '_cabinet'):
            # Remove physics callback if we added it
            if self._is_setup:
                try:
                    self._world.remove_physics_callback("franka_cabinet_forward")
                except:
                    pass


# Optional: Keep standalone functionality for testing
def main():
    """Standalone main function for testing"""
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})
    
    # Create and run environment
    env = FrankaCabinetEnv(physics_dt=1/120, render_dt=1/60)
    env._world.reset()
    env.setup()
    
    # Run simulation
    while simulation_app.is_running():
        if env.needs_reset:
            env._reset_environment()
        
        env._world.step(render=True)
        if env._world.is_stopped():
            break
    
    # Cleanup
    env.cleanup()
    simulation_app.close()


if __name__ == "__main__":
    main()