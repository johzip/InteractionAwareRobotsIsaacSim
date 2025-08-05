from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import os
from pathlib import Path
import omni.appwindow
import cv2

from pxr import UsdGeom, UsdPhysics, Sdf

from isaacsim.core.utils.prims import create_prim
from omni.isaac.sensor import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import time

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.extensions import enable_extension
from spot_policy import SpotFlatTerrainPolicy, SpotArmFlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path
#from omni.isaac.core.utils.extensions import enable_extension
#enable_extension('isaacsim.ros2.bridge')


class SpotRunner(object):
    def __init__(self, physics_dt, render_dt) -> None:
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
            

        BASE_DIR = Path(__file__).resolve().parent.parent
        self._world.scene.add_default_ground_plane()  # add ground plane

        prim = define_prim("/World/Room", "Xform")
        asset_path = os.path.join(BASE_DIR, "Assets/Scenes/07f5b601ee.usda")
        prim.GetReferences().AddReference(asset_path, "/Room")
        self.add_triangle_mesh_colliders(prim)
        
        policy_path = os.path.join(BASE_DIR, "Assets/spot_robots/policies/spot_arm/models", "spot_arm_policy.pt")
        policy_params_path = os.path.join(BASE_DIR, "Assets/spot_robots/policies/spot_arm/params", "env.yaml")
        usd_path = os.path.join(BASE_DIR, "Assets/spot_robots", "spot_arm.usd")

        self._spot = SpotArmFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            usd_path=usd_path,
            policy_path=policy_path,
            policy_params_path=policy_params_path,
            position=np.array([1, 0, 0.8]),
        )

        self._base_command = np.zeros(3)
        self._input_keyboard_mapping = {
            "NUMPAD_8": [1.0, 0.0, 0.0], "UP": [1.0, 0.0, 0.0],
            "NUMPAD_2": [-1.0, 0.0, 0.0], "DOWN": [-1.0, 0.0, 0.0],
            "NUMPAD_6": [0.0, -1.0, 0.0], "RIGHT": [0.0, -1.0, 0.0],
            "NUMPAD_4": [0.0, 1.0, 0.0], "LEFT": [0.0, 1.0, 0.0],
            "NUMPAD_7": [0.0, 0.0, 1.0], "N": [0.0, 0.0, 1.0],
            "NUMPAD_9": [0.0, 0.0, -1.0], "M": [0.0, 0.0, -1.0],
        }

        self.needs_reset = False
        self.first_step = True

        
        self.camera_prim_pathRight = "/World/Spot/body/frontright_fisheye"
        self.camera_prim_pathLeft = "/World/Spot/body/frontleft_fisheye" 
        
        self.cameraRight = Camera(self.camera_prim_pathRight)
        self.cameraLeft = Camera(self.camera_prim_pathLeft)
    
        self.picfreq = 2  # frequency to take pictures in seconds
        self.IDcounter = 0  # counter for the image ID
        self.last_capture_time = time.time()
        self.output_dir = Path(os.path.join(BASE_DIR, "output/", str(int(time.time()))))

    def add_triangle_mesh_colliders(self, prim):
        # If this prim is a mesh, add a triangle mesh collider
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            attr = prim.GetAttribute("collision:approximation")
            if not attr.IsValid():
                attr = prim.CreateAttribute("collision:approximation", Sdf.ValueTypeNames.Token)
            attr.Set("triangleMesh")
        # Recurse for all children
        for child in prim.GetChildren():
            self.add_triangle_mesh_colliders(child)

    def setup(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._sub_keyboard_event
        )
        self._world.add_physics_callback("spot_forward", callback_fn=self.on_physics_step)

    def on_physics_step(self, step_size) -> None:
        if self.first_step:
            self._spot.initialize()
            self.first_step = False
        elif self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
        else:
            self._spot.forward(step_size, self._base_command)

        #TODO Camera controller
        #current_time = time.time()
        #if current_time - self.last_capture_time >= self.picfreq: # take picture every picfreq seconds
        #    # Capture image rgb + depth + world cords of the camera
        #    worldcords = self.camera.get_world_pose()
        #    rgb = self.camera.get_rgba()
        #    depth = self.camera.get_depth()
        #    # Save image using OpenCV
        #    rgbdId = self.IDcounter 
        #    self.IDcounter += 1
  #
        #    print("position: ")
        #    print( worldcords)
#
        #    if rgb is not None:
        #        self.output_dir.mkdir(parents=True, exist_ok=True)
        #        rgb_path = self.output_dir / f"spot_camera_{rgbdId}.png"
        #        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR))
        #        #cv2.imwrite(f"spot_camera_depth_{rgbdId}.txt", depth * 255)
        #        #cv2.imwrite(f"spot_camera_worldcords_{rgbdId}.txt", worldcords * 255)
        #    self.last_capture_time = current_time

    def run(self) -> None:
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True
        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True


def main():
    physics_dt = 1 / 200.0
    render_dt = 1 / 60.0

    runner = SpotRunner(physics_dt=physics_dt, render_dt=render_dt)

    if runner is None:
        print("Failed to create SpotRunner")
        simulation_app.close()
        return

    simulation_app.update()
    runner._world.reset()
    simulation_app.update()
    runner.setup()
    simulation_app.update()
    while simulation_app.is_running():
        simulation_app.update()
    runner.run()
    simulation_app.close()


if __name__ == "__main__":
    main()
