# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import sys
import carb

import omni.usd
import numpy as np
from pxr import Sdf, UsdLux, Gf, Tf
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.types import ArticulationAction, XFormPrimViewState
from isaacsim.core.prims import Articulation
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

# preparing the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()


my_world = World(stage_units_in_meters=1.0)

# Add Ground Plane
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

#Add Room Scene
# add Room by staging the usda file
usd_path = os.path.abspath("Assets/Scenes/07f5b601ee.usda")
omni.usd.get_context().open_stage(usd_path)

# Check if a collider attribute exists in the each children of the Room, if not create it and set its value to triangle mesh collider
#children = room.GetAllChildren()
#for child in children:
#    attr = child.GetAttribute("collider")
#    if not attr.IsValid():
#        attribute = child.CreateAttribute("collider", Sdf.ValueTypeNames.Token)
#    attr.Set("triangle_mesh")

#Add Spot with arm Robot
asset_path = "/home/zipfelj/isaacsim/Assets/spot_robots/spot_arm.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/spot")  # add robot to stage
spot = Articulation(prim_paths_expr="/World/spot", name="spot_with_arm")  # create an articulation object
spot.set_world_poses(positions=np.array([[1.4, 2.88, 0.93]]), orientations=np.array([[0.0, 0.0, 0.0, 0.5]]))    





#keep it running
while simulation_app.is_running():
    simulation_app.update()
    
# shutdown the simulator automatically
#simulation_app.close() 
