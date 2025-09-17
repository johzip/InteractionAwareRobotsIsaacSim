# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional

import numpy as np
import omni.kit.commands
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController


class SpotFlatTerrainPolicy(PolicyController):
    """The Spot quadruped"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "spot",
        usd_path: str = None,
        policy_path: str = None, 
        policy_params_path: str = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(policy_path, policy_params_path)
        self._action_scale = 0.2
        self._previous_action = np.zeros(12)
        self._policy_counter = 0

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(48)
        # Base lin vel
        obs[:3] = lin_vel_b
        # Base ang vel
        obs[3:6] = ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9:12] = command
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[12:24] = current_joint_pos - self.default_pos
        obs[24:36] = current_joint_vel
        # Previous Action
        obs[36:48] = self._previous_action

        return obs
    

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        action = ArticulationAction(joint_positions=self.default_pos + (self.action * self._action_scale))
        self.robot.apply_action(action)

        self._policy_counter += 1


class SpotArmFlatTerrainPolicy(PolicyController):
    """The Spot quadruped"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "spot",
        usd_path: str = None,
        policy_path: str = None, 
        policy_params_path: str = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.arm_joint_indices = [0, 1, 2, 7, 12, 17, 18]  # arm0_sh1, arm0_sh0, arm0_el0, arm0_el1, arm0_wr0, arm0_wr1
        self.leg_joint_indices = [3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16]  # All non-arm joints

        self.load_policy(policy_path, policy_params_path)
        self._action_scale = 0.2
        self._previous_action = np.zeros(19)
        self._policy_counter = 0

    #compute observation with arm joint hiding when in manual control
    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy
        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(69)
        obs[:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = command
        
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        
        if hasattr(self, 'manual_arm_control') and self.manual_arm_control:
            # Create fake observations where arms appear at default positions
            fake_joint_pos = np.array(current_joint_pos).copy()
            fake_joint_vel = np.array(current_joint_vel).copy()
            default_pos_array = np.array(self.default_pos)
            
            # Set arm joints to default in observation (policy won't see arm movement)
            fake_joint_pos[self.arm_joint_indices] = default_pos_array[self.arm_joint_indices]
            fake_joint_vel[self.arm_joint_indices] = 0.0
            
            obs[12:31] = fake_joint_pos - self.default_pos
            obs[31:50] = fake_joint_vel
            
            # Also fake previous arm actions
            fake_previous_action = self._previous_action.copy()
            fake_previous_action[self.arm_joint_indices] = 0.0
            obs[50:69] = fake_previous_action
            
        else:
            # Normal observation
            obs[12:31] = current_joint_pos - self.default_pos
            obs[31:50] = current_joint_vel
            obs[50:69] = self._previous_action

        return obs


    def forward_removedPhysic(self, dt, command, manual_arm_control=False, arm_changes=None):
        """
        Move base by transform (v_x, v_y, w_z) and control arm joints.
        Legs are held at default positions.
        """
        # Move base via transform
        v_x, v_y, w_z = float(command[0]), float(command[1]), float(command[2])

        # Current pose
        pos_w, q_w = self.robot.get_world_pose()  # pos: (3,), q: (w,x,y,z)
        R_wb = quat_to_rot_matrix(q_w)

        # Configurable speeds
        lin_scale = 1.0  # m/s per command unit
        yaw_scale = 1.5  # rad/s per command unit

        # Translate in robot frame then rotate to world
        delta_local = np.array([v_x, v_y, 0.0]) * lin_scale * dt
        delta_world = R_wb @ delta_local
        new_pos = pos_w + delta_world

        # Yaw update around world Z: q_new = q_delta ⊗ q
        dyaw = w_z * yaw_scale * dt
        c, s = np.cos(dyaw * 0.5), np.sin(dyaw * 0.5)
        q_delta = np.array([c, 0.0, 0.0, s])  # (w,x,y,z)

        # Quaternion multiply: (w1,x1,y1,z1) ⊗ (w2,x2,y2,z2)
        def quat_mul(a, b):
            aw, ax, ay, az = a
            bw, bx, by, bz = b
            return np.array([
                aw*bw - ax*bx - ay*by - az*bz,
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw,
            ], dtype=float)

        new_q = quat_mul(q_delta, q_w)
        self.robot.set_world_pose(new_pos, new_q)

        # Build joint targets:
        # - Legs: hold at default
        # - Arm: apply manual changes if provided; else default
        current_joint_pos = self.robot.get_joint_positions()
        target = np.array(self.default_pos, dtype=float)

        # Preserve current arm as base to add changes smoothly
        if manual_arm_control and arm_changes is not None:
            arm_changes = np.asarray(arm_changes, dtype=float)
            if len(arm_changes) >= len(self.arm_joint_indices):
                target[self.arm_joint_indices] = (
                    current_joint_pos[self.arm_joint_indices] + arm_changes[:len(self.arm_joint_indices)]
                )
        # Clamp arm to limits (same as before)
        arm_joint_limits = [
            (np.deg2rad(-179.99985), np.deg2rad(30.00001)),   # arm0_sh1
            (np.deg2rad(-149.99977), np.deg2rad(179.99985)),  # arm0_sh0
            (0.0, np.deg2rad(179.99985)),                     # arm0_el0
            (np.deg2rad(-160.00018), np.deg2rad(160.00018)),  # arm0_el1
            (np.deg2rad(-105.00024), np.deg2rad(105.00024)),  # arm0_wr0
            (np.deg2rad(-165.00554), np.deg2rad(164.9998)),   # arm0_wr1
            (np.deg2rad(-90.0), np.deg2rad(0.0)),             # arm0_f1x
        ]
        if manual_arm_control and arm_changes is not None:
            updated = target[self.arm_joint_indices]
            for i, (lo, hi) in enumerate(arm_joint_limits):
                if updated[i] < lo: updated[i] = lo
                if updated[i] > hi: updated[i] = hi
            target[self.arm_joint_indices] = updated

        # Hold legs at default
        target[self.leg_joint_indices] = np.array(self.default_pos)[self.leg_joint_indices]

        # Apply joints
        self.robot.apply_action(ArticulationAction(joint_positions=target))

        # Book-keeping (no leg policy used)
        self._previous_action = np.zeros_like(self._previous_action)
        self._policy_counter += 1


    def forward(self, dt, command, manual_arm_control=False, arm_changes=None):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        # Set the flag so _compute_observation can use it
        self.manual_arm_control = manual_arm_control

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)

            self._previous_action = self.action.copy()

        action = ArticulationAction(joint_positions=self.default_pos + (self.action * self._action_scale))

        if manual_arm_control:
            new_arm_changes = np.array(arm_changes)
            current_joint_pos = self.robot.get_joint_positions()
            
            
            updated_arm_pos = current_joint_pos[self.arm_joint_indices] + new_arm_changes
        
            # Define joint limits for each arm joint
            arm_joint_limits = [
                (np.deg2rad(-179.99985), np.deg2rad(30.00001)),   # arm0_sh1: -180° to 30°
                (np.deg2rad(-149.99977), np.deg2rad(179.99985)),  # arm0_sh0: -150° to 179.99985°
                (0.0, np.deg2rad(179.99985)),                     # arm0_el0: 0° to 180°
                (np.deg2rad(-160.00018), np.deg2rad(160.00018)),  # arm0_el1: -160° to 160°
                (np.deg2rad(-105.00024), np.deg2rad(105.00024)),  # arm0_wr0: -105° to 105°
                (np.deg2rad(-165.00554), np.deg2rad(164.9998)),    # arm0_wr1: -165° to 165°
                (np.deg2rad(-90.0), np.deg2rad(0.0))          # arm0_f1x: -90° to 0°
            ]

            # Enforce joint limits
            for i, (low, high) in enumerate(arm_joint_limits):
                if low is not None and updated_arm_pos[i] < low:
                    updated_arm_pos[i] = low
                if high is not None and updated_arm_pos[i] > high:
                    updated_arm_pos[i] = high

            action.joint_positions[self.arm_joint_indices] = updated_arm_pos


        self.robot.apply_action(action)

        self._policy_counter += 1