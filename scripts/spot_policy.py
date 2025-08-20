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
    """The Spot quadruped with separate arm control"""

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
        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(policy_path, policy_params_path)
        self._action_scale = 0.2
        self._previous_action = np.zeros(19)
        self._policy_counter = 0
        
        # Define arm joint indices based on your output
        self.arm_joint_indices = [0, 1, 2, 7, 12, 17, 18]  # arm0_sh1, arm0_sh0, arm0_el0, arm0_el1, arm0_wr0, arm0_wr1, arm0_f1x
        self.leg_joint_indices = [3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16]  # All non-arm joints
        
        # Manual arm control state
        self.manual_arm_control = False
        self.manual_arm_targets = None

    def set_manual_arm_control(self, enabled: bool, arm_targets: np.ndarray = None):
        """Enable/disable manual arm control"""
        self.manual_arm_control = enabled
        if enabled and arm_targets is not None:
            self.manual_arm_targets = arm_targets.copy()

    def _compute_observation_legs_only(self, command):
        """
        Compute observation for legs only (exclude arm joints from feedback)
        This prevents the policy from "fighting" manual arm control
        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        # Get only leg joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        
        # FIX: Use numpy array indexing or list comprehension
        current_joint_pos = np.array(current_joint_pos)
        current_joint_vel = np.array(current_joint_vel)
        default_pos_array = np.array(self.default_pos)
        previous_action_array = np.array(self._previous_action)
        
        leg_positions = current_joint_pos[self.leg_joint_indices]
        leg_velocities = current_joint_vel[self.leg_joint_indices]
        leg_default_pos = default_pos_array[self.leg_joint_indices]
        
        # Previous leg actions only
        previous_leg_actions = previous_action_array[self.leg_joint_indices]

        # Create observation with legs only (48 elements like original Spot policy)
        obs = np.zeros(48)
        obs[:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = command
        obs[12:24] = leg_positions - leg_default_pos  # 12 leg joints
        obs[24:36] = leg_velocities  # 12 leg velocities
        obs[36:48] = previous_leg_actions  # 12 previous leg actions

        return obs

    def _compute_observation_full(self, command):
        """Original full observation for when arm is policy-controlled"""
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
        obs[12:31] = current_joint_pos - self.default_pos
        obs[31:50] = current_joint_vel
        obs[50:69] = self._previous_action

        return obs

    def forward(self, dt, command, manual_arm_control=False, arm_targets=None):
        """
        Compute and apply actions with separate arm/leg control
        """
        self.manual_arm_control = manual_arm_control
        if manual_arm_control and arm_targets is not None:
            self.manual_arm_targets = np.array(arm_targets)

        if self._policy_counter % self._decimation == 0:
            # Always use full observation - policy needs 69 elements
            obs = self._compute_observation_full(command)
            self.action = self._compute_action(obs)
            
            # If manual arm control is active, override arm actions
            if self.manual_arm_control and self.manual_arm_targets is not None:
                # Override arm actions with manual targets
                default_pos_array = np.array(self.default_pos)
                arm_target_deviation = self.manual_arm_targets - default_pos_array[self.arm_joint_indices]
                
                # Replace arm actions with manual commands
                self.action = np.array(self.action)  # Ensure numpy array
                self.action[self.arm_joint_indices] = arm_target_deviation
                
                print(f"Overriding arm actions: {self.action[self.arm_joint_indices]}")

            self._previous_action = self.action.copy()

        # Apply the combined action
        default_pos_array = np.array(self.default_pos)
        joint_positions = default_pos_array + (self.action * self._action_scale)
        
        action = ArticulationAction(joint_positions=joint_positions)
        self.robot.apply_action(action)

        self._policy_counter += 1

    def _compute_leg_action(self, leg_obs):
        """
        Compute action for legs only using adapted policy
        We need to create a fake full observation for the policy
        """
        import torch
        
        with torch.no_grad():
            # FIX: The policy expects 69-element observation, not 48
            # We need to reconstruct a full observation with fake arm data
            
            # Get current full joint states
            current_joint_pos = np.array(self.robot.get_joint_positions())
            current_joint_vel = np.array(self.robot.get_joint_velocities())
            default_pos_array = np.array(self.default_pos)
            
            # Create fake observation where arm joints are at default positions
            fake_joint_pos = current_joint_pos.copy()
            fake_joint_vel = current_joint_vel.copy()
            
            # Set arm joints to default positions in the fake observation
            fake_joint_pos[self.arm_joint_indices] = default_pos_array[self.arm_joint_indices]
            fake_joint_vel[self.arm_joint_indices] = 0.0  # Zero arm velocities
            
            # Reconstruct full 69-element observation
            full_obs = np.zeros(69)
            full_obs[:12] = leg_obs[:12]  # Copy base state and command from leg_obs
            full_obs[12:31] = fake_joint_pos - default_pos_array  # All joints (with fake arm)
            full_obs[31:50] = fake_joint_vel  # All joint velocities (with fake arm)
            full_obs[50:69] = self._previous_action  # Previous actions
            
            obs_tensor = torch.from_numpy(full_obs).view(1, -1).float()
            
            # Get full 19-DOF action from policy
            full_action = self.policy(obs_tensor).detach().view(-1).numpy()
            
            # Extract only leg actions using numpy array indexing
            leg_actions = full_action[self.leg_joint_indices]
            
            return leg_actions