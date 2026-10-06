"""
utils.py -- GIVEN infrastructure for the Force Control assignment
(Robot Manipulators): admittance vs. impedance control for surface
interaction.

This file is provided to you and you should NOT need to modify it. It
contains:
  - Configuration constants (Section 1)
  - Quaternion / orientation helpers (Section 2)
  - ArmController: kinematics, dynamics, torque actuation, tool attachment (3)
  - World / environment builder: manipulator + contact surface (4)
  - MinJerkTrajectory and a circular-path generator (5)
  - Contact force/torque sensing (6)
  - Logging / metrics (7)
  - GUI trajectory drawing (8)
  - Matplotlib plotting helpers (9)

Your work happens in main.py, which imports everything it needs from here.

NOTE ON SIMULATION STABILITY: this file bakes in several fixes that took
real debugging effort to find, and they matter more than they look:
  - inverse_kinematics() solves for the TOOL TIP pose (accounting for the
    tool offset), not the flange -- skipping this silently biases every
    pose by the tool length.
  - There is deliberately NO zero-torque "settle" loop after resetting
    joint positions -- stepping physics with zero applied torque before
    your controller starts lets gravity spike joint velocities before you
    even begin.
  - Physics runs at 1000 Hz (not the more common 240 Hz): at 240 Hz,
    semi-implicit Euler integration diverges for this URDF's mass chain on
    this PyBullet build. This is a numerical-integration issue, not a
    modeling one.
  - clamp_joint_velocities() is a safety valve applied every step -- torque
    control at high stiffness can occasionally spike a single joint's
    velocity for a step or two; silently clamping is far more robust than
    hand-tuning every gain to avoid it, and has no effect during normal
    (slow, quasi-static) motions.
  - Contact stiffness/damping are softened (see build_world) -- PyBullet's
    default contact model combined with torque control can otherwise
    produce unrealistic, simulation-only impact force spikes.
"""

import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# ---- Simulation ----
SIM_HZ = 1000
DT = 1.0 / SIM_HZ
GRAVITY = -9.81

# ---- Robot ----
ROBOT_URDF = "kuka_iiwa/model.urdf"
ROBOT_BASE_POSITION = [0.0, 0.0, 0.0]
EE_LINK_INDEX = 6
HOME_JOINT_POSITIONS = [0.0, 0.3, 0.0, -1.6, 0.0, 1.2, 0.0]

TOOL_LENGTH = 0.08
TOOL_OFFSET_LOCAL = np.array([0.0, 0.0, TOOL_LENGTH])
TOOL_RADIUS = 0.012
TOOL_MASS = 0.05

# ---- Contact / writing surface ----
SURFACE_CENTER_XY = np.array([0.5, 0.0])
SURFACE_HALF_EXTENT = 0.20     # flat square writing surface, half-size in x/y
SURFACE_THICKNESS = 0.10
SURFACE_TOP_Z = 0.30           # world-frame height of the surface's top face
SURFACE_TILT_DEG = 0.0

# Pre-contact hover pose: above the surface center, tool pointing straight down.
PRE_CONTACT_HEIGHT = 0.10
PEN_LIFT_HEIGHT = 0.03         # m above the surface during "pen-up" transitions

# ---- Desired contact force ----
F_DESIRED = 5.0                # N, along the surface normal (push into the surface)

# ---- "MIT" letter-tracing path (Parts 2, 4) ----
# Coordinates are in a local writing-plane frame (x = right, y = up on the
# page), in meters, centered roughly on the origin; get_mit_strokes() below
# offsets them onto the actual surface. Each letter is a list of PEN-DOWN
# strokes (line segments); moving between strokes/letters is a PEN-UP
# (lifted, stiff, no force regulation) transition.
LETTER_HEIGHT = 0.05
STROKE_SPEED = 0.04            # m/s along each pen-down stroke
PEN_UP_DURATION = 0.5          # s for each pen-up transition

# ---- Material identification samples (Part 6) ----
# Three small pads of different SIMULATED contact stiffness placed on the
# table -- a proxy for soft / medium / hard materials. The robot doesn't
# know these labels; it presses into each and estimates stiffness from the
# measured force-vs-penetration slope.
MATERIAL_SAMPLES = {
    "sample_A": dict(offset_xy=(0.0, -0.12), contact_stiffness=300, true_label="soft"),
    "sample_B": dict(offset_xy=(0.0, 0.0), contact_stiffness=1200, true_label="medium"),
    "sample_C": dict(offset_xy=(0.0, 0.12), contact_stiffness=4000, true_label="hard"),
}
MATERIAL_PROBE_DEPTH = 0.012   # m, target quasi-static press depth used to sweep out a loading curve
MATERIAL_PROBE_DURATION = 3.0  # s, slow controlled press (quasi-static -> clean stiffness estimate)
MATERIAL_CLASS_THRESHOLDS = (250.0, 350.0)  # N/m: soft < t0 <= medium < t1 <= hard

# ---- Admittance control gains (outer loop: force error -> position correction) ----
ADMITTANCE_MASS = 1.5          # kg, virtual mass
ADMITTANCE_DAMPING = 60.0      # N.s/m, virtual damping
# (no virtual stiffness term -- classical velocity/position admittance
#  integrates a pure mass-damper; this is what makes admittance control
#  "drift" to equilibrium rather than snapping back like a spring)

# ---- Inner motion-tracking gains (used to track the admittance-corrected
#      pose, and directly as the impedance gains for Parts 3-4) ----
GAINS_INNER_STIFF = dict(
    K_trans=np.array([600., 600., 600.]), K_rot=np.array([30., 30., 30.]), damping_ratio=1.8)

# Impedance control renders compliance directly via a LOWER normal-axis
# stiffness plus a virtual penetration setpoint (see main.py Part 3/4):
GAINS_IMPEDANCE = dict(
    K_trans=np.array([150., 150., 150.]), K_rot=np.array([15., 15., 15.]), damping_ratio=1.5)

# Stiff free-space gains, used for pen-up transitions between strokes.
GAINS_PEN_UP = dict(
    K_trans=np.array([900., 900., 900.]), K_rot=np.array([35., 35., 35.]), damping_ratio=1.4)

REFLECTED_MASS_TRANS = 2.0
REFLECTED_INERTIA_ROT = 0.05

# ---- Trial / episode control ----
APPROACH_DURATION = 2.0
SETTLE_DURATION = 3.0          # static-interaction hold time after making contact
MAX_EPISODE_TIME = 10.0
RANDOM_SEED_BASE = 42

# ---- Plot colors ----
NAVY, TEAL, GOLD, RED, GREY = "#1B2A4A", "#0E7C7B", "#C08A2E", "#B23A2E", "#666666"


# ==============================================================================
# 2. QUATERNION / ORIENTATION HELPERS
# ==============================================================================

def _quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_to_axis_angle_error(q_cur, q_des):
    """Orientation error e_rot (3-vector, world frame, ~axis*angle in rad) of
    the CURRENT orientation relative to the DESIRED one.

    Same sign convention as the position error e_pos = pos - pos_des, so the
    impedance law can use  F = -K * [e_pos; e_rot] - B * xdot  directly.
    """
    q_cur = np.asarray(q_cur)
    q_des = np.asarray(q_des)
    q_des_inv = np.array([-q_des[0], -q_des[1], -q_des[2], q_des[3]])
    q_err = _quat_mul(q_cur, q_des_inv)          # rotation taking q_des -> q_cur
    if q_err[3] < 0:
        q_err = -np.asarray(q_err)
    return np.array(q_err[0:3]) * 2.0


# ==============================================================================
# 3. ARM CONTROLLER
# ==============================================================================

class ArmController:
    def __init__(self, client_id, urdf=ROBOT_URDF, base_position=ROBOT_BASE_POSITION,
                 ee_link_index=EE_LINK_INDEX):
        self.cid = client_id
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.robot_id = p.loadURDF(urdf, base_position, useFixedBase=True, physicsClientId=self.cid)
        self.ee_link_index = ee_link_index

        self.joint_indices = []
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=self.cid)):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.cid)
            if info[2] != p.JOINT_FIXED:
                self.joint_indices.append(j)
        self.n_joints = len(self.joint_indices)

        p.setJointMotorControlArray(
            self.robot_id, self.joint_indices, p.VELOCITY_CONTROL,
            forces=[0.0] * self.n_joints, physicsClientId=self.cid)

        self.tool_id = None
        self.tool_constraint = None

    # ---------------------------------------------------------------- state
    def set_joint_positions(self, q, qdot=None):
        qdot = qdot if qdot is not None else np.zeros(self.n_joints)
        for idx, qi, qdi in zip(self.joint_indices, q, qdot):
            p.resetJointState(self.robot_id, idx, qi, qdi, physicsClientId=self.cid)

    def get_joint_state(self):
        states = p.getJointStates(self.robot_id, self.joint_indices, physicsClientId=self.cid)
        q = np.array([s[0] for s in states])
        qdot = np.array([s[1] for s in states])
        return q, qdot

    def get_ee_pose(self):
        state = p.getLinkState(self.robot_id, self.ee_link_index,
                                computeLinkVelocity=1, computeForwardKinematics=1,
                                physicsClientId=self.cid)
        pos, orn = state[4], state[5]
        lin_vel, ang_vel = state[6], state[7]
        return np.array(pos), np.array(orn), np.array(lin_vel), np.array(ang_vel)

    def get_tool_pose(self):
        pos, orn, lin_vel, ang_vel = self.get_ee_pose()
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        tool_pos = pos + R @ TOOL_OFFSET_LOCAL
        tool_lin_vel = lin_vel + np.cross(ang_vel, R @ TOOL_OFFSET_LOCAL)
        return tool_pos, orn, tool_lin_vel, ang_vel

    # ------------------------------------------------------------ dynamics
    def jacobian(self, q=None, qdot=None):
        if q is None:
            q, qdot = self.get_joint_state()
        qddot = [0.0] * self.n_joints
        jac_t, jac_r = p.calculateJacobian(
            self.robot_id, self.ee_link_index, list(TOOL_OFFSET_LOCAL),
            list(q), list(qdot), qddot, physicsClientId=self.cid)
        return np.vstack([np.array(jac_t), np.array(jac_r)])

    def gravity_compensation(self, q=None):
        if q is None:
            q, _ = self.get_joint_state()
        zeros = [0.0] * self.n_joints
        tau_g = p.calculateInverseDynamics(self.robot_id, list(q), zeros, zeros,
                                            physicsClientId=self.cid)
        tau_g = np.array(tau_g)
        if self.tool_id is not None:
            J = self.jacobian(q, zeros)
            # Add torque to hold up the tool's mass. GRAVITY is e.g. -9.81
            # F_comp = -m * g
            F_comp = np.array([0.0, 0.0, -TOOL_MASS * GRAVITY, 0.0, 0.0, 0.0])
            tau_g += J.T @ F_comp
        return tau_g

    def apply_joint_torques(self, tau):
        tau = np.clip(tau, -200, 200)
        p.setJointMotorControlArray(
            self.robot_id, self.joint_indices, p.TORQUE_CONTROL,
            forces=list(tau), physicsClientId=self.cid)

    def clamp_joint_velocities(self, limit=2.5):
        """Safety valve: cap joint speed after each physics step (see module
        docstring for why this is here)."""
        q, qdot = self.get_joint_state()
        if np.max(np.abs(qdot)) > limit:
            qdot_c = np.clip(qdot, -limit, limit)
            for idx, qi, qdi in zip(self.joint_indices, q, qdot_c):
                p.resetJointState(self.robot_id, idx, qi, qdi, physicsClientId=self.cid)

    # ------------------------------------------------------------------ IK
    def inverse_kinematics(self, target_pos, target_orn):
        """target_pos/target_orn describe the desired TOOL TIP pose (matches
        get_tool_pose()). PyBullet's IK solves for the LINK frame, so we
        convert the tool-tip target into the equivalent link-frame target
        first -- skipping this silently biases every pose by TOOL_LENGTH."""
        R = np.array(p.getMatrixFromQuaternion(target_orn)).reshape(3, 3)
        link_target_pos = np.asarray(target_pos) - R @ TOOL_OFFSET_LOCAL
        q = p.calculateInverseKinematics(
            self.robot_id, self.ee_link_index, list(link_target_pos), target_orn,
            physicsClientId=self.cid, maxNumIterations=200, residualThreshold=1e-5)
        return np.array(q[:self.n_joints])

    # ------------------------------------------------------------- tool
    def attach_tool(self, radius=TOOL_RADIUS, length=TOOL_LENGTH, color=(0.85, 0.55, 0.15, 1.0)):
        """A slim probe welded to the flange, ending in a small spherical
        tip -- stands in for a force-controlled end-effector. The COLLISION
        shape is a sphere (not the visual rod) positioned exactly at the
        tool tip: a flat cylinder face pressed against a flat surface is a
        classic degenerate rigid-body contact case (the contact point can
        jitter between corners each step, causing force chatter); a sphere
        always makes clean single-point contact."""
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=list(color),
                                   physicsClientId=self.cid)
        ee_pos, ee_orn, _, _ = self.get_ee_pose()
        R = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        tool_tip = ee_pos + R @ np.array([0, 0, length])
        self.tool_id = p.createMultiBody(
            baseMass=TOOL_MASS, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=list(tool_tip), baseOrientation=list(ee_orn), physicsClientId=self.cid)
            
        # PyBullet createConstraint parentFramePosition is relative to the parent link's center of mass.
        # We must subtract the COM offset to place the tool relative to the link frame origin.
        com = np.array(p.getDynamicsInfo(self.robot_id, self.ee_link_index, physicsClientId=self.cid)[3])
        parent_frame_pos = list(np.array([0, 0, length]) - com)
        
        self.tool_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id, parentLinkIndex=self.ee_link_index,
            childBodyUniqueId=self.tool_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=parent_frame_pos,
            childFramePosition=[0, 0, 0], physicsClientId=self.cid)
        p.changeConstraint(self.tool_constraint, maxForce=20000, physicsClientId=self.cid)
        p.changeDynamics(self.tool_id, -1, lateralFriction=0.4, physicsClientId=self.cid)
        return self.tool_id

    def sync_tool_pose(self, length=TOOL_LENGTH):
        if self.tool_id is None:
            return
        ee_pos, ee_orn, _, _ = self.get_ee_pose()
        R = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        tool_tip = ee_pos + R @ np.array([0, 0, length])
        p.resetBasePositionAndOrientation(self.tool_id, list(tool_tip), list(ee_orn),
                                           physicsClientId=self.cid)
        p.resetBaseVelocity(self.tool_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)


# ==============================================================================
# 4. WORLD / ENVIRONMENT
# ==============================================================================

class World:
    def __init__(self, client_id, arm, surface_id, surface_center_xy, surface_top_z,
                 surface_normal, nominal_quat, pre_contact_pos, material_ids=None):
        self.cid = client_id
        self.arm = arm
        self.surface_id = surface_id
        self.material_ids = material_ids or {}   # name -> (body_id, top_z, true_label)
        self.static_ids = [surface_id] + [v[0] for v in self.material_ids.values()]
        self.surface_center_xy = np.asarray(surface_center_xy)
        self.surface_top_z = surface_top_z
        self.surface_normal = np.asarray(surface_normal)  # unit vector, world frame
        self.nominal_quat = nominal_quat
        self.pre_contact_pos = np.asarray(pre_contact_pos)

    def penetration_depth(self, tool_pos, ref_top_z=None):
        """How far the tool tip is BELOW a reference top height (default: the
        writing surface) along the surface normal (positive = pressing in)."""
        ref = self.surface_top_z if ref_top_z is None else ref_top_z
        return float(ref - tool_pos[2])

    def disconnect(self):
        p.disconnect(physicsClientId=self.cid)


def build_world(gui=False, surface_center_xy=SURFACE_CENTER_XY, surface_top_z=SURFACE_TOP_Z,
                 surface_half_extent=SURFACE_HALF_EXTENT, tilt_deg=SURFACE_TILT_DEG,
                 surface_friction=0.5, with_material_samples=False):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setGravity(0, 0, GRAVITY, physicsClientId=cid)
    p.setTimeStep(DT, physicsClientId=cid)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)

    arm = ArmController(cid)

    tilt_rad = np.deg2rad(tilt_deg)
    surface_orn = p.getQuaternionFromEuler([tilt_rad, 0, 0])
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[surface_half_extent, surface_half_extent,
                                                            SURFACE_THICKNESS / 2], physicsClientId=cid)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[surface_half_extent, surface_half_extent,
                                                         SURFACE_THICKNESS / 2],
                               rgbaColor=[0.55, 0.58, 0.62, 1.0], physicsClientId=cid)
    surface_id = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
        basePosition=[surface_center_xy[0], surface_center_xy[1], surface_top_z - SURFACE_THICKNESS / 2],
        baseOrientation=surface_orn, physicsClientId=cid)
    p.changeDynamics(surface_id, -1, lateralFriction=surface_friction, physicsClientId=cid)
    # Soften contact response -- see module docstring.
    p.changeDynamics(surface_id, -1, contactStiffness=1500, contactDamping=15, physicsClientId=cid)

    material_ids = {}
    if with_material_samples:
        pad_half = 0.035
        pad_thickness = 0.02
        colors = {"sample_A": [0.85, 0.35, 0.35, 1.0], "sample_B": [0.35, 0.65, 0.85, 1.0],
                  "sample_C": [0.45, 0.75, 0.40, 1.0]}
        for name, spec in MATERIAL_SAMPLES.items():
            ox, oy = spec["offset_xy"]
            cx, cy = surface_center_xy[0] + ox, surface_center_xy[1] + oy
            pad_top_z = surface_top_z + pad_thickness  # sits on top of the writing surface
            pcol = p.createCollisionShape(p.GEOM_BOX, halfExtents=[pad_half, pad_half, pad_thickness / 2],
                                           physicsClientId=cid)
            pvis = p.createVisualShape(p.GEOM_BOX, halfExtents=[pad_half, pad_half, pad_thickness / 2],
                                        rgbaColor=colors.get(name, [0.6, 0.6, 0.6, 1.0]), physicsClientId=cid)
            pad_id = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=pcol, baseVisualShapeIndex=pvis,
                basePosition=[cx, cy, surface_top_z + pad_thickness / 2], physicsClientId=cid)
            p.changeDynamics(pad_id, -1, contactStiffness=spec["contact_stiffness"],
                              contactDamping=3.9 * np.sqrt(spec["contact_stiffness"]),
                              physicsClientId=cid)
            material_ids[name] = (pad_id, pad_top_z, spec["true_label"])

    R_tilt = np.array(p.getMatrixFromQuaternion(surface_orn)).reshape(3, 3)
    surface_normal = R_tilt @ np.array([0.0, 0.0, 1.0])

    nominal_quat = p.getQuaternionFromEuler([np.pi, 0, 0])  # tool pointing straight down
    pre_contact_pos = np.array([surface_center_xy[0], surface_center_xy[1],
                                 surface_top_z + PRE_CONTACT_HEIGHT])

    arm.set_joint_positions(HOME_JOINT_POSITIONS)
    q_home = arm.inverse_kinematics(pre_contact_pos, nominal_quat)
    arm.set_joint_positions(q_home)
    arm.attach_tool()
    p.changeDynamics(arm.tool_id, -1, contactStiffness=1500, contactDamping=15, physicsClientId=cid)
    # NOTE: deliberately no zero-torque stepSimulation warm-up here -- see
    # module docstring for why that is a real (if subtle) bug to avoid.

    return World(cid, arm, surface_id, surface_center_xy, surface_top_z, surface_normal,
                 nominal_quat, pre_contact_pos, material_ids=material_ids)


# ==============================================================================
# 5. TRAJECTORY GENERATION
# ==============================================================================

class MinJerkTrajectory:
    def __init__(self, p0, p1, duration):
        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)
        self.T = max(duration, 1e-6)

    def eval(self, t):
        tau = np.clip(t / self.T, 0.0, 1.0)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        sdot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.T
        pos = self.p0 + (self.p1 - self.p0) * s
        vel = (self.p1 - self.p0) * sdot
        return pos, vel


# Each letter is defined on a unit box (0<=x<=~0.65, 0<=y<=1, y-up) as a list
# of strokes; each stroke is a polyline of >=2 points drawn WITHOUT lifting
# the pen. Moving between strokes/letters is a PEN-UP transition (handled
# by main.py). These are simplified single-line "stick font" block letters
# (curves approximated by straight segments) -- optimized for being easy to
# trace with a compliant probe and easy to recognize from a top-down plot,
# not typographic accuracy.
LETTER_STROKES = {
    "A": [[(0, 0), (0.33, 1), (0.66, 0)], [(0.16, 0.35), (0.5, 0.35)]],
    "B": [[(0, 0), (0, 1)],
          [(0, 1), (0.45, 1), (0.6, 0.75), (0.45, 0.55), (0, 0.55)],
          [(0, 0.55), (0.5, 0.55), (0.65, 0.28), (0.5, 0), (0, 0)]],
    "C": [[(0.6, 0.82), (0.15, 1), (0, 0.5), (0.15, 0), (0.6, 0.18)]],
    "D": [[(0, 0), (0, 1)], [(0, 1), (0.4, 1), (0.62, 0.5), (0.4, 0), (0, 0)]],
    "E": [[(0.55, 0), (0, 0), (0, 1), (0.55, 1)], [(0, 0.5), (0.4, 0.5)]],
    "F": [[(0, 0), (0, 1), (0.55, 1)], [(0, 0.5), (0.4, 0.5)]],
    "G": [[(0.6, 0.82), (0.15, 1), (0, 0.5), (0.15, 0), (0.6, 0.18), (0.6, 0.42), (0.35, 0.42)]],
    "H": [[(0, 0), (0, 1)], [(0.55, 0), (0.55, 1)], [(0, 0.5), (0.55, 0.5)]],
    "I": [[(0.28, 0), (0.28, 1)]],
    "J": [[(0.45, 1), (0.45, 0.2), (0.3, 0), (0.1, 0), (0, 0.2)]],
    "K": [[(0, 0), (0, 1)], [(0.55, 1), (0, 0.45)], [(0, 0.45), (0.55, 0)]],
    "L": [[(0, 1), (0, 0), (0.5, 0)]],
    "M": [[(0, 0), (0, 1)], [(0, 1), (0.325, 0.35)], [(0.325, 0.35), (0.65, 1)], [(0.65, 1), (0.65, 0)]],
    "N": [[(0, 0), (0, 1)], [(0, 1), (0.6, 0)], [(0.6, 0), (0.6, 1)]],
    "O": [[(0.3, 1), (0.6, 0.8), (0.6, 0.2), (0.3, 0), (0, 0.2), (0, 0.8), (0.3, 1)]],
    "P": [[(0, 0), (0, 1)], [(0, 1), (0.45, 1), (0.6, 0.75), (0.45, 0.5), (0, 0.5)]],
    "Q": [[(0.3, 1), (0.6, 0.8), (0.6, 0.2), (0.3, 0), (0, 0.2), (0, 0.8), (0.3, 1)],
          [(0.32, 0.25), (0.62, -0.05)]],
    "R": [[(0, 0), (0, 1)], [(0, 1), (0.45, 1), (0.6, 0.75), (0.45, 0.5), (0, 0.5)],
          [(0.18, 0.5), (0.55, 0)]],
    "S": [[(0.58, 0.82), (0.15, 1), (0, 0.75), (0.15, 0.55), (0.45, 0.45), (0.58, 0.25),
           (0.43, 0), (0, 0.15)]],
    "T": [[(0, 1), (0.6, 1)], [(0.3, 1), (0.3, 0)]],
    "U": [[(0, 1), (0, 0.25), (0.15, 0), (0.42, 0), (0.57, 0.25), (0.57, 1)]],
    "V": [[(0, 1), (0.3, 0), (0.6, 1)]],
    "W": [[(0, 1), (0.17, 0), (0.33, 0.5), (0.49, 0), (0.65, 1)]],
    "X": [[(0, 0), (0.55, 1)], [(0, 1), (0.55, 0)]],
    "Y": [[(0, 1), (0.28, 0.5), (0.28, 0)], [(0.56, 1), (0.28, 0.5)]],
    "Z": [[(0, 1), (0.55, 1), (0, 0), (0.55, 0)]],
    " ": [],

    # Lowercase letters (simplified)
    "a": [[(0, 0.4), (0.45, 0.4), (0.45, 0), (0, 0), (0, 0.4)], [(0.45, 0.4), (0.45, 0)]],
    "b": [[(0, 1), (0, 0), (0.45, 0), (0.45, 0.4), (0, 0.4)]],
    "c": [[(0.45, 0.4), (0, 0.4), (0, 0), (0.45, 0)]],
    "d": [[(0.45, 1), (0.45, 0), (0, 0), (0, 0.4), (0.45, 0.4)]],
    "e": [[(0, 0.2), (0.45, 0.2), (0.45, 0.4), (0, 0.4), (0, 0), (0.45, 0)]],
    "f": [[(0.3, 1), (0.1, 1), (0.1, 0)], [(0, 0.5), (0.3, 0.5)]],
    "g": [[(0.45, 0.4), (0, 0.4), (0, 0), (0.45, 0), (0.45, -0.4), (0, -0.4)]],
    "h": [[(0, 1), (0, 0)], [(0, 0.4), (0.45, 0.4), (0.45, 0)]],
    "i": [[(0.2, 0.4), (0.2, 0)], [(0.2, 0.6), (0.2, 0.55)]], # dot
    "j": [[(0.3, 0.4), (0.3, -0.4), (0, -0.4)], [(0.3, 0.6), (0.3, 0.55)]],
    "k": [[(0, 1), (0, 0)], [(0.45, 0.4), (0, 0.1)], [(0.15, 0.2), (0.45, 0)]],
    "l": [[(0.2, 1), (0.2, 0)]],
    "m": [[(0, 0.4), (0, 0)], [(0, 0.4), (0.2, 0.4), (0.2, 0)], [(0.2, 0.4), (0.45, 0.4), (0.45, 0)]],
    "n": [[(0, 0.4), (0, 0)], [(0, 0.4), (0.45, 0.4), (0.45, 0)]],
    "o": [[(0, 0.4), (0.45, 0.4), (0.45, 0), (0, 0), (0, 0.4)]],
    "p": [[(0, 0.4), (0, -0.4)], [(0, 0.4), (0.45, 0.4), (0.45, 0), (0, 0)]],
    "q": [[(0.45, 0.4), (0.45, -0.4)], [(0.45, 0.4), (0, 0.4), (0, 0), (0.45, 0)]],
    "r": [[(0, 0.4), (0, 0)], [(0, 0.3), (0.4, 0.4)]],
    "s": [[(0.45, 0.3), (0.2, 0.4), (0, 0.3), (0.45, 0.1), (0.2, 0), (0, 0.1)]],
    "t": [[(0.2, 0.8), (0.2, 0)], [(0, 0.5), (0.4, 0.5)]],
    "u": [[(0, 0.4), (0, 0), (0.45, 0), (0.45, 0.4)]],
    "v": [[(0, 0.4), (0.2, 0), (0.45, 0.4)]],
    "w": [[(0, 0.4), (0.1, 0), (0.2, 0.3), (0.3, 0), (0.45, 0.4)]],
    "x": [[(0, 0.4), (0.45, 0)], [(0, 0), (0.45, 0.4)]],
    "y": [[(0, 0.4), (0, 0), (0.45, 0), (0.45, 0.4), (0.45, -0.4), (0, -0.4)]],
    "z": [[(0, 0.4), (0.45, 0.4), (0, 0), (0.45, 0)]],

}


def get_word_strokes(word, height=LETTER_HEIGHT):
    h = height
    letter_w = 0.65 * h
    gap = 0.30 * h

    strokes = []
    cursor_x = 0.0
    
    for ch in word:
        ch_key = ch if ch in LETTER_STROKES else ch.upper()
        if ch_key not in LETTER_STROKES:
            raise KeyError(f"get_word_strokes: no stroke font entry for {ch!r}.")
        
        char_strokes = []
        for stroke in LETTER_STROKES[ch_key]:
            char_strokes.append([(cursor_x + px * h, py * h) for (px, py) in stroke])
        
        strokes.extend(char_strokes)
        cursor_x += letter_w + gap

    total_width = cursor_x - gap
    strokes = [[(px - total_width / 2, py) for (px, py) in seg] for seg in strokes]
    return strokes



def get_mit_strokes(height=LETTER_HEIGHT):
    """Backwards-compatible alias -- writes 'MIT' specifically."""
    return get_word_strokes("MIT", height=height)


# ==============================================================================
# 6. CONTACT FORCE / TORQUE SENSING
# ==============================================================================

def get_contact_wrench(cid, body_id, static_ids, ref_point):
    force = np.zeros(3)
    torque = np.zeros(3)
    n_contacts = 0
    for obs_id in static_ids:
        pts = p.getContactPoints(bodyA=body_id, bodyB=obs_id, physicsClientId=cid)
        if pts is None:  # PyBullet returns None (not an empty tuple) on some
            pts = []     # platforms/builds when there are no contacts -- e.g.
                         # observed on macOS Metal builds. Guard against it here
                         # rather than at every call site.
        for c in pts:
            pos_on_a = np.array(c[5])
            normal_dir = np.array(c[7])
            normal_force = c[9]
            lat_dir1, lat_f1 = np.array(c[11]), c[10]
            lat_dir2, lat_f2 = np.array(c[13]), c[12]
            f = normal_dir * normal_force + lat_dir1 * lat_f1 + lat_dir2 * lat_f2
            force += f
            torque += np.cross(pos_on_a - np.asarray(ref_point), f)
            n_contacts += 1
    return force, torque, n_contacts


# ==============================================================================
# 7. LOGGING / METRICS
# ==============================================================================

class TrialLogger:
    def __init__(self):
        self.t, self.pos, self.pos_des = [], [], []
        self.force, self.force_des, self.torque = [], [], []
        self._wall_start = self._wall_end = None

    def start_timer(self):
        self._wall_start = time.perf_counter()

    def stop_timer(self):
        self._wall_end = time.perf_counter()

    def record(self, t, pos, pos_des, force=(0, 0, 0), force_des=0.0, torque=(0, 0, 0)):
        self.t.append(t)
        self.pos.append(np.array(pos))
        self.pos_des.append(np.array(pos_des))
        self.force.append(np.array(force))
        self.force_des.append(force_des)
        self.torque.append(np.array(torque))

    def arrays(self):
        return dict(t=np.array(self.t), pos=np.array(self.pos), pos_des=np.array(self.pos_des),
                    force=np.array(self.force), force_des=np.array(self.force_des),
                    torque=np.array(self.torque))

    def wall_clock_time(self):
        if self._wall_start is None or self._wall_end is None:
            return None
        return self._wall_end - self._wall_start

    def force_tracking_rmse(self, mask=None, axis=2):
        f = np.array(self.force)[:, axis]
        fd = np.array(self.force_des)
        if mask is not None:
            f, fd = f[mask], fd[mask]
        if len(f) == 0:
            return 0.0
        return float(np.sqrt(np.mean((f - fd) ** 2)))

    def settling_time(self, axis=2, tolerance=0.5, contact_start_idx=0):
        """First time (relative to contact_start_idx) the force stays within
        `tolerance` N of the desired force for the rest of the trial."""
        f = np.array(self.force)[:, axis]
        fd = np.array(self.force_des)
        t = np.array(self.t)
        err = np.abs(f - fd)
        for i in range(contact_start_idx, len(err)):
            if np.all(err[i:] < tolerance):
                return float(t[i] - t[contact_start_idx])
        return None

    def path_smoothness(self):
        pos, t = np.array(self.pos), np.array(self.t)
        if len(pos) < 4:
            return 0.0
        dt = np.mean(np.diff(t)) if len(t) > 1 else DT
        vel = np.gradient(pos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        jerk = np.gradient(acc, dt, axis=0)
        return float(np.mean(np.sum(jerk**2, axis=1)))

    def to_summary(self, extra=None):
        summary = dict(wall_clock_time_s=self.wall_clock_time(),
                        sim_time_s=float(self.t[-1]) if self.t else 0.0,
                        force_tracking_rmse_N=self.force_tracking_rmse(),
                        path_smoothness_jerk2=self.path_smoothness())
        if extra:
            summary.update(extra)
        return summary


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))



import matplotlib.pyplot as plt

# 8. GUI DEBUG-LINE TRAJECTORY DRAWING
# ==============================================================================

class TrajectoryDrawer:
    def __init__(self, cid, color=(0, 0, 0), width=2.5, min_dist=0.002):
        self.cid, self.color, self.width, self.min_dist = cid, color, width, min_dist
        self.last_point = None

    def update(self, point):
        if self.last_point is not None:
            d = sum((a - b) ** 2 for a, b in zip(point, self.last_point)) ** 0.5
            if d < self.min_dist:
                return
            p.addUserDebugLine(list(self.last_point), list(point), lineColorRGB=list(self.color),
                                lineWidth=self.width, lifeTime=0, physicsClientId=self.cid)
        self.last_point = tuple(point)


# ==============================================================================
# 9. PLOTTING HELPERS
# ==============================================================================

def plot_force_tracking(arr, save_path, title="Contact Force Tracking", axis=2, axis_label="Fz"):
    t = arr["t"]
    f = arr["force"][:, axis]
    fd = arr["force_des"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, fd, color=GREY, ls="--", lw=1.8, label=f"desired {axis_label}")
    ax.plot(t, f, color=RED, lw=1.8, label=f"measured {axis_label}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=13, weight="bold", color=NAVY)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_motion_trajectory(arr, save_path, title="End-Effector Motion Trajectory"):
    t, pos, pos_des = arr["t"], arr["pos"], arr["pos_des"]
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True)
    labels = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        ax.plot(t, pos_des[:, i], color=GREY, ls="--", lw=1.8, label="reference")
        ax.plot(t, pos[:, i], color=NAVY, lw=1.8, label="achieved")
        ax.set_ylabel(f"{labels[i]} (m)")
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.legend(fontsize=9, frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=13, weight="bold", color=NAVY)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_xy_path(arr, save_path, title="End-Effector Path (top-down)"):
    import numpy as np
    pos = arr["pos"].copy()
    
    # Insert NaNs where pen is lifted to break the line
    contact = arr["force_des"] > 0
    pos[~contact, :] = np.nan
    
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(pos[:, 0], pos[:, 1], color="black", lw=1.6)
    
    # Start and end
    start_idx = np.argmax(contact)
    end_idx = len(contact) - 1 - np.argmax(contact[::-1])
    ax.scatter(*pos[start_idx, :2], color=TEAL, s=40, label="start", zorder=3)
    ax.scatter(*pos[end_idx, :2], color=RED, s=40, label="end", zorder=3)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(title, fontsize=12, weight="bold", color=NAVY)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor="white")
    plt.close(fig)


def bar_compare(group_labels, values_dict, ylabel, save_path, title):
    n_groups = len(group_labels)
    n_series = len(values_dict)
    width = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [NAVY, GOLD, RED, TEAL]
    for i, (name, vals) in enumerate(values_dict.items()):
        xs = np.arange(n_groups) + i * width
        ax.bar(xs, vals, width=width, color=colors[i % len(colors)], label=name)
    ax.set_xticks(np.arange(n_groups) + width * (n_series - 1) / 2)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=12, weight="bold", color=NAVY)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor="white")
    plt.close(fig)


