#!/usr/bin/env python3
"""
AR523: Robot Manipulators -- Assignment 2
Force-Controlled Writing & Haptic Material Identification
"""

import argparse
import os
import time

import numpy as np
import pybullet as p
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import *


# ==============================================================================
# PART 1: Trajectory Generation
# ==============================================================================

def generate_pattern(shape, scale=0.1):
    """
    TODO 1: Implement shape trajectory generation.
    Returns a list of strokes (each stroke is a list of (x,y) tuples).
    """
    raise NotImplementedError("TODO 1: Write the code to generate the (x,y) coordinates for the shapes: \'circle\', \'spiral\', \'infinity\', and \'heart\'.")


# ==============================================================================
# PART 2: Cartesian Impedance Control Law
# ==============================================================================

class CartesianImpedanceController:

    def __init__(self, arm):
        self.arm = arm

    def step(self, pos_des, quat_des, K_trans, K_rot, damping_ratio=1.0):
        # ----------------------------------------------------------------------
        # TODO 2: Implement the Cartesian impedance torque law.
        # Inputs:
        #   pos_des: (3,) desired position
        #   quat_des: (4,) desired orientation quaternion
        #   K_trans: (3,) translational stiffness (x,y,z)
        #   K_rot: (3,) rotational stiffness (x,y,z)
        #   damping_ratio: scalar
        # Returns:
        #   tau: (7,) joint torques
        #   info: dict for logging (provided in solution)
        # ----------------------------------------------------------------------
        raise NotImplementedError("TODO 2: Write the math for the impedance controller to calculate the required joint torques.")


# ==============================================================================
# PART 3: Admittance Control Outer Loop
# ==============================================================================

class AdmittanceController:
    def __init__(self, mass=ADMITTANCE_MASS, damping=ADMITTANCE_DAMPING):
        self.M = mass
        self.B = damping
        self.xc = 0.0
        self.xc_dot = 0.0

    def reset(self):
        self.xc = 0.0
        self.xc_dot = 0.0

    def step(self, f_measured, f_desired, dt, xc_limit=0.03, xc_dot_limit=0.08):
        """
        TODO 3: Implement admittance mass-damper outer loop step.
        Returns: float (self.xc)
        """
        raise NotImplementedError("TODO 3: Write the code to update the admittance mass-damper system over time.")

ADMITTANCE_NOMINAL_PENETRATION = 0.015


# ==============================================================================
# LIVE VISUALIZATION (TODO 8)
# ==============================================================================

class LiveDashboard:
    def __init__(self, mode="writing"):
        """
        TODO 7: Implement a real-time matplotlib dashboard using plt.ion().
        """
        raise NotImplementedError("TODO 7: Create a live plotting window using matplotlib.")

    def update_writing(self, t, pos_act, pos_des, f_act, f_des):
        self.t_data.append(t)
        if f_des > 0:
            self.x_act.append(pos_act[0]); self.y_act.append(pos_act[1])
            self.x_des.append(pos_des[0]); self.y_des.append(pos_des[1])
        else:
            self.x_act.append(np.nan); self.y_act.append(np.nan)
            self.x_des.append(np.nan); self.y_des.append(np.nan)
        self.f_act.append(f_act[2])
        self.f_des.append(f_des)
        
        now = time.time()
        if now - self.last_update > 0.05:
            self.line_xy_act.set_data(self.x_act, self.y_act)
            self.line_xy_des.set_data(self.x_des, self.y_des)
            self.ax_xy.relim()
            self.ax_xy.autoscale_view()
            
            self.line_f_act.set_data(self.t_data, self.f_act)
            self.line_f_des.set_data(self.t_data, self.f_des)
            self.ax_f.relim()
            self.ax_f.autoscale_view()
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.last_update = now

    def update_material(self, depth, f_act, k_hat=None):
        self.depths.append(depth)
        self.f_act.append(f_act)
        
        now = time.time()
        if now - self.last_update > 0.05:
            self.line_fd.set_data(self.depths, self.f_act)
            if k_hat is not None and len(self.depths) > 0:
                d_max = max(self.depths)
                self.line_fit.set_data([0, d_max], [0, k_hat * d_max])
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.last_update = now
            
    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ==============================================================================
# WRITING & PROBING RUNNERS
# ==============================================================================

def build_stroke_plan(world, letter_origin_xy, word=None, pattern=None):
    strokes = []
    if word:
        strokes.extend(get_word_strokes(word))
    if pattern:
        pattern_strokes = generate_pattern(pattern)
        offset = 0.5 if word else 0.0
        pattern_strokes = [[(px + offset, py) for (px, py) in seg] for seg in pattern_strokes]
        strokes.extend(pattern_strokes)
        
    plan = []
    for stroke in strokes:
        pts = [np.array(letter_origin_xy) + np.array(pt) for pt in stroke]
        for i in range(len(pts)):
            if i == 0:
                if plan: plan.append(("up", plan[-1][2], pts[0]))
                else:    plan.append(("up", world.pre_contact_pos[:2], pts[0]))
            if i < len(pts) - 1:
                plan.append(("down", pts[i], pts[i+1]))
    return plan


def run_writing_task(world, ctrl, controller_mode, f_desired, gui=False, save_dir="outputs", tag="writing", word="Robot", pattern=None, live=False):
    plan = build_stroke_plan(world, world.surface_center_xy, word=word, pattern=pattern)
    adm = AdmittanceController()

    drawer = TrajectoryDrawer(world.cid, color=(0, 0, 0)) if gui else None
    dashboard = LiveDashboard(mode="writing") if live else None
    logger = TrialLogger()
    logger.start_timer()

    first_xy = plan[0][1]
    lift_pos = np.array([first_xy[0], first_xy[1], world.surface_top_z + PEN_LIFT_HEIGHT])
    traj_in = MinJerkTrajectory(world.pre_contact_pos, lift_pos, APPROACH_DURATION)

    t = 0.0
    for step in range(int(APPROACH_DURATION / DT)):
        pos_des, _ = traj_in.eval(t)
        tau, _ = ctrl.step(pos_des, world.nominal_quat, GAINS_PEN_UP["K_trans"], GAINS_PEN_UP["K_rot"], GAINS_PEN_UP["damping_ratio"])
        world.arm.apply_joint_torques(tau)
        p.stepSimulation(physicsClientId=world.cid)
        world.arm.clamp_joint_velocities()
        tool_pos, _, _, _ = world.arm.get_tool_pose()
        logger.record(t, tool_pos, pos_des, force=np.zeros(3), force_des=0.0)
        t += DT

    prev_kind = None
    for kind, xy_from, xy_to in plan:
        if kind == "up":
            p_from = np.array([xy_from[0], xy_from[1], world.surface_top_z + PEN_LIFT_HEIGHT])
            p_to = np.array([xy_to[0], xy_to[1], world.surface_top_z + PEN_LIFT_HEIGHT])
            traj = MinJerkTrajectory(p_from, p_to, PEN_UP_DURATION)
            for step in range(int(PEN_UP_DURATION / DT)):
                pos_des, _ = traj.eval(step * DT)
                tau, _ = ctrl.step(pos_des, world.nominal_quat, GAINS_PEN_UP["K_trans"], GAINS_PEN_UP["K_rot"], GAINS_PEN_UP["damping_ratio"])
                world.arm.apply_joint_torques(tau)
                p.stepSimulation(physicsClientId=world.cid)
                world.arm.clamp_joint_velocities()
                tool_pos, _, _, _ = world.arm.get_tool_pose()
                logger.record(t, tool_pos, pos_des, force=np.zeros(3), force_des=0.0)
                t += DT
            adm.reset()
        else:
            if prev_kind != "down":
                adm.reset()
                z_start = world.surface_top_z + PEN_LIFT_HEIGHT
                z_end = world.surface_top_z + TOOL_RADIUS
                traj_app = MinJerkTrajectory(np.array([xy_from[0], xy_from[1], z_start]), np.array([xy_from[0], xy_from[1], z_end]), 0.15)
                for step in range(int(0.15 / DT)):
                    pos_des, _ = traj_app.eval(step * DT)
                    tau, _ = ctrl.step(pos_des, world.nominal_quat, GAINS_PEN_UP["K_trans"], GAINS_PEN_UP["K_rot"], GAINS_PEN_UP["damping_ratio"])
                    world.arm.apply_joint_torques(tau)
                    p.stepSimulation(physicsClientId=world.cid)
                    world.arm.clamp_joint_velocities()
                
            stroke_len = np.linalg.norm(xy_to - xy_from)
            duration = max(stroke_len / STROKE_SPEED, 0.15)
            xy_traj = MinJerkTrajectory(np.append(xy_from, 0), np.append(xy_to, 0), duration)
            for step in range(int(duration / DT)):
                xy_pos, _ = xy_traj.eval(step * DT)
                tool_pos, _, _, _ = world.arm.get_tool_pose()
                force, _, _ = get_contact_wrench(world.cid, world.arm.tool_id, world.static_ids, tool_pos)
                f_meas_z = force[2]

                if controller_mode == "admittance":
                    xc = adm.step(f_meas_z, f_desired, DT)
                    z_des = world.surface_top_z + TOOL_RADIUS - ADMITTANCE_NOMINAL_PENETRATION + xc
                    gains = GAINS_INNER_STIFF
                else:
                    # ----------------------------------------------------------
                    # TODO 2 (continued): Impedance Writing
                    # Write the logic to achieve the desired 5N contact force.
                    # ----------------------------------------------------------
                    raise NotImplementedError("TODO 2 (continued): Write the logic to achieve the desired 5N contact force.")
                    gains = GAINS_IMPEDANCE

                pos_des = np.array([xy_pos[0], xy_pos[1], z_des])
                tau, _ = ctrl.step(pos_des, world.nominal_quat, gains["K_trans"], gains["K_rot"], gains["damping_ratio"])
                world.arm.apply_joint_torques(tau)
                p.stepSimulation(physicsClientId=world.cid)
                world.arm.clamp_joint_velocities()

                tool_pos, _, _, _ = world.arm.get_tool_pose()
                force, torque, _ = get_contact_wrench(world.cid, world.arm.tool_id, world.static_ids, tool_pos)
                
                logger.record(t, tool_pos, pos_des, force=force, force_des=f_desired, torque=torque)
                if dashboard:
                    dashboard.update_writing(t, tool_pos, pos_des, force, f_desired)
                if drawer is not None and step % 3 == 0:
                    drawer.update(tool_pos)
                t += DT
        prev_kind = kind
    
    logger.stop_timer()
    if dashboard: dashboard.close()

    os.makedirs(save_dir, exist_ok=True)
    arr = logger.arrays()
    
    # --------------------------------------------------------------------------
    # TODO 4: Compute Metrics
    # Compute force tracking RMSE (only when f_des > 0) and path smoothness (mean squared jerk).
    # Store them in variables named `rmse` and `smoothness`.
    # --------------------------------------------------------------------------
    raise NotImplementedError("TODO 4: Calculate the error between the desired force and the actual measured force.")
    rmse = 0.0
    smoothness = 0.0
    
    plot_force_tracking(arr, os.path.join(save_dir, f"{tag}_force_tracking.png"), title=f"{tag}: Force Tracking While Writing")
    plot_xy_path(arr, os.path.join(save_dir, f"{tag}_xy_path.png"), title=f"{tag}: Traced Path (top-down)")
    
    summary = dict(force_tracking_rmse_N=rmse, path_smoothness_jerk2=smoothness, wall_clock_time_s=logger.wall_clock_time(), sim_time_s=float(arr["t"][-1]))
    save_json(os.path.join(save_dir, f"{tag}_summary.json"), summary)
    print(f"[{tag}] force RMSE = {rmse:.3f} N")
    return summary, arr


def probe_material(world, ctrl, pad_id, pad_top_z, xy, save_dir, tag, fixed_penetration=0.010, settle_duration=3.0, live=False):
    gains = dict(K_trans=np.array([600., 600., 600.]), K_rot=np.array([30., 30., 30.]), damping_ratio=1.8)
    just_touching_z = pad_top_z + TOOL_RADIUS
    target = np.array([xy[0], xy[1], just_touching_z - fixed_penetration])

    hover = np.array([xy[0], xy[1], just_touching_z + 0.05])
    q_hover = world.arm.inverse_kinematics(hover, world.nominal_quat)
    world.arm.set_joint_positions(q_hover)
    world.arm.sync_tool_pose()

    traj = MinJerkTrajectory(hover, target, APPROACH_DURATION)
    dashboard = LiveDashboard(mode="material") if live else None
    logger = TrialLogger()
    t = 0.0
    for step in range(int((APPROACH_DURATION + settle_duration) / DT)):
        pos_des, _ = traj.eval(t)
        tau, _ = ctrl.step(pos_des, world.nominal_quat, gains["K_trans"], gains["K_rot"], gains["damping_ratio"])
        world.arm.apply_joint_torques(tau)
        p.stepSimulation(physicsClientId=world.cid)
        world.arm.clamp_joint_velocities()

        tool_pos, _, _, _ = world.arm.get_tool_pose()
        force, torque, _ = get_contact_wrench(world.cid, world.arm.tool_id, [pad_id], tool_pos)
        
        depth = max(0.0, just_touching_z - tool_pos[2])
        logger.record(t, tool_pos, pos_des, force=force, force_des=0.0, torque=torque)
        if dashboard:
            dashboard.update_material(depth, force[2])
        t += DT

    arr = logger.arrays()
    forces = arr["force"][:, 2]
    depths = np.maximum(0.0, just_touching_z - arr["pos_des"][:, 2])

    # --------------------------------------------------------------------------
    # TODO 5: Estimate Contact Stiffness
    # Estimate contact stiffness K_hat via least-squares fit (F = K * depth).
    # --------------------------------------------------------------------------
    raise NotImplementedError("TODO 5: Calculate the stiffness by finding the slope of the force vs. depth data.")
    K_hat = 0.0

    if dashboard:
        dashboard.update_material(depths[-1], forces[-1], k_hat=K_hat)
        time.sleep(1)
        dashboard.close()

    os.makedirs(save_dir, exist_ok=True)
    steady_force = float(np.percentile(forces[-max(int(2.0/DT), 5):], 75))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(arr["t"], forces, color=RED, lw=1.8)
    ax.axhline(steady_force, color=GREY, ls="--", lw=1.2, label=f"steady state: F={steady_force:.1f} N, K̂={K_hat:.0f} N/m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Contact force (N)")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Material Probe -- {tag}", fontsize=12, weight="bold", color=NAVY)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{tag}_loading_curve.png"), dpi=180, facecolor="white")
    plt.close(fig)
    return K_hat


# ==============================================================================
# MAIN EXECUTIONS
# ==============================================================================

def run_part1(save_dir="outputs/part1"):
    print("--- Part 1: Trajectory Generation ---")
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    shapes = ["circle", "spiral", "infinity", "heart"]
    for i, shape in enumerate(shapes):
        strokes = generate_pattern(shape, scale=0.1)
        ax = axes[i]
        for s in strokes:
            pts = np.array(s)
            ax.plot(pts[:, 0], pts[:, 1], 'b-', lw=2)
        ax.set_aspect("equal")
        ax.set_title(shape.capitalize())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shapes.png"), dpi=150)
    plt.close()

def run_part2_writing(gui=False, save_dir="outputs/part3", f_desired=F_DESIRED, word="Robot", pattern="circle", live=False):
    print("--- Part 2 (continued): Impedance Writing ---")
    world = build_world(gui=gui)
    ctrl = CartesianImpedanceController(world.arm)
    s, _ = run_writing_task(world, ctrl, "impedance", f_desired, gui=gui, save_dir=save_dir, tag="part3_impedance", word=word, pattern=pattern, live=live)
    world.disconnect()
    return s

def run_part2_writing_admittance(gui=False, save_dir="outputs/part4", f_desired=F_DESIRED, word="Robot", pattern="circle", live=False):
    print("--- Part 3: Admittance Writing ---")
    world = build_world(gui=gui)
    ctrl = CartesianImpedanceController(world.arm)
    s, _ = run_writing_task(world, ctrl, "admittance", f_desired, gui=gui, save_dir=save_dir, tag="part4_admittance", word=word, pattern=pattern, live=live)
    world.disconnect()
    return s

def run_part3(save_dir="outputs/part5"):
    print("--- Part 4: Analysis and Comparison ---")
    os.makedirs(save_dir, exist_ok=True)
    s3 = run_part2_writing(gui=False)
    s4 = run_part3_admittance(gui=False)
    
    dynamic_cmp = {"Impedance (Part 2)": s3, "Admittance (Part 3)": s4}
    bar_compare(["Force RMSE (N)"], {k: [v["force_tracking_rmse_N"]] for k, v in dynamic_cmp.items()}, "RMSE (N)", os.path.join(save_dir, "part5_writing_force_rmse.png"), "Writing: Force Tracking RMSE")
    bar_compare(["Path Smoothness (mean sq. jerk)"], {k: [v["path_smoothness_jerk2"]] for k, v in dynamic_cmp.items()}, "Mean squared jerk", os.path.join(save_dir, "part5_writing_smoothness.png"), "Writing: Motion Smoothness")

def run_part4(gui=False, save_dir="outputs/part6", live=False):
    print("--- Part 5: Haptic Material Identification ---")
    os.makedirs(save_dir, exist_ok=True)
    world = build_world(gui=gui, with_material_samples=True)
    ctrl = CartesianImpedanceController(world.arm)
    results = {}
    for name, (pad_id, pad_top_z, true_label) in world.material_ids.items():
        spec = MATERIAL_SAMPLES[name]
        xy = world.surface_center_xy + np.array(spec["offset_xy"])
        K_hat = probe_material(world, ctrl, pad_id, pad_top_z, xy, save_dir, name, live=live)
        
        # ----------------------------------------------------------------------
        # TODO 6: Classify Material
        # Classify material as 'soft', 'medium', or 'hard' based on K_hat and MATERIAL_CLASS_THRESHOLDS.
        # ----------------------------------------------------------------------
        raise NotImplementedError("TODO 6: Use the stiffness to classify the material as \'soft\', \'medium\', or \'hard\'.")
        pred_label = "unknown"
            
        results[name] = dict(true_stiffness_N_per_m=spec["contact_stiffness"], estimated_stiffness_N_per_m=K_hat, true_label=true_label, predicted_label=pred_label, correct=(pred_label == true_label))
        print(f"[Part 5] {name}: estimated K̂={K_hat:.0f} N/m -> predicted={pred_label}")
    n_correct = sum(1 for r in results.values() if r["correct"])
    save_json(os.path.join(save_dir, "part6_results.json"), dict(accuracy=n_correct/3.0, results=results))
    world.disconnect()
    return results

def main():
    parser = argparse.ArgumentParser(description="Force-Controlled Writing")
    parser.add_argument("--part", choices=["1", "2", "3", "4", "5", "all"], default="all")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--word", type=str, default="Robot", help="Word to write")
    parser.add_argument("--pattern", type=str, default="circle", help="Pattern to draw")
    parser.add_argument("--live", action="store_true", help="Enable live dashboard visualization")
    args = parser.parse_args()
    
    pattern = None if args.pattern == "none" else args.pattern
    
    if args.part in ["1", "all"]: run_part1()
    if args.part in ["2", "all"]: run_part2_writing(gui=args.gui, word=args.word, pattern=pattern, live=args.live)
    if args.part in ["3", "all"]: run_part3(gui=args.gui, word=args.word, pattern=pattern, live=args.live)
    if args.part in ["4", "all"]: run_part4()
    if args.part in ["5", "all"]: run_part5(gui=args.gui, live=args.live)

if __name__ == '__main__':
    main()
