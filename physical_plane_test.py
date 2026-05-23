import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *


# ============================================================
# File path
# ============================================================

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


# ============================================================
# Constants
# ============================================================

G = 9.81
RHO = 1.225
MU_AIR = 1.81e-5
DT = 0.005

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


# ============================================================
# Force arrow visualization
# ============================================================

DRAW_FORCE_ARROWS = True

FORCE_ARROW_SCALE = 0.15
MAX_FORCE_ARROW_LEN = 0.80


# ============================================================
# HUD display
# ============================================================

DRAW_FORCE_HUD = True
HUD_FONT_SIZE = 18
HUD_LINE_SPACING = 24


# ============================================================
# V-tail geometry
# ============================================================

TAIL_ANGLE_DEG = 30.0

TAIL_SPAN_M = 0.32
TAIL_HALF_SPAN_M = TAIL_SPAN_M / 2.0

TAIL_ROOT_X = -0.90
TAIL_ROOT_Z = 0.02


# ============================================================
# V-tail comparison printout
# ============================================================

def print_vtail_angle_comparison():
    print("\n" + "=" * 88)
    print("V-TAIL ANGLE COMPARISON: 30 deg vs 45 deg")
    print(f"Initial simulation tail angle = {TAIL_ANGLE_DEG:.1f} deg")
    print("=" * 88)

    print(
        f"{'Angle':>8} | "
        f"{'center_y(m)':>11} | "
        f"{'center_z(m)':>11} | "
        f"{'vertical lift frac':>18} | "
        f"{'side force frac':>15} | "
        f"{'pitch/yaw ratio':>15}"
    )
    print("-" * 88)

    for angle_deg in [30.0, 45.0]:
        angle_rad = angle_deg * DEG2RAD

        center_y = math.cos(angle_rad) * TAIL_HALF_SPAN_M
        center_z = TAIL_ROOT_Z + math.sin(angle_rad) * TAIL_HALF_SPAN_M

        vertical_lift_fraction = math.cos(angle_rad)
        side_force_fraction = math.sin(angle_rad)

        if side_force_fraction > 1e-9:
            pitch_yaw_ratio = vertical_lift_fraction / side_force_fraction
        else:
            pitch_yaw_ratio = float("inf")

        print(
            f"{angle_deg:8.1f} | "
            f"{center_y:11.4f} | "
            f"{center_z:11.4f} | "
            f"{vertical_lift_fraction:18.3f} | "
            f"{side_force_fraction:15.3f} | "
            f"{pitch_yaw_ratio:15.3f}"
        )

    print("-" * 88)
    print("Interpretation:")
    print("30 deg: more horizontal-tail-like. Stronger pitch stability.")
    print("45 deg: balanced pitch and yaw contribution. More standard V-tail-like.")
    print()
    print("OpenGL controls:")
    print("Press P: pause / resume")
    print("Press 3: switch V-tail to 30 degrees")
    print("Press 4: switch V-tail to 45 degrees")
    print("Press F: show/hide force arrows")
    print("Press R: reset aircraft state")
    print("W/S: both main wings down/up")
    print("A/D: left/right turn input")
    print("UP/DOWN: throttle")
    print("=" * 88 + "\n")


# ============================================================
# Vector helpers
# ============================================================

def norm(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(3)
    return v / n


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ============================================================
# Quaternion helpers
# q = [w, x, y, z]
# ============================================================

def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=float)


def quat_to_rotmat(q):
    q = quat_normalize(q)
    w, x, y, z = q

    return np.array([
        [1 - 2 * y * y - 2 * z * z,     2 * x * y - 2 * z * w,       2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w,         1 - 2 * x * x - 2 * z * z,   2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w,         2 * y * z + 2 * x * w,       1 - 2 * x * x - 2 * y * y]
    ], dtype=float)


def integrate_quaternion(q, omega_body, dt):
    omega_q = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    q_dot = 0.5 * quat_multiply(q, omega_q)
    return quat_normalize(q + q_dot * dt)


def euler_from_rotmat(R):
    pitch = math.asin(clamp(-R[2, 0], -1.0, 1.0))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])

    return roll * RAD2DEG, pitch * RAD2DEG, yaw * RAD2DEG


# ============================================================
# Flow5 polar table reader
# ============================================================

class Flow5PolarTable:
    def __init__(self, cl_csv_path, cd_csv_path):
        self.CL_data = self._load_one_table(cl_csv_path)
        self.CD_data = self._load_one_table(cd_csv_path)

        self.re_values = sorted(
            set(self.CL_data.keys()).intersection(set(self.CD_data.keys()))
        )

        if len(self.re_values) == 0:
            raise ValueError("No matching Reynolds number found between CL and CD tables.")

        print("Loaded Flow5 polar data:")
        for Re in self.re_values:
            print(f"  Re = {Re:.0f}")

    def _extract_re_from_name(self, name):
        match = re.search(r"Re([0-9.]+)", str(name))
        if match is None:
            return None

        re_million = float(match.group(1))
        return re_million * 1_000_000.0

    def _load_one_table(self, csv_path):
        df = pd.read_csv(csv_path)
        cols = list(df.columns)

        data = {}

        for i in range(0, len(cols) - 1, 2):
            alpha_col = cols[i]
            value_col = cols[i + 1]

            Re = self._extract_re_from_name(value_col)
            if Re is None:
                continue

            alpha = pd.to_numeric(df[alpha_col], errors="coerce").to_numpy()
            value = pd.to_numeric(df[value_col], errors="coerce").to_numpy()

            valid = np.isfinite(alpha) & np.isfinite(value)
            alpha = alpha[valid]
            value = value[valid]

            if len(alpha) < 2:
                continue

            order = np.argsort(alpha)
            alpha = alpha[order]
            value = value[order]

            data[Re] = {
                "alpha": alpha,
                "value": value
            }

        return data

    def _lookup_single_table(self, table, Re, alpha_deg):
        Re = float(Re)

        if Re <= self.re_values[0]:
            Re_low = self.re_values[0]
            Re_high = self.re_values[0]
        elif Re >= self.re_values[-1]:
            Re_low = self.re_values[-1]
            Re_high = self.re_values[-1]
        else:
            Re_low = self.re_values[0]
            Re_high = self.re_values[-1]

            for i in range(len(self.re_values) - 1):
                if self.re_values[i] <= Re <= self.re_values[i + 1]:
                    Re_low = self.re_values[i]
                    Re_high = self.re_values[i + 1]
                    break

        def interp_alpha(Re_table):
            alpha_table = table[Re_table]["alpha"]
            value_table = table[Re_table]["value"]

            alpha_used = np.clip(alpha_deg, alpha_table[0], alpha_table[-1])
            return np.interp(alpha_used, alpha_table, value_table)

        value_low = interp_alpha(Re_low)
        value_high = interp_alpha(Re_high)

        if Re_low == Re_high:
            return float(value_low)

        t = (Re - Re_low) / (Re_high - Re_low)
        return float(value_low + t * (value_high - value_low))

    def lookup(self, Re, alpha_deg):
        CL = self._lookup_single_table(self.CL_data, Re, alpha_deg)
        CD = self._lookup_single_table(self.CD_data, Re, alpha_deg)
        return CL, CD


# ============================================================
# Aircraft component
# ============================================================

class AircraftPart:
    def __init__(
        self,
        name,
        mass_kg,
        r_body,
        area_m2,
        chord_m,
        span_m,
        lift_axis_body,
        span_axis_body,
        is_lifting_surface=True,
        cd0=0.04
    ):
        self.name = name
        self.mass = mass_kg

        self.r_body = np.array(r_body, dtype=float)

        self.area = area_m2
        self.chord = chord_m
        self.span = span_m

        if area_m2 > 1e-9:
            self.aspect_ratio = span_m * span_m / area_m2
        else:
            self.aspect_ratio = 1.0

        self.lift_axis_body = norm(np.array(lift_axis_body, dtype=float))
        self.span_axis_body = norm(np.array(span_axis_body, dtype=float))

        self.is_lifting_surface = is_lifting_surface
        self.cd0 = cd0

        # Positive deflection means trailing edge down.
        self.deflection = 0.0


# ============================================================
# Aircraft rigid body
# ============================================================

class Aircraft:
    def __init__(self):
        self.airfoil_table = Flow5PolarTable(
            cl_csv_path=str(BASE_DIR / "CL&α_Polar_Graph.csv"),
            cd_csv_path=str(BASE_DIR / "Cd&α_Polar_Graph.csv")
        )

        self.parts = []
        self.force_debug = []

        self.tail_angle_deg = TAIL_ANGLE_DEG
        self.tail_angle_rad = self.tail_angle_deg * DEG2RAD

        self.parts.append(AircraftPart(
            name="fuselage",
            mass_kg=2.0,
            r_body=[0.25, 0.0, 0.0],
            area_m2=0.035,
            chord_m=0.50,
            span_m=0.12,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[0, 1, 0],
            is_lifting_surface=False,
            cd0=0.12
        ))

        self.parts.append(AircraftPart(
            name="rod",
            mass_kg=0.1,
            r_body=[-0.45, 0.0, 0.0],
            area_m2=0.010,
            chord_m=0.90,
            span_m=0.035,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[0, 1, 0],
            is_lifting_surface=False,
            cd0=0.08
        ))

        self.left_wing = AircraftPart(
            name="left_wing",
            mass_kg=0.25,
            r_body=[0.05, -0.35, 0.0],
            area_m2=0.075,
            chord_m=0.18,
            span_m=0.65,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[0, -1, 0],
            is_lifting_surface=True,
            cd0=0.035
        )

        self.right_wing = AircraftPart(
            name="right_wing",
            mass_kg=0.25,
            r_body=[0.05, 0.35, 0.0],
            area_m2=0.075,
            chord_m=0.18,
            span_m=0.65,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[0, 1, 0],
            is_lifting_surface=True,
            cd0=0.035
        )

        tail_center_y = math.cos(self.tail_angle_rad) * TAIL_HALF_SPAN_M
        tail_center_z = TAIL_ROOT_Z + math.sin(self.tail_angle_rad) * TAIL_HALF_SPAN_M

        self.left_tail = AircraftPart(
            name="left_tail",
            mass_kg=0.1,
            r_body=[
                TAIL_ROOT_X,
                -tail_center_y,
                tail_center_z
            ],
            area_m2=0.030,
            chord_m=0.12,
            span_m=TAIL_SPAN_M,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[
                0,
                -math.cos(self.tail_angle_rad),
                math.sin(self.tail_angle_rad)
            ],
            is_lifting_surface=True,
            cd0=0.04
        )

        self.right_tail = AircraftPart(
            name="right_tail",
            mass_kg=0.1,
            r_body=[
                TAIL_ROOT_X,
                tail_center_y,
                tail_center_z
            ],
            area_m2=0.030,
            chord_m=0.12,
            span_m=TAIL_SPAN_M,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[
                0,
                math.cos(self.tail_angle_rad),
                math.sin(self.tail_angle_rad)
            ],
            is_lifting_surface=True,
            cd0=0.04
        )

        self.parts.append(self.left_wing)
        self.parts.append(self.right_wing)
        self.parts.append(self.left_tail)
        self.parts.append(self.right_tail)

        self.mass = sum(p.mass for p in self.parts)
        self.cg_body = self.compute_center_of_mass_body()

        self.pos = np.array([0.0, 0.0, 20.0], dtype=float)
        self.vel = np.array([18.0, 0.0, 0.0], dtype=float)

        # Still air by default.
        self.wind_world = np.array([0.0, 0.0, 0.0], dtype=float)

        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.omega_body = np.array([0.0, 0.0, 0.0], dtype=float)

        self.throttle = 0.45
        self.max_thrust = 12.0

        self.I_body = self.compute_inertia_tensor()
        self.I_body_inv = np.linalg.inv(self.I_body)

        self.print_current_tail_angle_data()

    def compute_center_of_mass_body(self):
        weighted_sum = np.zeros(3, dtype=float)

        for p in self.parts:
            weighted_sum += p.mass * p.r_body

        return weighted_sum / self.mass

    def compute_inertia_tensor(self):
        I = np.zeros((3, 3), dtype=float)

        # Inertia around center of gravity, not body origin.
        for p in self.parts:
            r = p.r_body - self.cg_body
            I += p.mass * ((np.dot(r, r) * np.eye(3)) - np.outer(r, r))

        # Small base inertia for numerical stability.
        I += np.diag([0.02, 0.08, 0.08])

        return I

    def add_force_debug(self, name, part_name, start_world, force_world, color):
        """
        Store exact force used by physics solver.

        start_world and force_world are both WORLD coordinates.
        """

        if np.linalg.norm(force_world) < 1e-8:
            return

        self.force_debug.append({
            "name": name,
            "part_name": part_name,
            "start_world": start_world.copy(),
            "force_world": force_world.copy(),
            "color": color
        })

    def print_current_tail_angle_data(self):
        vertical_lift_fraction = math.cos(self.tail_angle_rad)
        side_force_fraction = math.sin(self.tail_angle_rad)

        if side_force_fraction > 1e-9:
            pitch_yaw_ratio = vertical_lift_fraction / side_force_fraction
        else:
            pitch_yaw_ratio = float("inf")

        print("\n" + "=" * 64)
        print(f"Current V-tail angle = {self.tail_angle_deg:.1f} deg")
        print(f"left_tail center  = {self.left_tail.r_body}")
        print(f"right_tail center = {self.right_tail.r_body}")
        print(f"center of gravity = {self.cg_body}")
        print(f"vertical lift fraction ≈ {vertical_lift_fraction:.3f}")
        print(f"side force fraction    ≈ {side_force_fraction:.3f}")
        print(f"pitch/yaw ratio        ≈ {pitch_yaw_ratio:.3f}")
        print("=" * 64 + "\n")

    def set_tail_angle(self, angle_deg):
        self.tail_angle_deg = angle_deg
        self.tail_angle_rad = angle_deg * DEG2RAD

        tail_center_y = math.cos(self.tail_angle_rad) * TAIL_HALF_SPAN_M
        tail_center_z = TAIL_ROOT_Z + math.sin(self.tail_angle_rad) * TAIL_HALF_SPAN_M

        self.left_tail.r_body = np.array([
            TAIL_ROOT_X,
            -tail_center_y,
            tail_center_z
        ], dtype=float)

        self.right_tail.r_body = np.array([
            TAIL_ROOT_X,
            tail_center_y,
            tail_center_z
        ], dtype=float)

        self.left_tail.span_axis_body = norm(np.array([
            0,
            -math.cos(self.tail_angle_rad),
            math.sin(self.tail_angle_rad)
        ], dtype=float))

        self.right_tail.span_axis_body = norm(np.array([
            0,
            math.cos(self.tail_angle_rad),
            math.sin(self.tail_angle_rad)
        ], dtype=float))

        self.cg_body = self.compute_center_of_mass_body()
        self.I_body = self.compute_inertia_tensor()
        self.I_body_inv = np.linalg.inv(self.I_body)

        self.print_current_tail_angle_data()

    def reset(self):
        self.pos = np.array([0.0, 0.0, 20.0], dtype=float)
        self.vel = np.array([18.0, 0.0, 0.0], dtype=float)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.omega_body = np.array([0.0, 0.0, 0.0], dtype=float)

        self.left_wing.deflection = 0.0
        self.right_wing.deflection = 0.0
        self.left_tail.deflection = 0.0
        self.right_tail.deflection = 0.0

        self.throttle = 0.45

    def apply_controls(self, keys, dt):
        flap_rate = 30.0 * DEG2RAD
        max_flap = 12.0 * DEG2RAD

        left_cmd = 0.0
        right_cmd = 0.0

        if keys[K_w]:
            left_cmd += flap_rate
            right_cmd += flap_rate

        if keys[K_s]:
            left_cmd -= flap_rate
            right_cmd -= flap_rate

        if keys[K_a]:
            left_cmd += flap_rate
            right_cmd -= flap_rate

        if keys[K_d]:
            left_cmd -= flap_rate
            right_cmd += flap_rate

        self.left_wing.deflection += left_cmd * dt
        self.right_wing.deflection += right_cmd * dt

        self.left_wing.deflection = clamp(self.left_wing.deflection, -max_flap, max_flap)
        self.right_wing.deflection = clamp(self.right_wing.deflection, -max_flap, max_flap)

        self.left_tail.deflection *= 0.98
        self.right_tail.deflection *= 0.98

        if keys[K_UP]:
            self.throttle += 0.5 * dt
        if keys[K_DOWN]:
            self.throttle -= 0.5 * dt

        self.throttle = clamp(self.throttle, 0.0, 1.0)

        if not (keys[K_w] or keys[K_s] or keys[K_a] or keys[K_d]):
            self.left_wing.deflection *= 0.95
            self.right_wing.deflection *= 0.95

    def compute_part_forces(self, part, R):
        """
        Compute actual forces acting on one aircraft part.

        All force arrows are stored and displayed in WORLD coordinates.

        Forces:
            gravity: world -Z
            drag: opposite local relative air velocity
            lift: perpendicular to local relative velocity and span
        """

        r_body_from_origin = part.r_body
        r_body_from_cg = part.r_body - self.cg_body

        r_world_from_origin = R @ r_body_from_origin
        r_world_from_cg = R @ r_body_from_cg

        part_pos_world = self.pos + r_world_from_origin

        omega_world = R @ self.omega_body

        # Local part velocity = translation + rotation around CG.
        v_part_world = self.vel + np.cross(omega_world, r_world_from_cg)

        # Velocity relative to surrounding air.
        v_rel_world = v_part_world - self.wind_world
        speed = np.linalg.norm(v_rel_world)

        if speed > 0.1:
            v_rel_hat_world = v_rel_world / speed
        else:
            v_rel_hat_world = np.zeros(3)

        # =====================================================
        # Gravity
        # =====================================================

        F_gravity_world = np.array([0.0, 0.0, -part.mass * G])

        self.add_force_debug(
            name="gravity",
            part_name=part.name,
            start_world=part_pos_world,
            force_world=F_gravity_world,
            color=(1.0, 0.1, 0.1)
        )

        # =====================================================
        # Drag
        # =====================================================

        F_drag_world = np.zeros(3)

        if speed > 0.1:
            q_dyn = 0.5 * RHO * speed * speed
            F_drag_world = -q_dyn * part.cd0 * part.area * v_rel_hat_world

        # =====================================================
        # Lift
        # =====================================================

        F_lift_world = np.zeros(3)

        if part.is_lifting_surface and speed > 0.1:
            v_rel_body = R.T @ v_rel_world

            vx = v_rel_body[0]
            vz = v_rel_body[2]

            alpha = math.atan2(-vz, max(abs(vx), 1e-3))

            alpha_eff = alpha + part.deflection
            alpha_eff_deg = alpha_eff * RAD2DEG

            Re = RHO * speed * part.chord / MU_AIR

            CL_2D, CD_2D = self.airfoil_table.lookup(Re, alpha_eff_deg)

            AR = max(part.aspect_ratio, 0.1)
            e = 0.75

            CL = CL_2D / (1.0 + abs(CL_2D) / (math.pi * e * AR))
            CD_induced = CL * CL / (math.pi * e * AR)
            CD = CD_2D + CD_induced

            q_dyn = 0.5 * RHO * speed * speed

            F_drag_world = -q_dyn * CD * part.area * v_rel_hat_world

            span_world = R @ part.span_axis_body

            lift_dir_world = np.cross(v_rel_hat_world, span_world)
            lift_dir_world = norm(lift_dir_world)

            body_up_world = R @ np.array([0.0, 0.0, 1.0])

            if np.dot(lift_dir_world, body_up_world) < 0:
                lift_dir_world = -lift_dir_world

            F_lift_world = q_dyn * CL * part.area * lift_dir_world

        self.add_force_debug(
            name="drag",
            part_name=part.name,
            start_world=part_pos_world + np.array([-0.04, 0.0, 0.0]),
            force_world=F_drag_world,
            color=(1.0, 0.0, 1.0)
        )

        self.add_force_debug(
            name="lift",
            part_name=part.name,
            start_world=part_pos_world + np.array([0.04, 0.0, 0.0]),
            force_world=F_lift_world,
            color=(0.1, 1.0, 0.1)
        )

        F_total_world = F_gravity_world + F_drag_world + F_lift_world

        # Convert to body frame only for rotational dynamics.
        F_total_body = R.T @ F_total_world
        tau_body = np.cross(r_body_from_cg, F_total_body)

        return F_total_world, tau_body

    def compute_total_forces_and_torques(self):
        R = quat_to_rotmat(self.q)

        F_total_world = np.zeros(3)
        tau_total_body = np.zeros(3)

        self.force_debug = []

        for part in self.parts:
            F_world, tau_body = self.compute_part_forces(part, R)
            F_total_world += F_world
            tau_total_body += tau_body

        # =====================================================
        # Thrust
        # =====================================================

        thrust_body = np.array([
            self.max_thrust * self.throttle,
            0.0,
            0.0
        ])

        thrust_world = R @ thrust_body

        thrust_pos_body = np.array([0.35, 0.0, 0.0])
        thrust_pos_world = self.pos + R @ thrust_pos_body

        thrust_tau_body = np.cross(
            thrust_pos_body - self.cg_body,
            thrust_body
        )

        F_total_world += thrust_world
        tau_total_body += thrust_tau_body

        self.add_force_debug(
            name="thrust",
            part_name="engine",
            start_world=thrust_pos_world,
            force_world=thrust_world,
            color=(1.0, 0.6, 0.0)
        )

        # Angular damping.
        tau_total_body += -0.45 * self.omega_body

        return F_total_world, tau_total_body

    def step(self, dt):
        F_world, tau_body = self.compute_total_forces_and_torques()

        # Linear dynamics in WORLD frame.
        acc_world = F_world / self.mass

        self.vel += acc_world * dt
        self.pos += self.vel * dt

        if self.pos[2] < 0.3:
            self.pos[2] = 0.3

            if self.vel[2] < 0:
                self.vel[2] *= -0.15

            self.vel[0] *= 0.96
            self.vel[1] *= 0.96

        # Rotational dynamics in BODY frame.
        I = self.I_body
        Iomega = I @ self.omega_body

        omega_dot = self.I_body_inv @ (
            tau_body - np.cross(self.omega_body, Iomega)
        )

        self.omega_body += omega_dot * dt

        max_omega = 3.0
        wmag = np.linalg.norm(self.omega_body)

        if wmag > max_omega:
            self.omega_body = self.omega_body / wmag * max_omega

        self.q = integrate_quaternion(self.q, self.omega_body, dt)


# ============================================================
# OpenGL drawing
# ============================================================

def draw_arrow_world(start_world, force_world, scale, color):
    force_mag = np.linalg.norm(force_world)

    if force_mag < 1e-6:
        return

    direction = force_world / force_mag

    arrow_len = force_mag * scale
    arrow_len = min(arrow_len, MAX_FORCE_ARROW_LEN)

    end_world = start_world + direction * arrow_len

    glColor3f(color[0], color[1], color[2])

    glLineWidth(3.0)
    glBegin(GL_LINES)
    glVertex3f(start_world[0], start_world[1], start_world[2])
    glVertex3f(end_world[0], end_world[1], end_world[2])
    glEnd()
    glLineWidth(1.0)

    head_len = 0.08
    head_width = 0.04

    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    side1 = norm(np.cross(direction, tmp))
    side2 = norm(np.cross(direction, side1))

    base = end_world - direction * head_len

    p1 = base + side1 * head_width
    p2 = base - side1 * head_width
    p3 = base + side2 * head_width
    p4 = base - side2 * head_width

    glBegin(GL_LINES)

    for p in [p1, p2, p3, p4]:
        glVertex3f(end_world[0], end_world[1], end_world[2])
        glVertex3f(p[0], p[1], p[2])

    glEnd()


def draw_force_arrows(aircraft):
    if not DRAW_FORCE_ARROWS:
        return

    for info in aircraft.force_debug:
        draw_arrow_world(
            info["start_world"],
            info["force_world"],
            FORCE_ARROW_SCALE,
            info["color"]
        )


def draw_box(size):
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2

    vertices = [
        [-sx, -sy, -sz],
        [ sx, -sy, -sz],
        [ sx,  sy, -sz],
        [-sx,  sy, -sz],
        [-sx, -sy,  sz],
        [ sx, -sy,  sz],
        [ sx,  sy,  sz],
        [-sx,  sy,  sz]
    ]

    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4]
    ]

    glBegin(GL_QUADS)
    for face in faces:
        for idx in face:
            glVertex3fv(vertices[idx])
    glEnd()


def draw_wing(span, chord, thickness=0.02):
    draw_box([chord, span, thickness])


def draw_aircraft(aircraft):
    R = quat_to_rotmat(aircraft.q)

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = aircraft.pos

    glPushMatrix()
    glMultMatrixf(M.T)

    glColor3f(1.0, 1.0, 0.0)
    draw_box([0.05, 0.05, 0.05])

    glPushMatrix()
    glTranslatef(
        aircraft.cg_body[0],
        aircraft.cg_body[1],
        aircraft.cg_body[2]
    )
    glColor3f(1.0, 0.4, 0.0)
    draw_box([0.045, 0.045, 0.045])
    glPopMatrix()

    glPushMatrix()
    glTranslatef(TAIL_ROOT_X, 0.0, TAIL_ROOT_Z)
    glColor3f(0.8, 0.8, 0.8)
    draw_box([0.08, 0.08, 0.08])
    glPopMatrix()

    for p in aircraft.parts:
        glPushMatrix()
        glTranslatef(p.r_body[0], p.r_body[1], p.r_body[2])

        if p.name == "fuselage":
            glColor3f(0.75, 0.75, 0.75)
            draw_box([0.65, 0.12, 0.12])

        elif p.name == "rod":
            glColor3f(0.55, 0.55, 0.55)
            draw_box([0.90, 0.035, 0.035])

        elif p.name == "left_wing":
            glRotatef(p.deflection * RAD2DEG, 0, 1, 0)
            glColor3f(0.2, 0.4, 1.0)
            draw_wing(span=0.65, chord=0.18)

        elif p.name == "right_wing":
            glRotatef(p.deflection * RAD2DEG, 0, 1, 0)
            glColor3f(0.2, 0.4, 1.0)
            draw_wing(span=0.65, chord=0.18)

        elif p.name == "left_tail":
            if aircraft.tail_angle_deg < 40.0:
                glColor3f(0.2, 0.9, 0.4)
            else:
                glColor3f(1.0, 0.6, 0.1)

            glRotatef(-aircraft.tail_angle_deg, 1, 0, 0)
            glRotatef(p.deflection * RAD2DEG, 0, 1, 0)
            draw_wing(span=TAIL_SPAN_M, chord=0.12)

        elif p.name == "right_tail":
            if aircraft.tail_angle_deg < 40.0:
                glColor3f(0.2, 0.9, 0.4)
            else:
                glColor3f(1.0, 0.6, 0.1)

            glRotatef(aircraft.tail_angle_deg, 1, 0, 0)
            glRotatef(p.deflection * RAD2DEG, 0, 1, 0)
            draw_wing(span=TAIL_SPAN_M, chord=0.12)

        glPopMatrix()

    glPopMatrix()

    # Force arrows are drawn after leaving aircraft transform.
    draw_force_arrows(aircraft)


def draw_ground():
    glColor3f(0.25, 0.25, 0.25)

    glBegin(GL_LINES)
    for i in range(-50, 51):
        glVertex3f(i, -50, 0)
        glVertex3f(i, 50, 0)

        glVertex3f(-50, i, 0)
        glVertex3f(50, i, 0)
    glEnd()


def setup_opengl(width, height):
    glViewport(0, 0, width, height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    gluPerspective(60.0, width / height, 0.1, 1000.0)

    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)

    glClearColor(0.05, 0.07, 0.10, 1.0)


def set_camera(aircraft):
    """
    Camera follows aircraft but uses world-up, not aircraft-up.
    """

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    R = quat_to_rotmat(aircraft.q)
    forward = R @ np.array([1.0, 0.0, 0.0])

    target = aircraft.pos
    cam_pos = aircraft.pos - forward * 5.0 + np.array([0.0, 0.0, 2.0])

    world_up = np.array([0.0, 0.0, 1.0])

    gluLookAt(
        cam_pos[0], cam_pos[1], cam_pos[2],
        target[0], target[1], target[2],
        world_up[0], world_up[1], world_up[2]
    )


# ============================================================
# HUD drawing
# ============================================================

def force_totals_from_debug(aircraft):
    """
    Sum current forces from aircraft.force_debug.
    All forces are already stored in WORLD coordinates.
    """

    totals = {
        "gravity": np.zeros(3),
        "drag": np.zeros(3),
        "lift": np.zeros(3),
        "thrust": np.zeros(3),
    }

    for info in aircraft.force_debug:
        name = info["name"]
        if name in totals:
            totals[name] += info["force_world"]

    totals["total"] = (
        totals["gravity"]
        + totals["drag"]
        + totals["lift"]
        + totals["thrust"]
    )

    return totals


def format_force_line(name, F):
    mag = np.linalg.norm(F)

    return (
        f"{name:<8s} "
        f"Fx={F[0]:8.2f}  "
        f"Fy={F[1]:8.2f}  "
        f"Fz={F[2]:8.2f}  "
        f"|F|={mag:8.2f} N"
    )


def begin_2d_overlay():
    """
    Switch OpenGL into 2D screen-coordinate mode.

    Coordinates:
        x = pixels from left
        y = pixels from top
    """

    viewport = glGetIntegerv(GL_VIEWPORT)
    width = viewport[2]
    height = viewport[3]

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, width, height, 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    return width, height


def end_2d_overlay():
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()

    glMatrixMode(GL_MODELVIEW)


def draw_filled_rect_2d(x, y, w, h, color):
    """
    Draw a solid 2D rectangle.

    color = (r, g, b, a), each from 0.0 to 1.0
    """

    glColor4f(color[0], color[1], color[2], color[3])

    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


def draw_text_2d(x, y, text, font, color=(255, 255, 255)):
    """
    Draw text using an OpenGL texture.

    This avoids the white-rectangle problem caused by glDrawPixels.
    """

    text_surface = font.render(text, True, color)
    text_surface = text_surface.convert_alpha()

    width = text_surface.get_width()
    height = text_surface.get_height()

    text_data = pygame.image.tostring(text_surface, "RGBA", True)

    texture_id = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texture_id)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        text_data
    )

    glEnable(GL_TEXTURE_2D)
    glColor4f(1.0, 1.0, 1.0, 1.0)

    glBegin(GL_QUADS)

    glTexCoord2f(0.0, 1.0)
    glVertex2f(x, y)

    glTexCoord2f(1.0, 1.0)
    glVertex2f(x + width, y)

    glTexCoord2f(1.0, 0.0)
    glVertex2f(x + width, y + height)

    glTexCoord2f(0.0, 0.0)
    glVertex2f(x, y + height)

    glEnd()

    glDisable(GL_TEXTURE_2D)

    glBindTexture(GL_TEXTURE_2D, 0)
    glDeleteTextures([texture_id])


def draw_force_hud(aircraft, font, paused):
    """
    Draw force values only when paused.

    When paused:
        black background box
        white text
    """

    if not DRAW_FORCE_HUD:
        return

    if not paused:
        return

    totals = force_totals_from_debug(aircraft)
    speed = np.linalg.norm(aircraft.vel)

    lines = [
        "PAUSED - WORLD FRAME FORCE ANALYSIS",
        f"speed = {speed:.2f} m/s    altitude = {aircraft.pos[2]:.2f} m    throttle = {aircraft.throttle:.2f}",
        "",
        format_force_line("gravity", totals["gravity"]),
        format_force_line("drag", totals["drag"]),
        format_force_line("lift", totals["lift"]),
        format_force_line("thrust", totals["thrust"]),
        format_force_line("total", totals["total"]),
        "",
        "red=gravity   magenta=drag   green=lift   orange=thrust",
        "P: resume   F: force arrows   R: reset   3/4: tail angle",
    ]

    x = 16
    y = 16
    padding_x = 14
    padding_y = 12

    text_w = 0
    for line in lines:
        if line == "":
            continue
        text_w = max(text_w, font.size(line)[0])

    box_w = text_w + padding_x * 2
    box_h = len(lines) * HUD_LINE_SPACING + padding_y * 2

    begin_2d_overlay()

    draw_filled_rect_2d(
        x,
        y,
        box_w,
        box_h,
        color=(0.0, 0.0, 0.0, 0.88)
    )

    # White border.
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glLineWidth(2.0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x, y)
    glVertex2f(x + box_w, y)
    glVertex2f(x + box_w, y + box_h)
    glVertex2f(x, y + box_h)
    glEnd()
    glLineWidth(1.0)

    text_x = x + padding_x
    text_y = y + padding_y

    for i, line in enumerate(lines):
        if line == "":
            continue

        draw_text_2d(
            text_x,
            text_y + i * HUD_LINE_SPACING,
            line,
            font,
            color=(255, 255, 255)
        )

    end_2d_overlay()


# ============================================================
# Debug print
# ============================================================

def print_force_summary(aircraft, frame_count):
    if frame_count % 60 != 0:
        return

    print("\n" + "-" * 88)
    print("FORCE SUMMARY, WORLD COORDINATES")
    print("-" * 88)

    for info in aircraft.force_debug:
        name = info["name"]
        part_name = info["part_name"]
        F = info["force_world"]
        mag = np.linalg.norm(F)

        print(
            f"{part_name:12s} | {name:8s} | "
            f"F_world = [{F[0]:8.3f}, {F[1]:8.3f}, {F[2]:8.3f}] N | "
            f"|F| = {mag:8.3f} N"
        )

    print("-" * 88)


# ============================================================
# Main loop
# ============================================================

def main():
    global DRAW_FORCE_ARROWS

    print_vtail_angle_comparison()

    pygame.init()
    pygame.font.init()

    width, height = 1200, 800
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Aircraft Simulation: Pause HUD + World Forces")

    setup_opengl(width, height)

    clock = pygame.time.Clock()

    # Monospace font is better for aligned force values.
    hud_font = pygame.font.SysFont("Consolas", HUD_FONT_SIZE)
    if hud_font is None:
        hud_font = pygame.font.Font(None, HUD_FONT_SIZE)

    aircraft = Aircraft()

    # Generate force_debug before first physics step.
    aircraft.compute_total_forces_and_torques()

    running = True
    paused = False
    accumulator = 0.0
    frame_count = 0

    while running:
        frame_dt = clock.tick(60) / 1000.0
        frame_count += 1

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

                if event.key == K_p:
                    paused = not paused
                    print(f"PAUSED = {paused}")
                    accumulator = 0.0

                if event.key == K_r:
                    aircraft.reset()
                    aircraft.compute_total_forces_and_torques()
                    accumulator = 0.0

                if event.key == K_3:
                    aircraft.set_tail_angle(30.0)
                    aircraft.compute_total_forces_and_torques()

                if event.key == K_4:
                    aircraft.set_tail_angle(45.0)
                    aircraft.compute_total_forces_and_torques()

                if event.key == K_f:
                    DRAW_FORCE_ARROWS = not DRAW_FORCE_ARROWS
                    print(f"DRAW_FORCE_ARROWS = {DRAW_FORCE_ARROWS}")

        keys = pygame.key.get_pressed()

        if not paused:
            accumulator += frame_dt

            aircraft.apply_controls(keys, frame_dt)

            while accumulator >= DT:
                aircraft.step(DT)
                accumulator -= DT

        else:
            # Keep arrows and HUD current while paused.
            aircraft.compute_total_forces_and_torques()
            accumulator = 0.0

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        set_camera(aircraft)
        draw_ground()
        draw_aircraft(aircraft)

        # Pause HUD appears only when paused.
        draw_force_hud(aircraft, hud_font, paused)

        pygame.display.flip()

        print_force_summary(aircraft, frame_count)

        R = quat_to_rotmat(aircraft.q)
        roll, pitch, yaw = euler_from_rotmat(R)
        speed = np.linalg.norm(aircraft.vel)

        pause_text = "PAUSED" if paused else "RUNNING"

        pygame.display.set_caption(
            f"{pause_text} | WORLD FORCES + HUD | "
            f"tail={aircraft.tail_angle_deg:4.1f} deg | "
            f"force_arrows={DRAW_FORCE_ARROWS} | "
            f"speed={speed:5.2f} m/s | "
            f"alt={aircraft.pos[2]:5.2f} m | "
            f"thr={aircraft.throttle:4.2f} | "
            f"roll={roll:6.2f} deg | "
            f"pitch={pitch:6.2f} deg | "
            f"yaw={yaw:6.2f} deg | "
            f"L flap={aircraft.left_wing.deflection * RAD2DEG:6.2f} deg | "
            f"R flap={aircraft.right_wing.deflection * RAD2DEG:6.2f} deg"
        )

    pygame.quit()


if __name__ == "__main__":
    main()