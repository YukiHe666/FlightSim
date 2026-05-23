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

# Arrow length scale: meters per Newton in OpenGL world frame.
FORCE_ARROW_SCALE = 0.15

# Limit arrow length to avoid huge arrows.
MAX_FORCE_ARROW_LEN = 0.80


# ============================================================
# V-tail geometry
# ============================================================

# Initial tail angle. Press 3 for 30 deg, press 4 for 45 deg.
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

        # Body frame:
        # +x = nose direction
        # +y = aircraft right
        # +z = aircraft upward
        #
        # World frame:
        # +x = initial forward direction
        # +y = horizontal side direction
        # +z = vertical upward
        #
        # All forces in force_debug are stored in WORLD frame.

        self.parts.append(AircraftPart(
            name="fuselage",
            mass_kg=1.0,
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
        # Drag is opposite to velocity relative to this wind field.
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
        Store the exact force used by the physics solver.

        start_world:
            Application point in WORLD frame.

        force_world:
            Force vector in WORLD frame.

        OpenGL arrows are drawn directly from these world-frame values.
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

        # W: both main wings down
        if keys[K_w]:
            left_cmd += flap_rate
            right_cmd += flap_rate

        # S: both main wings up
        if keys[K_s]:
            left_cmd -= flap_rate
            right_cmd -= flap_rate

        # A: turn left
        # left wing down, right wing up
        if keys[K_a]:
            left_cmd += flap_rate
            right_cmd -= flap_rate

        # D: turn right
        # left wing up, right wing down
        if keys[K_d]:
            left_cmd -= flap_rate
            right_cmd += flap_rate

        self.left_wing.deflection += left_cmd * dt
        self.right_wing.deflection += right_cmd * dt

        self.left_wing.deflection = clamp(self.left_wing.deflection, -max_flap, max_flap)
        self.right_wing.deflection = clamp(self.right_wing.deflection, -max_flap, max_flap)

        # Tail surfaces are passive here.
        self.left_tail.deflection *= 0.98
        self.right_tail.deflection *= 0.98

        # Throttle
        if keys[K_UP]:
            self.throttle += 0.5 * dt
        if keys[K_DOWN]:
            self.throttle -= 0.5 * dt

        self.throttle = clamp(self.throttle, 0.0, 1.0)

        # Return to neutral when no key is pressed.
        if not (keys[K_w] or keys[K_s] or keys[K_a] or keys[K_d]):
            self.left_wing.deflection *= 0.95
            self.right_wing.deflection *= 0.95

    def compute_part_forces(self, part, R):
        """
        Compute actual forces acting on one aircraft part.

        Important:
        All forces here are stored and visualized in WORLD COORDINATES.

        Forces:
            gravity: world -Z
            drag: opposite to local relative air velocity
            lift: perpendicular to local relative velocity and span direction
        """

        r_body_from_origin = part.r_body
        r_body_from_cg = part.r_body - self.cg_body

        r_world_from_origin = R @ r_body_from_origin
        r_world_from_cg = R @ r_body_from_cg

        part_pos_world = self.pos + r_world_from_origin

        omega_world = R @ self.omega_body

        # Local velocity of this part:
        # aircraft translation + rotational velocity around CG.
        v_part_world = self.vel + np.cross(omega_world, r_world_from_cg)

        # Relative velocity between the part and the air.
        # If wind_world = 0, this is just velocity through still air.
        v_rel_world = v_part_world - self.wind_world
        speed = np.linalg.norm(v_rel_world)

        if speed > 0.1:
            v_rel_hat_world = v_rel_world / speed
        else:
            v_rel_hat_world = np.zeros(3)

        # =====================================================
        # 1. Gravity
        # =====================================================

        # Gravity is always vertical downward in WORLD frame.
        # This should ALWAYS be [0, 0, negative].
        F_gravity_world = np.array([0.0, 0.0, -part.mass * G])

        self.add_force_debug(
            name="gravity",
            part_name=part.name,
            start_world=part_pos_world + np.array([0.00, 0.00, 0.00]),
            force_world=F_gravity_world,
            color=(1.0, 0.1, 0.1)
        )

        # =====================================================
        # 2. Drag
        # =====================================================

        F_drag_world = np.zeros(3)

        if speed > 0.1:
            q_dyn = 0.5 * RHO * speed * speed

            # Basic drag for non-lifting parts.
            # Direction is always opposite local relative velocity.
            F_drag_world = -q_dyn * part.cd0 * part.area * v_rel_hat_world

        # =====================================================
        # 3. Lift
        # =====================================================

        F_lift_world = np.zeros(3)

        if part.is_lifting_surface and speed > 0.1:
            v_rel_body = R.T @ v_rel_world

            vx = v_rel_body[0]
            vz = v_rel_body[2]

            # Angle of attack.
            alpha = math.atan2(-vz, max(abs(vx), 1e-3))

            # Control surface deflection changes effective angle of attack.
            alpha_eff = alpha + part.deflection
            alpha_eff_deg = alpha_eff * RAD2DEG

            Re = RHO * speed * part.chord / MU_AIR

            CL_2D, CD_2D = self.airfoil_table.lookup(Re, alpha_eff_deg)

            AR = max(part.aspect_ratio, 0.1)
            e = 0.75

            # Finite-wing correction.
            CL = CL_2D / (1.0 + abs(CL_2D) / (math.pi * e * AR))

            # Induced drag.
            CD_induced = CL * CL / (math.pi * e * AR)
            CD = CD_2D + CD_induced

            q_dyn = 0.5 * RHO * speed * speed

            # For lifting surfaces, replace simple drag with polar-based drag.
            F_drag_world = -q_dyn * CD * part.area * v_rel_hat_world

            # Lift direction in WORLD frame.
            # Lift is perpendicular to velocity and span.
            span_world = R @ part.span_axis_body

            lift_dir_world = np.cross(v_rel_hat_world, span_world)
            lift_dir_world = norm(lift_dir_world)

            # Choose the lift direction closer to aircraft body-up.
            body_up_world = R @ np.array([0.0, 0.0, 1.0])

            if np.dot(lift_dir_world, body_up_world) < 0:
                lift_dir_world = -lift_dir_world

            F_lift_world = q_dyn * CL * part.area * lift_dir_world

        # Record drag after possible polar-based replacement.
        self.add_force_debug(
            name="drag",
            part_name=part.name,
            start_world=part_pos_world + np.array([-0.04, 0.00, 0.00]),
            force_world=F_drag_world,
            color=(1.0, 0.0, 1.0)
        )

        # Record lift.
        self.add_force_debug(
            name="lift",
            part_name=part.name,
            start_world=part_pos_world + np.array([0.04, 0.00, 0.00]),
            force_world=F_lift_world,
            color=(0.1, 1.0, 0.1)
        )

        # =====================================================
        # 4. Total force and torque
        # =====================================================

        F_total_world = F_gravity_world + F_drag_world + F_lift_world

        # Torque about center of gravity.
        #
        # Convert total force to body frame only for rotational dynamics.
        # This does NOT affect the displayed arrows.
        F_total_body = R.T @ F_total_world
        tau_body = np.cross(r_body_from_cg, F_total_body)

        return F_total_world, tau_body

    def compute_total_forces_and_torques(self):
        R = quat_to_rotmat(self.q)

        F_total_world = np.zeros(3)
        tau_total_body = np.zeros(3)

        # Clear debug force list every physics step.
        self.force_debug = []

        # Forces on each aircraft part.
        for part in self.parts:
            F_world, tau_body = self.compute_part_forces(part, R)

            F_total_world += F_world
            tau_total_body += tau_body

        # =====================================================
        # Thrust
        # =====================================================

        # Thrust is defined in body frame along aircraft nose direction.
        thrust_body = np.array([
            self.max_thrust * self.throttle,
            0.0,
            0.0
        ])

        # Convert thrust to WORLD frame for force accumulation and drawing.
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

        # Simple ground collision.
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
    """
    Draw a force arrow directly in WORLD coordinates.

    No aircraft body transform is active when this function is called.
    Therefore:
        gravity [0, 0, -mg] will always point world-down.
    """

    force_mag = np.linalg.norm(force_world)

    if force_mag < 1e-6:
        return

    direction = force_world / force_mag

    arrow_len = force_mag * scale
    arrow_len = min(arrow_len, MAX_FORCE_ARROW_LEN)

    end_world = start_world + direction * arrow_len

    glColor3f(color[0], color[1], color[2])

    # Main line.
    glLineWidth(3.0)
    glBegin(GL_LINES)
    glVertex3f(start_world[0], start_world[1], start_world[2])
    glVertex3f(end_world[0], end_world[1], end_world[2])
    glEnd()
    glLineWidth(1.0)

    # Arrow head.
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
    """
    Draw exact forces used in physics simulation.

    Each arrow comes directly from aircraft.force_debug.

    Everything is WORLD frame:
        start_world
        force_world

    Colors:
        gravity = red
        lift    = green
        drag    = magenta
        thrust  = orange
    """

    if not DRAW_FORCE_ARROWS:
        return

    for info in aircraft.force_debug:
        start_world = info["start_world"]
        force_world = info["force_world"]
        color = info["color"]

        draw_arrow_world(
            start_world,
            force_world,
            FORCE_ARROW_SCALE,
            color=color
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

    # ========================================================
    # Draw aircraft body in aircraft/body coordinates
    # ========================================================

    glPushMatrix()
    glMultMatrixf(M.T)

    # Origin marker.
    glColor3f(1.0, 1.0, 0.0)
    draw_box([0.05, 0.05, 0.05])

    # CG marker.
    glPushMatrix()
    glTranslatef(
        aircraft.cg_body[0],
        aircraft.cg_body[1],
        aircraft.cg_body[2]
    )
    glColor3f(1.0, 0.4, 0.0)
    draw_box([0.045, 0.045, 0.045])
    glPopMatrix()

    # Tail joint / V-tail root connector.
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

    # ========================================================
    # Draw force arrows in WORLD coordinates
    # ========================================================
    #
    # This is deliberately outside the aircraft body transform.
    # Therefore gravity will not rotate with the aircraft.
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
    Camera follows the aircraft position and forward direction,
    but it does NOT roll with the aircraft.

    Important:
    world_up = [0, 0, 1]

    This prevents world-down gravity from visually looking like
    aircraft-backward gravity.
    """

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    R = quat_to_rotmat(aircraft.q)

    forward = R @ np.array([1.0, 0.0, 0.0])

    target = aircraft.pos

    # Follow behind aircraft, but keep camera world-up.
    cam_pos = aircraft.pos - forward * 5.0 + np.array([0.0, 0.0, 2.0])

    world_up = np.array([0.0, 0.0, 1.0])

    gluLookAt(
        cam_pos[0], cam_pos[1], cam_pos[2],
        target[0], target[1], target[2],
        world_up[0], world_up[1], world_up[2]
    )


# ============================================================
# Debug print
# ============================================================

def print_force_summary(aircraft, frame_count):
    """
    Print force magnitude summary occasionally.
    This helps verify that OpenGL arrows match simulated forces.

    Gravity should always print as:
        [0.000, 0.000, negative]
    """

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

    width, height = 1200, 800
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Aircraft Simulation: World-Coordinate Force Arrows")

    setup_opengl(width, height)

    clock = pygame.time.Clock()
    aircraft = Aircraft()

    running = True
    accumulator = 0.0
    frame_count = 0

    while running:
        frame_dt = clock.tick(60) / 1000.0
        accumulator += frame_dt
        frame_count += 1

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

                if event.key == K_r:
                    aircraft.reset()

                if event.key == K_3:
                    aircraft.set_tail_angle(30.0)

                if event.key == K_4:
                    aircraft.set_tail_angle(45.0)

                if event.key == K_f:
                    DRAW_FORCE_ARROWS = not DRAW_FORCE_ARROWS
                    print(f"DRAW_FORCE_ARROWS = {DRAW_FORCE_ARROWS}")

        keys = pygame.key.get_pressed()
        aircraft.apply_controls(keys, frame_dt)

        while accumulator >= DT:
            aircraft.step(DT)
            accumulator -= DT

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        set_camera(aircraft)
        draw_ground()
        draw_aircraft(aircraft)

        pygame.display.flip()

        print_force_summary(aircraft, frame_count)

        R = quat_to_rotmat(aircraft.q)
        roll, pitch, yaw = euler_from_rotmat(R)
        speed = np.linalg.norm(aircraft.vel)

        pygame.display.set_caption(
            f"WORLD FORCE ARROWS | "
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