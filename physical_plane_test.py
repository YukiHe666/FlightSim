import math
import re
import numpy as np
import pandas as pd
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *


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
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def quat_to_rotmat(q):
    q = quat_normalize(q)
    w, x, y, z = q

    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
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
            cl_csv_path="CL&α_Polar_Graph.csv",
            cd_csv_path="Cd&α_Polar_Graph.csv"
        )

        self.parts = []

        # Body frame:
        # +x = nose direction
        # +y = aircraft right
        # +z = aircraft upward
        #
        # Origin = connection point between fuselage and rod

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

        self.left_tail = AircraftPart(
            name="left_tail",
            mass_kg=0.1,
            r_body=[-0.90, -0.18, 0.02],
            area_m2=0.030,
            chord_m=0.12,
            span_m=0.32,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[0, -1, 0],
            is_lifting_surface=True,
            cd0=0.04
        )

        self.right_tail = AircraftPart(
            name="right_tail",
            mass_kg=0.1,
            r_body=[-0.90, 0.18, 0.02],
            area_m2=0.030,
            chord_m=0.12,
            span_m=0.32,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[0, 1, 0],
            is_lifting_surface=True,
            cd0=0.04
        )

        self.parts.append(self.left_wing)
        self.parts.append(self.right_wing)
        self.parts.append(self.left_tail)
        self.parts.append(self.right_tail)

        self.mass = sum(p.mass for p in self.parts)

        self.pos = np.array([0.0, 0.0, 20.0], dtype=float)
        self.vel = np.array([18.0, 0.0, 0.0], dtype=float)

        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.omega_body = np.array([0.0, 0.0, 0.0], dtype=float)

        self.throttle = 0.45
        self.max_thrust = 12.0

        self.I_body = self.compute_inertia_tensor()
        self.I_body_inv = np.linalg.inv(self.I_body)

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

    def compute_inertia_tensor(self):
        I = np.zeros((3, 3), dtype=float)

        for p in self.parts:
            r = p.r_body
            I += p.mass * ((np.dot(r, r) * np.eye(3)) - np.outer(r, r))

        I += np.diag([0.02, 0.08, 0.08])

        return I

    def apply_controls(self, keys, dt):
        # Reduced sensitivity
        flap_rate = 30.0 * DEG2RAD
        max_flap = 12.0 * DEG2RAD

        left_cmd = 0.0
        right_cmd = 0.0

        # W: both wings down
        if keys[K_w]:
            left_cmd += flap_rate
            right_cmd += flap_rate

        # S: both wings up
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

        # Faster return to neutral when no key is pressed
        if not (keys[K_w] or keys[K_s] or keys[K_a] or keys[K_d]):
            self.left_wing.deflection *= 0.95
            self.right_wing.deflection *= 0.95

    def compute_part_forces(self, part, R):
        r_world = R @ part.r_body
        omega_world = R @ self.omega_body

        v_part_world = self.vel + np.cross(omega_world, r_world)
        speed = np.linalg.norm(v_part_world)

        if speed > 0.1:
            v_hat_world = v_part_world / speed
        else:
            v_hat_world = np.zeros(3)

        # Gravity
        F_gravity_world = np.array([0.0, 0.0, -part.mass * G])

        # Basic drag for every part
        F_drag_world = np.zeros(3)
        if speed > 0.1:
            q_dyn = 0.5 * RHO * speed * speed
            F_drag_world = -q_dyn * part.cd0 * part.area * v_hat_world

        F_lift_world = np.zeros(3)

        if part.is_lifting_surface and speed > 0.1:
            v_body = R.T @ v_part_world

            vx = v_body[0]
            vz = v_body[2]

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

            F_drag_world = -q_dyn * CD * part.area * v_hat_world

            span_world = R @ part.span_axis_body
            lift_dir_world = np.cross(span_world, v_hat_world)
            lift_dir_world = norm(lift_dir_world)

            body_up_world = R @ np.array([0.0, 0.0, 1.0])

            if np.dot(lift_dir_world, body_up_world) < 0:
                lift_dir_world = -lift_dir_world

            F_lift_world = q_dyn * CL * part.area * lift_dir_world

        F_total_world = F_gravity_world + F_drag_world + F_lift_world

        F_body = R.T @ F_total_world
        tau_body = np.cross(part.r_body, F_body)

        return F_total_world, tau_body

    def compute_total_forces_and_torques(self):
        R = quat_to_rotmat(self.q)

        F_total_world = np.zeros(3)
        tau_total_body = np.zeros(3)

        for part in self.parts:
            F_world, tau_body = self.compute_part_forces(part, R)
            F_total_world += F_world
            tau_total_body += tau_body

        # Thrust along body +x
        thrust_body = np.array([self.max_thrust * self.throttle, 0.0, 0.0])
        thrust_world = R @ thrust_body

        thrust_pos_body = np.array([0.35, 0.0, 0.0])
        thrust_tau_body = np.cross(thrust_pos_body, thrust_body)

        F_total_world += thrust_world
        tau_total_body += thrust_tau_body

        # Increased angular damping
        tau_total_body += -0.45 * self.omega_body

        return F_total_world, tau_total_body

    def step(self, dt):
        F_world, tau_body = self.compute_total_forces_and_torques()

        # Linear dynamics
        acc_world = F_world / self.mass

        self.vel += acc_world * dt
        self.pos += self.vel * dt

        # Simple ground collision
        if self.pos[2] < 0.3:
            self.pos[2] = 0.3

            if self.vel[2] < 0:
                self.vel[2] *= -0.15

            self.vel[0] *= 0.96
            self.vel[1] *= 0.96

        # Rotational dynamics
        I = self.I_body
        Iomega = I @ self.omega_body

        omega_dot = self.I_body_inv @ (
            tau_body - np.cross(self.omega_body, Iomega)
        )

        self.omega_body += omega_dot * dt

        # Reduced maximum angular velocity
        max_omega = 3.0
        wmag = np.linalg.norm(self.omega_body)

        if wmag > max_omega:
            self.omega_body = self.omega_body / wmag * max_omega

        self.q = integrate_quaternion(self.q, self.omega_body, dt)


# ============================================================
# OpenGL drawing
# ============================================================

def draw_box(size):
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2

    vertices = [
        [-sx, -sy, -sz], [ sx, -sy, -sz], [ sx,  sy, -sz], [-sx,  sy, -sz],
        [-sx, -sy,  sz], [ sx, -sy,  sz], [ sx,  sy,  sz], [-sx,  sy,  sz]
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

    # Origin marker
    glColor3f(1.0, 1.0, 0.0)
    draw_box([0.05, 0.05, 0.05])

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
            glRotatef(p.deflection * RAD2DEG, 0, 1, 0)
            glColor3f(0.2, 0.9, 0.4)
            draw_wing(span=0.32, chord=0.12)

        elif p.name == "right_tail":
            glRotatef(p.deflection * RAD2DEG, 0, 1, 0)
            glColor3f(0.2, 0.9, 0.4)
            draw_wing(span=0.32, chord=0.12)

        glPopMatrix()

    glPopMatrix()


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
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    R = quat_to_rotmat(aircraft.q)

    forward = R @ np.array([1.0, 0.0, 0.0])
    up = R @ np.array([0.0, 0.0, 1.0])

    target = aircraft.pos
    cam_pos = aircraft.pos - forward * 5.0 + np.array([0.0, 0.0, 2.0])

    gluLookAt(
        cam_pos[0], cam_pos[1], cam_pos[2],
        target[0], target[1], target[2],
        up[0], up[1], up[2]
    )


# ============================================================
# Main loop
# ============================================================

def main():
    pygame.init()

    width, height = 1200, 800
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Flow5 Table Driven Aircraft Simulation")

    setup_opengl(width, height)

    clock = pygame.time.Clock()
    aircraft = Aircraft()

    running = True
    accumulator = 0.0

    while running:
        frame_dt = clock.tick(60) / 1000.0
        accumulator += frame_dt

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

                if event.key == K_r:
                    aircraft.reset()

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

        R = quat_to_rotmat(aircraft.q)
        roll, pitch, yaw = euler_from_rotmat(R)
        speed = np.linalg.norm(aircraft.vel)

        pygame.display.set_caption(
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