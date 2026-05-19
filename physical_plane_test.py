import math
import numpy as np
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *


# ============================================================
# Basic constants
# ============================================================

G = 9.81
RHO = 1.225          # air density at sea level, kg/m^3
DT = 0.005          # physics timestep, seconds

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


def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=float)


# ============================================================
# Quaternion helpers
# q = [w, x, y, z]
# ============================================================

def quat_normalize(q):
    return q / np.linalg.norm(q)


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
    """
    omega_body is angular velocity in body frame, rad/s.
    q_dot = 0.5 * q * [0, omega]
    """
    omega_q = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    q_dot = 0.5 * quat_multiply(q, omega_q)
    return quat_normalize(q + q_dot * dt)


def euler_from_rotmat(R):
    """
    Return roll, pitch, yaw in degrees.
    Approximate aerospace convention.
    """
    pitch = math.asin(clamp(-R[2, 0], -1.0, 1.0))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll * RAD2DEG, pitch * RAD2DEG, yaw * RAD2DEG


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
        lift_axis_body,
        span_axis_body,
        is_lifting_surface=True,
        cd0=0.04
    ):
        self.name = name
        self.mass = mass_kg

        # Position of the part center relative to aircraft origin in body frame
        self.r_body = np.array(r_body, dtype=float)

        self.area = area_m2
        self.chord = chord_m

        # Body-frame local axes
        self.lift_axis_body = norm(np.array(lift_axis_body, dtype=float))
        self.span_axis_body = norm(np.array(span_axis_body, dtype=float))

        self.is_lifting_surface = is_lifting_surface
        self.cd0 = cd0

        # Control deflection angle, rad
        # positive means trailing edge down, increasing effective AoA
        self.deflection = 0.0


# ============================================================
# Aircraft rigid body
# ============================================================

class Aircraft:
    def __init__(self):
        self.parts = []

        # ------------------------------------------------------------
        # Coordinate definition:
        # body +x = nose direction
        # body +y = right wing direction
        # body +z = upward
        #
        # origin = rod-fuselage connection point
        # ------------------------------------------------------------

        # Fuselage, 1 kg, located forward of origin
        self.parts.append(AircraftPart(
            name="fuselage",
            mass_kg=1.0,
            r_body=[0.25, 0.0, 0.0],
            area_m2=0.035,
            chord_m=0.50,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[0, 1, 0],
            is_lifting_surface=False,
            cd0=0.12
        ))

        # Rod / tail boom, 0.1 kg, located behind origin
        self.parts.append(AircraftPart(
            name="rod",
            mass_kg=0.1,
            r_body=[-0.45, 0.0, 0.0],
            area_m2=0.010,
            chord_m=0.90,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[0, 1, 0],
            is_lifting_surface=False,
            cd0=0.08
        ))

        # Left wing, 250 g
        self.left_wing = AircraftPart(
            name="left_wing",
            mass_kg=0.25,
            r_body=[0.05, -0.35, 0.0],
            area_m2=0.075,
            chord_m=0.18,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[0, -1, 0],
            is_lifting_surface=True,
            cd0=0.035
        )

        # Right wing, 250 g
        self.right_wing = AircraftPart(
            name="right_wing",
            mass_kg=0.25,
            r_body=[0.05, 0.35, 0.0],
            area_m2=0.075,
            chord_m=0.18,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[0, 1, 0],
            is_lifting_surface=True,
            cd0=0.035
        )

        self.parts.append(self.left_wing)
        self.parts.append(self.right_wing)

        # Left tail wing, 100 g
        self.left_tail = AircraftPart(
            name="left_tail",
            mass_kg=0.1,
            r_body=[-0.90, -0.18, 0.02],
            area_m2=0.030,
            chord_m=0.12,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[0, -1, 0],
            is_lifting_surface=True,
            cd0=0.04
        )

        # Right tail wing, 100 g
        self.right_tail = AircraftPart(
            name="right_tail",
            mass_kg=0.1,
            r_body=[-0.90, 0.18, 0.02],
            area_m2=0.030,
            chord_m=0.12,
            lift_axis_body=[0, 0, 1],
            span_axis_body=[0, 1, 0],
            is_lifting_surface=True,
            cd0=0.04
        )

        self.parts.append(self.left_tail)
        self.parts.append(self.right_tail)

        self.mass = sum(p.mass for p in self.parts)

        # Initial state in world frame
        self.pos = np.array([0.0, 0.0, 20.0], dtype=float)
        self.vel = np.array([18.0, 0.0, 0.0], dtype=float)

        # Quaternion: body to world
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        # Angular velocity in body frame, rad/s
        self.omega_body = np.array([0.0, 0.0, 0.0], dtype=float)

        self.throttle = 0.45
        self.max_thrust = 12.0

        # Calculate inertia tensor around aircraft origin
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
        """
        Approximate each part as a point mass at r_body.
        I = sum m * (|r|^2 I3 - r r^T)
        Add small diagonal terms to avoid singular matrix.
        """
        I = np.zeros((3, 3), dtype=float)

        for p in self.parts:
            r = p.r_body
            I += p.mass * ((np.dot(r, r) * np.eye(3)) - np.outer(r, r))

        I += np.diag([0.02, 0.08, 0.08])
        return I

    def apply_controls(self, keys, dt):
        flap_rate = 75.0 * DEG2RAD
        max_flap = 30.0 * DEG2RAD

        left_cmd = 0.0
        right_cmd = 0.0

        # W: both wings down
        # Positive deflection means trailing edge down,
        # increasing effective angle of attack and lift.
        if keys[K_w]:
            left_cmd += flap_rate
            right_cmd += flap_rate

        # S: both wings up
        if keys[K_s]:
            left_cmd -= flap_rate
            right_cmd -= flap_rate

        # A: left wing up, right wing down
        # This creates differential lift and roll torque.
        if keys[K_a]:
            left_cmd -= flap_rate
            right_cmd += flap_rate

        # D: left wing down, right wing up
        if keys[K_d]:
            left_cmd += flap_rate
            right_cmd -= flap_rate

        self.left_wing.deflection += left_cmd * dt
        self.right_wing.deflection += right_cmd * dt

        self.left_wing.deflection = clamp(self.left_wing.deflection, -max_flap, max_flap)
        self.right_wing.deflection = clamp(self.right_wing.deflection, -max_flap, max_flap)

        # Tail follows mild pitch stabilization.
        # You can remove this if you want the aircraft to be completely manual.
        self.left_tail.deflection *= 0.98
        self.right_tail.deflection *= 0.98

        # Throttle
        if keys[K_UP]:
            self.throttle += 0.5 * dt
        if keys[K_DOWN]:
            self.throttle -= 0.5 * dt
        self.throttle = clamp(self.throttle, 0.0, 1.0)

        # Slowly return wing deflection to neutral if no key is pressed
        if not (keys[K_w] or keys[K_s] or keys[K_a] or keys[K_d]):
            self.left_wing.deflection *= 0.985
            self.right_wing.deflection *= 0.985

    def compute_part_forces(self, part, R):
        """
        Compute force on one part.

        Return:
            F_world: force in world frame
            tau_body: torque around aircraft origin in body frame
        """

        # Position of part in world frame
        r_world = R @ part.r_body

        # Angular velocity in world frame
        omega_world = R @ self.omega_body

        # Velocity of this part due to translation + rotation
        v_part_world = self.vel + np.cross(omega_world, r_world)

        speed = np.linalg.norm(v_part_world)

        if speed < 0.1:
            v_hat_world = np.zeros(3)
        else:
            v_hat_world = v_part_world / speed

        # -----------------------------
        # Gravity
        # -----------------------------
        F_gravity_world = np.array([0.0, 0.0, -part.mass * G])

        # -----------------------------
        # Drag
        # -----------------------------
        # Drag always opposes local velocity.
        # For non-lifting parts, use simple quadratic drag.
        F_drag_world = np.zeros(3)

        if speed > 0.1:
            q_dyn = 0.5 * RHO * speed * speed
            Cd_body = part.cd0
            F_drag_world = -q_dyn * Cd_body * part.area * v_hat_world

        # -----------------------------
        # Lift
        # -----------------------------
        F_lift_world = np.zeros(3)

        if part.is_lifting_surface and speed > 0.1:
            # Convert velocity to body frame
            v_body = R.T @ v_part_world

            # Forward axis is body +x.
            # Up axis is body +z.
            # AoA is approximately atan2(-vz, vx).
            #
            # If aircraft moves forward and air comes from front,
            # local relative wind is opposite velocity.
            # This simplified formula gives positive AoA when nose is pitched up.
            vx = v_body[0]
            vz = v_body[2]

            alpha = math.atan2(-vz, max(abs(vx), 1e-3))

            # Wing deflection changes effective angle of attack.
            # Positive flap down increases effective alpha.
            alpha_eff = alpha + part.deflection

            # Thin airfoil approximation:
            # CL = 2*pi*alpha
            # Stall clamp prevents unrealistic infinite lift.
            alpha_eff = clamp(alpha_eff, -25.0 * DEG2RAD, 25.0 * DEG2RAD)
            CL = 2.0 * math.pi * alpha_eff
            CL = clamp(CL, -1.4, 1.4)

            # Induced drag approximation
            AR = 4.5
            e = 0.75
            k = 1.0 / (math.pi * e * AR)
            CD = part.cd0 + k * CL * CL

            q_dyn = 0.5 * RHO * speed * speed

            # Drag update for lifting surface
            F_drag_world = -q_dyn * CD * part.area * v_hat_world

            # Lift direction:
            # mostly perpendicular to velocity and span.
            span_world = R @ part.span_axis_body

            # lift direction = span x velocity direction
            # choose sign so that lift roughly points along body +z when level flight
            lift_dir_world = np.cross(span_world, v_hat_world)
            lift_dir_world = norm(lift_dir_world)

            body_up_world = R @ np.array([0.0, 0.0, 1.0])
            if np.dot(lift_dir_world, body_up_world) < 0:
                lift_dir_world = -lift_dir_world

            F_lift_world = q_dyn * CL * part.area * lift_dir_world

        F_total_world = F_gravity_world + F_drag_world + F_lift_world

        # Convert force to body frame for torque calculation
        F_body = R.T @ F_total_world
        tau_body = np.cross(part.r_body, F_body)

        return F_total_world, tau_body

    def compute_total_forces_and_torques(self):
        R = quat_to_rotmat(self.q)

        F_total_world = np.zeros(3)
        tau_total_body = np.zeros(3)

        # Sum all part forces
        for part in self.parts:
            F_world, tau_body = self.compute_part_forces(part, R)
            F_total_world += F_world
            tau_total_body += tau_body

        # -----------------------------
        # Thrust
        # -----------------------------
        # Thrust acts along body +x.
        # Apply it near fuselage/nose.
        thrust_body = np.array([self.max_thrust * self.throttle, 0.0, 0.0])
        thrust_world = R @ thrust_body

        thrust_pos_body = np.array([0.35, 0.0, 0.0])
        thrust_tau_body = np.cross(thrust_pos_body, thrust_body)

        F_total_world += thrust_world
        tau_total_body += thrust_tau_body

        # -----------------------------
        # Angular damping
        # -----------------------------
        # This is physical-like aerodynamic damping.
        # It prevents the simulation from becoming numerically unstable.
        tau_total_body += -0.15 * self.omega_body

        return F_total_world, tau_total_body

    def step(self, dt):
        R = quat_to_rotmat(self.q)

        F_world, tau_body = self.compute_total_forces_and_torques()

        # -----------------------------
        # Linear dynamics
        # -----------------------------
        acc_world = F_world / self.mass

        self.vel += acc_world * dt
        self.pos += self.vel * dt

        # Ground collision
        if self.pos[2] < 0.3:
            self.pos[2] = 0.3
            if self.vel[2] < 0:
                self.vel[2] *= -0.15
            self.vel[0] *= 0.96
            self.vel[1] *= 0.96

        # -----------------------------
        # Rotational dynamics
        # Euler equation:
        # I * omega_dot + omega x Iomega = tau
        # -----------------------------
        I = self.I_body
        Iomega = I @ self.omega_body

        omega_dot = self.I_body_inv @ (
            tau_body - np.cross(self.omega_body, Iomega)
        )

        self.omega_body += omega_dot * dt

        # Limit angular velocity to avoid numerical explosion
        max_omega = 8.0
        wmag = np.linalg.norm(self.omega_body)
        if wmag > max_omega:
            self.omega_body = self.omega_body / wmag * max_omega

        # Update orientation
        self.q = integrate_quaternion(self.q, self.omega_body, dt)


# ============================================================
# OpenGL drawing helpers
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
    for f in faces:
        for idx in f:
            glVertex3fv(vertices[idx])
    glEnd()


def draw_wing(span, chord, thickness=0.02):
    draw_box([chord, span, thickness])


def draw_aircraft(aircraft):
    R = quat_to_rotmat(aircraft.q)

    # Convert rotation matrix to OpenGL 4x4 matrix
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = aircraft.pos

    glPushMatrix()
    glMultMatrixf(M.T)

    # Draw origin marker
    glColor3f(1.0, 1.0, 0.0)
    draw_box([0.05, 0.05, 0.05])

    # Draw each part in body frame
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
    pygame.display.set_caption("Rigid Body Aircraft Physics Simulation")

    setup_opengl(width, height)

    clock = pygame.time.Clock()
    aircraft = Aircraft()

    running = True
    accumulator = 0.0

    font = pygame.font.SysFont("Consolas", 18)

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
            f"speed={speed:5.2f} m/s | alt={aircraft.pos[2]:5.2f} m | "
            f"thr={aircraft.throttle:4.2f} | "
            f"roll={roll:6.2f} deg | pitch={pitch:6.2f} deg | yaw={yaw:6.2f} deg | "
            f"L flap={aircraft.left_wing.deflection * RAD2DEG:6.2f} | "
            f"R flap={aircraft.right_wing.deflection * RAD2DEG:6.2f}"
        )

    pygame.quit()


if __name__ == "__main__":
    main()