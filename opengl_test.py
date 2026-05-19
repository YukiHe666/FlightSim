import math
import numpy as np
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *


# =========================
# Basic constants
# =========================

WIDTH = 1200
HEIGHT = 800

RHO = 1.225
G = 9.81

DT_MAX = 1.0 / 30.0


# =========================
# Math helpers
# =========================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v * 0.0
    return v / n


def rot_x(a):
    c = math.cos(a)
    s = math.sin(a)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ], dtype=float)


def rot_y(a):
    c = math.cos(a)
    s = math.sin(a)

    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=float)


def rot_z(a):
    c = math.cos(a)
    s = math.sin(a)

    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)


# =========================
# Aircraft model
#
# World frame:
#   x = forward initially
#   y = up
#   z = right
#
# Body frame:
#   x = nose
#   y = top
#   z = right wing
# =========================

class Aircraft:
    def __init__(self):
        self.mass = 3.0

        self.pos = np.array([0.0, 35.0, 0.0], dtype=float)
        self.vel = np.array([24.0, 0.0, 0.0], dtype=float)

        # Attitude angles
        self.yaw = 0.0
        self.pitch = math.radians(3.0)
        self.bank = 0.0

        # Commanded values
        self.target_bank = 0.0
        self.target_pitch = math.radians(3.0)

        # Controls
        self.throttle = 0.65
        self.rudder = 0.0

        # Aerodynamics
        self.wing_area = 0.75
        self.CL0 = 0.16
        self.CL_alpha = 4.5
        self.CL_max = 1.25
        self.CD0 = 0.04
        self.k_induced = 0.08

        # Engine
        self.max_thrust = 40.0

        # Debug values
        self.yaw_rate = 0.0
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.alpha = 0.0

    def reset(self):
        self.__init__()

    def rotation_matrix(self):
        """
        Body-to-world rotation.

        Body frame:
            x = nose
            y = aircraft top
            z = right wing

        This builds the display/physics orientation.
        """
        return rot_y(self.yaw) @ rot_z(self.pitch) @ rot_x(self.bank)

    def forward_vector(self):
        R = self.rotation_matrix()
        return R @ np.array([1.0, 0.0, 0.0], dtype=float)

    def right_vector(self):
        R = self.rotation_matrix()
        return R @ np.array([0.0, 0.0, 1.0], dtype=float)

    def up_vector(self):
        R = self.rotation_matrix()
        return R @ np.array([0.0, 1.0, 0.0], dtype=float)

    def update(self, dt):
        dt = min(dt, DT_MAX)

        speed = np.linalg.norm(self.vel)
        speed = max(speed, 0.1)

        # =========================
        # 1. Assisted attitude response
        # =========================

        # Roll / bank response
        bank_error = self.target_bank - self.bank

        max_roll_rate = math.radians(35.0)
        desired_roll_rate = 2.6 * bank_error
        desired_roll_rate = clamp(desired_roll_rate, -max_roll_rate, max_roll_rate)

        self.roll_rate = desired_roll_rate
        self.bank += self.roll_rate * dt

        # Pitch response
        pitch_error = self.target_pitch - self.pitch

        max_pitch_rate = math.radians(25.0)
        desired_pitch_rate = 2.0 * pitch_error
        desired_pitch_rate = clamp(desired_pitch_rate, -max_pitch_rate, max_pitch_rate)

        self.pitch_rate = desired_pitch_rate
        self.pitch += self.pitch_rate * dt

        # Clamp attitude
        self.bank = clamp(self.bank, math.radians(-45.0), math.radians(45.0))
        self.pitch = clamp(self.pitch, math.radians(-20.0), math.radians(20.0))

        # =========================
        # 2. Coordinated turn
        # =========================
        # Important direction fix:
        # In this coordinate/rendering convention, the sign needs to be negative.
        #
        # Result:
        #   bank < 0, left bank  -> left turn
        #   bank > 0, right bank -> right turn
        # =========================

        coordinated_yaw_rate = -G * math.tan(self.bank) / speed

        # Rudder yaw.
        # A/D input has already been sign-corrected in the main loop.
        rudder_yaw_rate = 0.7 * self.rudder

        self.yaw_rate = coordinated_yaw_rate + rudder_yaw_rate
        self.yaw += self.yaw_rate * dt

        # =========================
        # 3. Aerodynamic forces
        # =========================

        forward = self.forward_vector()
        right = self.right_vector()

        v_hat = normalize(self.vel)

        if np.linalg.norm(self.vel) < 0.5:
            v_hat = forward

        # Flight path angle
        gamma = math.asin(clamp(v_hat[1], -1.0, 1.0))

        # Angle of attack approximately equals pitch minus flight path angle
        self.alpha = self.pitch - gamma
        self.alpha = clamp(self.alpha, math.radians(-15.0), math.radians(15.0))

        CL = self.CL0 + self.CL_alpha * self.alpha
        CL = clamp(CL, -self.CL_max, self.CL_max)
        # lookup table CL V alpha

        CD = self.CD0 + self.k_induced * CL * CL

        qbar = 0.5 * RHO * speed * speed

        lift_mag = qbar * self.wing_area * CL
        drag_mag = qbar * self.wing_area * CD

        # Lift direction:
        # perpendicular to velocity and wing span direction.
        lift_dir = np.cross(right, v_hat)
        lift_dir = normalize(lift_dir)

        drag_dir = -v_hat

        lift = lift_mag * lift_dir
        drag = drag_mag * drag_dir

        thrust = self.throttle * self.max_thrust * forward
        gravity = np.array([0.0, -self.mass * G, 0.0], dtype=float)

        total_force = lift + drag + thrust + gravity

        acc = total_force / self.mass

        self.vel += acc * dt
        self.pos += self.vel * dt

        # =========================
        # 4. Simple ground handling
        # =========================

        if self.pos[1] < 0.5:
            self.pos[1] = 0.5

            if self.vel[1] < 0.0:
                self.vel[1] = 0.0

            # Ground friction
            self.vel[0] *= 0.97
            self.vel[2] *= 0.97

            # Keep it mostly upright on ground
            self.bank *= 0.95
            self.pitch = max(self.pitch, math.radians(0.0))

        # Safety bounds for demo
        self.pos[1] = clamp(self.pos[1], 0.5, 500.0)


# =========================
# OpenGL drawing
# =========================

def set_camera(aircraft):
    forward = aircraft.forward_vector()

    target = aircraft.pos + forward * 5.0
    camera_pos = aircraft.pos - forward * 16.0 + np.array([0.0, 6.0, 0.0])

    gluLookAt(
        camera_pos[0], camera_pos[1], camera_pos[2],
        target[0], target[1], target[2],
        0.0, 1.0, 0.0
    )


def draw_ground_grid(size=300, step=5):
    glDisable(GL_LIGHTING)

    glColor3f(0.25, 0.25, 0.25)
    glBegin(GL_LINES)

    for i in range(-size, size + 1, step):
        glVertex3f(-size, 0.0, i)
        glVertex3f(size, 0.0, i)

        glVertex3f(i, 0.0, -size)
        glVertex3f(i, 0.0, size)

    glEnd()

    # Axes
    glBegin(GL_LINES)

    # x axis red
    glColor3f(0.8, 0.2, 0.2)
    glVertex3f(0.0, 0.03, 0.0)
    glVertex3f(30.0, 0.03, 0.0)

    # z axis blue
    glColor3f(0.2, 0.2, 0.9)
    glVertex3f(0.0, 0.03, 0.0)
    glVertex3f(0.0, 0.03, 30.0)

    glEnd()

    glEnable(GL_LIGHTING)


def draw_aircraft_body():
    """
    Draw aircraft in body coordinates:
    x = nose
    y = up
    z = right wing
    """

    # Fuselage
    glColor3f(0.85, 0.85, 0.9)

    x_front = 1.6
    x_back = -2.4
    y_top = 0.18
    y_bottom = -0.18
    z_side = 0.18

    glBegin(GL_QUADS)

    # Top
    glNormal3f(0, 1, 0)
    glVertex3f(x_front, y_top, -z_side)
    glVertex3f(x_front, y_top, z_side)
    glVertex3f(x_back, y_top, z_side)
    glVertex3f(x_back, y_top, -z_side)

    # Bottom
    glNormal3f(0, -1, 0)
    glVertex3f(x_front, y_bottom, z_side)
    glVertex3f(x_front, y_bottom, -z_side)
    glVertex3f(x_back, y_bottom, -z_side)
    glVertex3f(x_back, y_bottom, z_side)

    # Right
    glNormal3f(0, 0, 1)
    glVertex3f(x_front, y_bottom, z_side)
    glVertex3f(x_back, y_bottom, z_side)
    glVertex3f(x_back, y_top, z_side)
    glVertex3f(x_front, y_top, z_side)

    # Left
    glNormal3f(0, 0, -1)
    glVertex3f(x_front, y_bottom, -z_side)
    glVertex3f(x_front, y_top, -z_side)
    glVertex3f(x_back, y_top, -z_side)
    glVertex3f(x_back, y_bottom, -z_side)

    glEnd()

    # Nose
    glColor3f(0.95, 0.95, 1.0)
    glBegin(GL_TRIANGLES)

    nose = [2.1, 0.0, 0.0]

    glVertex3f(*nose)
    glVertex3f(x_front, y_top, z_side)
    glVertex3f(x_front, y_top, -z_side)

    glVertex3f(*nose)
    glVertex3f(x_front, y_bottom, -z_side)
    glVertex3f(x_front, y_bottom, z_side)

    glVertex3f(*nose)
    glVertex3f(x_front, y_top, -z_side)
    glVertex3f(x_front, y_bottom, -z_side)

    glVertex3f(*nose)
    glVertex3f(x_front, y_bottom, z_side)
    glVertex3f(x_front, y_top, z_side)

    glEnd()

    # Main wing
    glColor3f(0.25, 0.45, 0.95)
    glBegin(GL_TRIANGLES)

    glNormal3f(0, 1, 0)

    # Right wing
    glVertex3f(0.6, 0.0, 0.15)
    glVertex3f(-0.5, 0.0, 0.15)
    glVertex3f(-0.2, 0.0, 2.6)

    # Left wing
    glVertex3f(0.6, 0.0, -0.15)
    glVertex3f(-0.2, 0.0, -2.6)
    glVertex3f(-0.5, 0.0, -0.15)

    glEnd()

    # Horizontal tail
    glColor3f(0.25, 0.8, 0.7)
    glBegin(GL_TRIANGLES)

    glNormal3f(0, 1, 0)

    glVertex3f(-1.7, 0.05, 0.12)
    glVertex3f(-2.3, 0.05, 0.12)
    glVertex3f(-2.1, 0.05, 1.0)

    glVertex3f(-1.7, 0.05, -0.12)
    glVertex3f(-2.1, 0.05, -1.0)
    glVertex3f(-2.3, 0.05, -0.12)

    glEnd()

    # Vertical tail
    glColor3f(0.95, 0.35, 0.25)
    glBegin(GL_TRIANGLES)

    glNormal3f(0, 0, 1)

    glVertex3f(-1.8, 0.1, 0.0)
    glVertex3f(-2.35, 0.1, 0.0)
    glVertex3f(-2.15, 1.0, 0.0)

    glEnd()


def draw_aircraft(aircraft):
    glPushMatrix()

    glTranslatef(
        aircraft.pos[0],
        aircraft.pos[1],
        aircraft.pos[2]
    )

    R = aircraft.rotation_matrix()

    M = np.eye(4, dtype=np.float32)
    M[0:3, 0:3] = R

    glMultMatrixf(M.T)

    draw_aircraft_body()

    glPopMatrix()


def setup_opengl():
    glViewport(0, 0, WIDTH, HEIGHT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, WIDTH / HEIGHT, 0.1, 1000.0)

    glMatrixMode(GL_MODELVIEW)

    glEnable(GL_DEPTH_TEST)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    glLightfv(GL_LIGHT0, GL_POSITION, [20.0, 50.0, 20.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.25, 0.25, 0.25, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.85, 0.85, 0.85, 1.0])

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glClearColor(0.08, 0.10, 0.14, 1.0)


def update_window_caption(aircraft):
    speed = np.linalg.norm(aircraft.vel)

    caption = (
        f"Flight Demo | "
        f"speed={speed:.1f} m/s | "
        f"alt={aircraft.pos[1]:.1f} m | "
        f"thr={aircraft.throttle:.2f} | "
        f"bank={math.degrees(aircraft.bank):.1f} deg | "
        f"target_bank={math.degrees(aircraft.target_bank):.1f} deg | "
        f"yaw={math.degrees(aircraft.yaw):.1f} deg | "
        f"yaw_rate={math.degrees(aircraft.yaw_rate):.1f} deg/s"
    )

    pygame.display.set_caption(caption)


# =========================
# Main
# =========================

def main():
    pygame.init()

    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Aircraft Flight Demo")

    setup_opengl()

    clock = pygame.time.Clock()
    aircraft = Aircraft()

    print("Controls:")
    print("  Left arrow         : bank left and turn left")
    print("  Right arrow        : bank right and turn right")
    print("  Up / Down arrow    : pitch target up / down")
    print("  A                  : yaw left")
    print("  D                  : yaw right")
    print("  W / S              : throttle up / down")
    print("  PageUp / PageDown  : throttle alternative")
    print("  R                  : reset")
    print("  ESC                : quit")
    print()
    print("This version uses a coordinated-turn model:")
    print("  yaw_rate = -g * tan(bank) / speed")
    print()
    print("Click the OpenGL window first, then press keys.")
    print()

    running = True
    debug_timer = 0.0

    while running:
        dt = clock.tick(60) / 1000.0
        debug_timer += dt

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

                if event.key == K_r:
                    aircraft.reset()

                if event.key == K_w or event.key == K_PAGEUP:
                    aircraft.throttle += 0.08

                if event.key == K_s or event.key == K_PAGEDOWN:
                    aircraft.throttle -= 0.08

        keys = pygame.key.get_pressed()

        # =========================
        # Left / Right bank command
        # =========================

        max_bank_cmd = math.radians(28.0)

        if keys[K_LEFT]:
            aircraft.target_bank = -max_bank_cmd
        elif keys[K_RIGHT]:
            aircraft.target_bank = max_bank_cmd
        else:
            aircraft.target_bank = 0.0

        # =========================
        # Pitch command
        # =========================

        neutral_pitch = math.radians(3.0)
        max_pitch_cmd = math.radians(14.0)
        min_pitch_cmd = math.radians(-8.0)

        if keys[K_UP]:
            aircraft.target_pitch += math.radians(25.0) * dt
        elif keys[K_DOWN]:
            aircraft.target_pitch -= math.radians(25.0) * dt
        else:
            # Slowly return pitch target to neutral
            if aircraft.target_pitch > neutral_pitch:
                aircraft.target_pitch -= math.radians(10.0) * dt
            elif aircraft.target_pitch < neutral_pitch:
                aircraft.target_pitch += math.radians(10.0) * dt

        aircraft.target_pitch = clamp(
            aircraft.target_pitch,
            min_pitch_cmd,
            max_pitch_cmd
        )

        # =========================
        # Rudder A / D
        # =========================
        # Direction fix:
        # A should yaw left, D should yaw right.
        # In this coordinate convention, A increases rudder,
        # D decreases rudder.
        # =========================

        rudder_rate = math.radians(90.0) * dt
        rudder_return_rate = math.radians(70.0) * dt

        if keys[K_a]:
            aircraft.rudder += rudder_rate
        elif keys[K_d]:
            aircraft.rudder -= rudder_rate
        else:
            if aircraft.rudder > 0:
                aircraft.rudder = max(0.0, aircraft.rudder - rudder_return_rate)
            else:
                aircraft.rudder = min(0.0, aircraft.rudder + rudder_return_rate)

        aircraft.rudder = clamp(
            aircraft.rudder,
            math.radians(-25.0),
            math.radians(25.0)
        )

        # =========================
        # Throttle W / S
        # =========================

        throttle_rate = 1.5 * dt

        if keys[K_w] or keys[K_PAGEUP]:
            aircraft.throttle += throttle_rate

        if keys[K_s] or keys[K_PAGEDOWN]:
            aircraft.throttle -= throttle_rate

        aircraft.throttle = clamp(aircraft.throttle, 0.0, 1.0)

        # =========================
        # Update physics
        # =========================

        aircraft.update(dt)

        # =========================
        # Debug print
        # =========================

        if debug_timer > 0.5:
            debug_timer = 0.0

            speed = np.linalg.norm(aircraft.vel)

            print(
                f"speed={speed:6.2f} m/s | "
                f"alt={aircraft.pos[1]:6.2f} m | "
                f"thr={aircraft.throttle:4.2f} | "
                f"bank={math.degrees(aircraft.bank):7.2f} deg | "
                f"target_bank={math.degrees(aircraft.target_bank):7.2f} deg | "
                f"yaw={math.degrees(aircraft.yaw):7.2f} deg | "
                f"yaw_rate={math.degrees(aircraft.yaw_rate):7.2f} deg/s | "
                f"pitch={math.degrees(aircraft.pitch):6.2f} deg | "
                f"alpha={math.degrees(aircraft.alpha):6.2f} deg | "
                f"rud={math.degrees(aircraft.rudder):6.2f} deg"
            )

        # =========================
        # Draw
        # =========================

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        set_camera(aircraft)

        draw_ground_grid()
        draw_aircraft(aircraft)

        update_window_caption(aircraft)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()