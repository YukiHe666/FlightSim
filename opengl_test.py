import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import random
import math
import time


WIDTH, HEIGHT = 1000, 700

GRAVITY = -9.8
RESTITUTION = 0.85
BOX_SIZE = 6.0
TIME_SCALE = 1.0


class Ball:
    def __init__(self):
        self.radius = random.uniform(0.25, 0.45)

        self.pos = [
            random.uniform(-2.0, 2.0),
            random.uniform(0.0, 4.0),
            random.uniform(-2.0, 2.0),
        ]

        self.vel = [
            random.uniform(-2.0, 2.0),
            random.uniform(-1.0, 2.0),
            random.uniform(-2.0, 2.0),
        ]

        self.color = [
            random.uniform(0.3, 1.0),
            random.uniform(0.3, 1.0),
            random.uniform(0.3, 1.0),
        ]

    def update(self, dt):
        # Gravity affects vertical velocity
        self.vel[1] += GRAVITY * dt

        # Integrate velocity into position
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        self.pos[2] += self.vel[2] * dt

        half = BOX_SIZE / 2.0

        # Bounce on x walls
        if self.pos[0] - self.radius < -half:
            self.pos[0] = -half + self.radius
            self.vel[0] *= -RESTITUTION

        if self.pos[0] + self.radius > half:
            self.pos[0] = half - self.radius
            self.vel[0] *= -RESTITUTION

        # Bounce on floor and ceiling
        if self.pos[1] - self.radius < -half:
            self.pos[1] = -half + self.radius
            self.vel[1] *= -RESTITUTION

        if self.pos[1] + self.radius > half:
            self.pos[1] = half - self.radius
            self.vel[1] *= -RESTITUTION

        # Bounce on z walls
        if self.pos[2] - self.radius < -half:
            self.pos[2] = -half + self.radius
            self.vel[2] *= -RESTITUTION

        if self.pos[2] + self.radius > half:
            self.pos[2] = half - self.radius
            self.vel[2] *= -RESTITUTION


def draw_sphere(ball):
    glPushMatrix()

    glTranslatef(ball.pos[0], ball.pos[1], ball.pos[2])
    glColor3f(ball.color[0], ball.color[1], ball.color[2])

    quad = gluNewQuadric()
    gluSphere(quad, ball.radius, 32, 32)
    gluDeleteQuadric(quad)

    glPopMatrix()


def draw_box():
    half = BOX_SIZE / 2.0

    vertices = [
        [-half, -half, -half],
        [ half, -half, -half],
        [ half,  half, -half],
        [-half,  half, -half],
        [-half, -half,  half],
        [ half, -half,  half],
        [ half,  half,  half],
        [-half,  half,  half],
    ]

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


def setup_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    glLightfv(GL_LIGHT0, GL_POSITION, [4.0, 8.0, 6.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.2, 0.2, 0.2, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.8, 0.8, 0.8, 1.0])

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glEnable(GL_DEPTH_TEST)


def main():
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Bouncing Balls - Python OpenGL Demo")

    glViewport(0, 0, WIDTH, HEIGHT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, WIDTH / HEIGHT, 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)

    setup_lighting()

    balls = [Ball() for _ in range(12)]

    clock = pygame.time.Clock()
    running = True

    camera_angle = 0.0

    while running:
        dt = clock.tick(60) / 1000.0
        dt *= TIME_SCALE

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

        for ball in balls:
            ball.update(dt)

        camera_angle += 20.0 * dt
        cam_x = 10.0 * math.sin(math.radians(camera_angle))
        cam_z = 10.0 * math.cos(math.radians(camera_angle))

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        gluLookAt(
            cam_x, 5.0, cam_z,
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0
        )

        draw_box()

        for ball in balls:
            draw_sphere(ball)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()