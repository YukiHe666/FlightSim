import pygame
from pygame.locals import *

pygame.init()

screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("W Key Test")

clock = pygame.time.Clock()
running = True

while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

        if event.type == KEYDOWN:
            print("KEYDOWN:", event.key, pygame.key.name(event.key))

        if event.type == KEYUP:
            print("KEYUP:", event.key, pygame.key.name(event.key))

    keys = pygame.key.get_pressed()

    if keys[K_w]:
        print("W is being held down")

    screen.fill((30, 30, 30))

    font = pygame.font.SysFont(None, 36)

    if keys[K_w]:
        text = font.render("W is pressed", True, (0, 255, 0))
    else:
        text = font.render("W is NOT pressed", True, (255, 255, 255))

    screen.blit(text, (180, 180))
    pygame.display.flip()

pygame.quit()