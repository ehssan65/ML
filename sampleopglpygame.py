import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math

# تنظیمات اولیه
def init():
    glClearColor(0.0, 1.0, 1.0, 1.0)
    gluOrtho2D(0, 3, 0, 3)

# رسم شبکه بازی دوز
def draw_grid():
    glColor3f(0.0, 0.0, 0.0)
    glBegin(GL_LINES)
    for i in range(1, 3):
        glVertex2f(i, 0)
        glVertex2f(i, 3)
        glVertex2f(0, i)
        glVertex2f(3, i)
    glEnd()

# رسم علامت X
def draw_x(x, y):
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_LINES)
    glVertex2f(x + 0.1, y + 0.1)
    glVertex2f(x + 0.9, y + 0.9)
    glVertex2f(x + 0.1, y + 0.9)
    glVertex2f(x + 0.9, y + 0.1)
    glEnd()

# رسم علامت O
def draw_o(x, y):
    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_LINE_LOOP)
    for i in range(360):
        angle = i * 3.14159 / 180
        glVertex2f(x + 0.5 + 0.4 * math.cos(angle), y + 0.5 + 0.4 * math.sin(angle))
    glEnd()

# بررسی برنده شدن
def check_winner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != 0:
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    return 0

def main():
    pygame.init()
    display = (600, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    init()

    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    turn = 1
    winner = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == MOUSEMOTION:
                (x,y) = pygame.mouse.get_pos()
                print('x=',x,'  ',x//200,'   y=',y,'   ',y//200)
            elif event.type == MOUSEBUTTONDOWN and not winner:
                x, y = event.pos
                print('x=',x,'y=',y)
                grid_x = x // 200
                grid_y = y // 200
                if board[grid_y][grid_x] == 0:
                    board[grid_y][grid_x] = turn
                    turn = 3 - turn  # تغییر نوبت بین 1 و 2
                    winner = check_winner(board)
                print(board)

        glClear(GL_COLOR_BUFFER_BIT)
        draw_grid()

        for i in range(3):
            for j in range(3):
                if board[i][j] == 1:
                    draw_x(j, 2 - i)
                elif board[i][j] == 2:
                    draw_o(j, 2 - i)

        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()
