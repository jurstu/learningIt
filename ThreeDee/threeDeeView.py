import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os 
import time
from rotation import Rotation

class ThreeDeeView:
    def __init__(self, startPos=[100, 100]):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (startPos[0], startPos[1])
        self.angle_yaw = 0
        self.angle_pitch = 0
        self.angle_roll = 0

        self.points = [
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ]
        
        self.edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
                (0, 7)
        ]
        pygame.init()
        display = (1200, 1200)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)



    def draw(self, p=None, e=None):
        if(p == None):
            p = self.points
        if(e == None):
            e = self.edges

        glPointSize(10)
        glBegin(GL_POINTS)
        for i in range(len(p)):
            glVertex3fv(p[i])  # Draw the point at the given position
        glEnd()


        glBegin(GL_LINES)
        for edge in e:
            for vertex in edge:
                
                glColor3f(0.5, 1, 0.5)
                glVertex3fv(p[vertex])
            
        glEnd()


    def update(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        
        # Apply rotation
        glRotatef(self.angle_yaw, 1, 0, 0)  # Rotate around X-axis
        glRotatef(self.angle_pitch, 0, 1, 0)  # Rotate around Y-axis
        glRotatef(self.angle_roll, 0, 0, 1)
        

        self.draw()
        
        glPopMatrix()
        
        pygame.display.flip()


    def updateMine(self, points, edges):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        
        # Apply rotation
        glRotatef(self.angle_yaw, 1, 0, 0)  # Rotate around X-axis
        glRotatef(self.angle_pitch, 0, 1, 0)  # Rotate around Y-axis
        glRotatef(self.angle_roll, 0, 0, 1)
        
        self.draw(p = points, e = edges)
        
        glPopMatrix()
        
        pygame.display.flip()




if __name__ == "__main__":
    t = ThreeDeeView()

    state = 0

    i = 0
    R = Rotation()
    while(1):
        
        i += 0.1
        R.pitch = np.deg2rad(i)
        #R.roll = np.deg2rad(i)
        #R.yaw = np.deg2rad(i)
        
        print(i)

        R1 = R.getRotationMatrix()
        R2 = R.getRotationMatrixInverted()
        R_none = np.dot(R1, R2)
        points = R.rotatePoints(t.points, R_none)
        
        #if(state == 0):
        #    points = R.rotatePoints(t.points, R.getRotationMatrix())    
        #else:
            


        t.updateMine(points, t.edges)





        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if(event.key == pygame.K_SLASH):
                    state = 1 - state
                    print(state)

                if event.key == pygame.K_LEFT:
                    t.angle_pitch -= 5
                elif event.key == pygame.K_RIGHT:
                    t.angle_pitch += 5
                elif event.key == pygame.K_UP:
                    t.angle_yaw -= 5
                elif event.key == pygame.K_DOWN:
                    t.angle_yaw += 5
                elif event.key == pygame.K_COMMA:
                    t.angle_roll += 5
                elif event.key == pygame.K_PERIOD:
                    t.angle_roll -= 5

        time.sleep(0.01)