"""
From https://openreview.net/pdf?id=S1l8oANFDH
"""


import numpy as np
import time
from collision import *
from system import *
from utils import *
import pygame


class CarReversePP(System):
    def __init__(self, n_steps=100):
        self.height = 5.0
        self.width = 1.8
        self.dist_min = 15.0
        self.dist_max = 15.0

        self.x_lane_1 = -1.5
        self.x_lane_2 = 1.0

        self.goal_ang = np.pi / 2.0
        self.dt = 0.02
        self.tol = 0.05

        self.num_actions = 2
        self.num_cond_features = 6
        self.num_act_features = 1

        self.world_size = 30
        self.viewer = None
        self.counter = 0
        self.n_steps = n_steps

        self.dt_scale = 1.0
        self.test_dt_scale = 10.0

        self.infinite_system = False
        self.time_weight = 0.01

        self.screen = None  # Pygame screen object

    def set_inp_limits(self, lim):
        self.dist_min = lim[0]
        self.dist_max = lim[1]

    def simulate(self, state, action, dt):
        # Adjust dt based on scale
        if dt < -0.01:
            dt = self.dt
        else:
            dt = dt / self.dt_scale
        ns = np.copy(state)
        v, w = action 
        w = w / 10.0

        # Clip actions to reasonable limits
        if (v > 5.0):
            v = 5
        if (v < -5.0):
            v = -5
        if (w > 0.5):
            w = 0.5
        if (w < -0.5):
            w = -0.5

        x, y, ang, _ = state
        beta = np.arctan(0.5 * np.tan(w))
        dx = v * np.cos(ang + beta) * dt 
        dy = v * np.sin(ang + beta) * dt 
        da = v / (self.height / 2.0) * np.sin(beta) * dt 

        ns[0] += dx 
        ns[1] += dy 
        ns[2] += da 

        self.counter += 1
        return ns 
        ''' # with torch
        def simulate(self, state, action, dt):
            if dt < -0.01:
                dt = self.dt
            else:
                dt = dt/self.dt_scale

            v, w = action 
            w = w/10.0


            x,y,ang,_ = state   
            beta = torch.atan(0.5*torch.tan(w))
            dx = v*torch.cos(ang + beta)*dt 
            dy = v*torch.sin(ang + beta)*dt 
            da = v/(2.5)*torch.sin(beta)*dt 

            # update counter
            self.counter += 1

            return state + np.array([dx, dy, da, 0])'''
        

    def abstract_actions(self, a):
        a[a>=0] = 1.0
        a[a<0] = -1.0
        return a 
        
    def get_features(self, state):
        # Extract some features from state (e.g., for cost or visualization)
        features = []
        x, y, ang, dist = state
        features.append(x)
        features.append(y)
        features.append(ang * 5.0)

        # Compute distances from obstacles (using vertices)
        d1 = 1e20  # min distance to front car
        d2 = 1e20  # min distance to back car
        d3 = 1e20 # min dist to end of lane 

        vertices = get_all_vertices(x, y, ang, self.width, self.height)
        for v in vertices:
            d = max(dist - self.height/2.0 - v[1],
                    self.x_lane_2 - self.width/2.0 - v[0])
            if d < d1:
                d1 = d 
            d = max(v[1] - self.height/2.0,
                    self.x_lane_2 - self.width/2.0 - v[0])
            if d < d2:
                d2 = d 

            d = 2.2 - v[0]
            if d < d3:
                d3 = d 

        features.append(d1)
        features.append(d2)
        #features.append(d3)

        return features

    def check_safe(self, state):
        # Check collisions and boundaries
        e1 = self.check_collision(state)
        e2 = self.check_boundaries(state)
        return e1 + e2

    def check_collision(self, state):
        x, y, ang, d = state

        # Obstacle 1 (front car)
        bx = self.x_lane_2
        by = 0.0
        e1 = check_collision_box(x, y, ang, bx, by, 'l', self.width, self.height)

        # Obstacle 2 (back car)
        bx = self.x_lane_2
        by = d
        e2 = check_collision_box(x, y, ang, bx, by, 'u', self.width, self.height)

        return e1 + e2

    def check_boundaries(self, state):
        x, y, ang, _ = state
        vertices = get_all_vertices(x, y, ang, self.width, self.height)
        d1 = 1e20 
        d2 = 1e20
        for v in vertices:
            d = 2.5 - v[0]
            if d < d1:
                d1 = d 
            d = v[0] - (-5)
            if d < d2:
                d2 = d 
        err = 0.0
        if d1 < 0.0:
            err += -d1 
        if d2 < 0.0:
            err += -d2 
        return err 

    def check_goal(self, state):
        # Compute error relative to the parking goal
        x, y, ang, dist = state

        error = 0.0
        error_x = 0.0
        error_y = 0.0
        error_ang = 0.0

        # error for x
        if x > self.x_lane_2 - self.width:
            error_x += x - self.x_lane_2 + self.width

        # error for ang
        if abs(ang - self.goal_ang) > self.tol:
            error_ang += abs(ang - self.goal_ang) - self.tol

        # error for y
		#if (y < dist - self.height):
		#	error_y += dist - self.height - y

        error = error_x + error_y + error_ang   # not used

        return [error_x, 5.0 * error_ang]
    
    def check_time(self, total_time):
        return 0.0

    def get_obj(self, state):
        return 0.0

    def done(self, state):
        # For now, we simply end after a fixed number of steps
        goal_err = self.check_goal(state)
        return self.counter >= self.n_steps     # or np.sum(goal_err) < 0.01

    def sample_init_state(self):
        # Initialize the state near a desired starting position
        x = self.x_lane_2 + rand(-0.04, 0.04)
        ang = np.pi/2.0 + rand(-0.04, 0.04)
        dist = rand(self.dist_min, self.dist_max) 
        y = self.height + 0.21
        return np.array([x, y, ang, dist])

    def get_neutral_state(self):
        x = 0.0
        ang = np.pi/2.0 
        dist = 15.0
        y = 2.5
        return np.array([x, y, ang, dist])


    def render(self, state, mode='human'):
        if mode == 'human' and self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            pygame.display.set_caption("Car Parking Simulation")
        
        if self.screen is not None:
            # Clear screen
            self.screen.fill((255, 255, 255))
            
            # Process Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.reset_render()
            
            # Draw elements
            scale = 600 / self.world_size
            vshift = -5 * scale

            # Draw obstacle cars
            pygame.draw.rect(
                self.screen, (0, 0, 200),
                pygame.Rect(
                    (self.x_lane_2 - self.width/2) * scale + 300 - 5,
                    300 + vshift - self.height/2 * scale,
                    self.width * scale,
                    self.height * scale
                )
            )

            pygame.draw.rect(
                self.screen, (0, 0, 200),
                pygame.Rect(
                    (self.x_lane_2 - self.width/2) * scale + 300 - 5,
                    (state[3] - self.height/2) * scale + 300 + vshift,
                    self.width * scale,
                    self.height * scale
                )
            )

            # Draw main car (blue)
            vertices = get_all_vertices(state[0], state[1], state[2], self.width, self.height)
            scaled_vertices = [
                (v[0] * scale + 300, v[1] * scale + 300 + vshift) 
                for v in vertices
            ]
            pygame.draw.polygon(self.screen, (100, 100, 255), scaled_vertices)

            pygame.display.flip()

        if mode == 'rgb_array':
            return pygame.surfarray.array3d(self.screen) if self.screen else None
        return None
    
    def reset_render(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        self.counter = 0

