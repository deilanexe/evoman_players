#################################################
# Punching Bag for EvoMan FrameWork - V0.1 2020 #
# Author: Daniel Macias                         #
# daniel.macias.galindo@gmail.com               #
#################################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from pb_controller import player_controller
sys.path.insert(1, '../evoman_framework')
from demo_controller import enemy_controller
import time
import numpy as np

experiment_name = 'punching_bag'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

class environm(Environment):

	# implements fitness function
	def fitness_single(self):

		if self.contacthurt == "player":
			return 0.9*(100 - self.get_enemylife()) + 0.1*self.get_playerlife() - np.log(self.get_time())

		else:
			return 0.9*(100 - self.get_playerlife()) + 0.1*self.get_enemylife() - np.log(self.get_time())



# initializes environment with ai player using random controller, playing against static enemy
# initializes simulation for coevolution evolution mode.
env = environm(experiment_name=experiment_name,
			   enemies=[1, 2, 3, 4, 5, 6, 7, 8],
			   playermode="ai",
               multiplemode='yes',
			   player_controller=player_controller(),
               enemy_controller=enemy_controller(0),
            #    loadenemy="no",
			   level=2,
			   speed="fastest")


env.state_to_log() # checks environment state


ini = time.time()  # sets time marker

env.play()
# runs simulation
# def simulation(env,x1,x2):
# 	f,p,e,t = env.play(pcont=x1,econt=x2)
# 	return f
