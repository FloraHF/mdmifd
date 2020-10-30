import numpy as np
import os

import Envs.scenarios as scenarios
from Envs.environment import MultiAgentEnv

from reader import resid, read_gazebo_param, read_gazebo_state, read_gazebo_cmd, read_gazebo_cap, read_gazebo_assign
from replay import replay_fromstart
# from plotter import plot_assign, compare_traj

# resid = 'res_00_0_value'
res_dir = '/home/flora/mdmi_data/'

def get_metric(res_path):
	states_gazebo, tmin, tmax  = read_gazebo_state(res_path=res_path)
	params = read_gazebo_param(res_path=res_path)

	# replay the gazebo simulation
	scenario = scenarios.load('game_mdmi').Scenario()
	world = scenario.make_world(r=params['r'], nd=params['nd'], ni=params['ni'], 
								vi=params['vi'], vd=params['vd'],
	   	                 		Rt=params['Rt'], Ro=params['Ro'],
	   	                 		xds=params['xds'], xis=params['xis'],
	                    		resid=resid)
	env = MultiAgentEnv(world, scenario.reset_world, 
						observation_callback=scenario.observation,
						state_callback=scenario.state, 
	                    info_callback=None, 
	                    done_callback=scenario.done_callback_defender)

	# print(np.linspace(tmin, tmax, 30))
	states_simple, tmax_s = replay_fromstart(env, scenario.dstrategy, tmin=tmin)

	tlmin_gazebo = 100
	for p, s in states_gazebo.items():
		x = np.array([s['x'](tmax), s['y'](tmax)])
		tl = world.target.level(x)
		tlmin_gazebo = min(tl, tlmin_gazebo)

	tlmin_simple = 100
	for p, s in states_simple.items():
		x = np.array([s['x'](tmax_s), s['y'](tmax_s)])
		tl = world.target.level(x)
		tlmin_simple = min(tl, tlmin_simple)
	
	return tmax, tmax_s, tlmin_gazebo, tlmin_simple


if __name__ == '__main__':
	for i in next(os.walk(res_dir))[1]:
		print(i)
		tmax, tmax_s, tlmin_gazebo, tlmin_simple = get_metric(res_path=res_dir+i+'/')
		print(tmax, tmax_s, tlmin_gazebo, tlmin_simple)
