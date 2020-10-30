import numpy as np

import Envs.scenarios as scenarios
from Envs.environment import MultiAgentEnv

from reader import resid, read_gazebo_param, read_gazebo_state, read_gazebo_cmd, read_gazebo_cap, read_gazebo_assign
from replay import replay_fromstart
from plotter import plot_assign, compare_traj

# read parameters
states_gazebo, tmin, tmax  = read_gazebo_state()
cmd_gazebo, tmin, tmax  = read_gazebo_cmd(tmin=tmin, tmax=tmax)
cap_gazebo, tmax = read_gazebo_cap(tmax=tmax)
# print(cap_gazebo)
params = read_gazebo_param()

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
assign_gazebo, tc_gazebo = read_gazebo_assign('/Itarg.csv')
assign_simple, tc_simple = read_gazebo_assign('/Itarg_pn.csv', toffset=tmin)

# ################# plot trajectories #################
compare_traj( np.linspace(tmin, tmax, 30), states_gazebo, assign_gazebo,
			np.linspace(tmin, tmax_s, 30), states_simple, assign_simple,
			cap_gazebo, params)

# ################# plot assignments ################# 
# assign_gazebo, tc_gazebo = read_gazebo_assign('/Itarg.csv')
# assign_simple, tc_simple = read_gazebo_assign('/Itarg_pn.csv', toffset=tmin)
# ax = plot_assign(assign_gazebo, cap_gazebo)
# ax = plot_assign(assign_simple, cap_gazebo)