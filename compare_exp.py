import numpy as np

import Envs.scenarios as scenarios
from Envs.environment import MultiAgentEnv

from reader import read_exp_param, read_gazebo_state, read_gazebo_cmd, read_gazebo_cap, read_gazebo_assign
from replay import replay_fromstart
from plotter import compare_traj, velocity_response

# result file
resid = 'data_20'
res_path = '/home/flora/crazyflie_mdmifd_expdata/' + resid + '/'

# read data
params = read_exp_param(res_path=res_path)
states_exp, tmin, tmax  = read_gazebo_state(res_path=res_path)
# print('from reading states', tmin, tmax)
cmd_exp, tmin, tmax  = read_gazebo_cmd(tmin=tmin, tmax=tmax)
# print('from reading cmd', tmin, tmax)
cap_exp, tmax = read_gazebo_cap(tmax=tmax)
# print('from reading cap', tmin, tmax)
assign_exp, tc_exp = read_gazebo_assign('/Itarg.csv')

# print(params)

# replay the experiment using simple dynamics
scenario = scenarios.load('game_mdmi').Scenario()
world = scenario.make_world(r=params['r'], nd=params['nd'], ni=params['ni'], 
							xt=params['x0targ'], yt=params['y0targ'], Rtarg=params['Rtarg'],
							vi=params['vi'], vd=params['vd'],
   	                 		Rt=params['Rt'], Ro=params['Ro'],
   	                 		xds=params['xds'], xis=params['xis'],
                    		resid=resid, mode='exp')
env = MultiAgentEnv(world, scenario.reset_world, 
					observation_callback=scenario.observation,
					state_callback=scenario.state, 
                    info_callback=None, 
                    done_callback=scenario.done_callback_defender)

# print(np.linspace(tmin, tmax, 30))
states_simple, tmax_s, cap_simple = replay_fromstart(env, scenario.dstrategy, tmin=tmin)
assign_simple, tc_simple = read_gazebo_assign('/Itarg_pn.csv', res_path=res_path, toffset=tmin)
# print(assign_simple)

################# plot trajectories #################
compare_traj(np.linspace(tmin, tmax, 100), states_exp, assign_exp, cap_exp,
			 np.linspace(tmin, tmax_s, 30), states_simple, assign_simple, cap_simple,
			 params)

# velocity_response(np.linspace(tmin, tmax, 100), cmd_exp, states_exp)