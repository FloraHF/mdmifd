import numpy as np

import Envs.scenarios as scenarios
from Envs.environment import MultiAgentEnv

from reader import resid, read_gazebo_param, read_gazebo_state, read_gazebo_cmd, read_gazebo_cap, read_gazebo_assign
from replay import replay_follow
from plotter import plot_assign, compare_cmdv, velocity_response, compare_traj_and_v

# read parameters
states_gazebo, tmin, tmax  = read_gazebo_state()
# print(tmin, tmax)
cmd_gazebo, tmin, tmax  = read_gazebo_cmd(tmin=tmin, tmax=tmax)
# print(tmin, tmax)
cap_gazebo, tmax = read_gazebo_cap(tmax=tmax)
# print(tmin, tmax)
params = read_gazebo_param()
# print(tmin, tmax)

# replay the gazebo simulation
scenario = scenarios.load('game_mdmi').Scenario()
world = scenario.make_world(r=params['r'], nd=params['nd'], ni=params['ni'], 
							vi=params['vi'], vd=params['vd'],
   	                 		Rd=params['Rt'], Ri=params['Ro'],
   	                 		xds=params['xds'], xis=params['xis'],
   	                 		resid=resid, iselect_mode=params['iselect_mode'])
env = MultiAgentEnv(world, scenario.reset_world, 
					observation_callback=scenario.observation,
					state_callback=scenario.state, 
                    info_callback=None, 
                    done_callback=scenario.done_callback_defender)

ts = np.linspace(tmin, tmax, 100)
cmd_simple = replay_follow(env, ts, 
							scenario.dstrategy, states_gazebo)


assign_gazebo, tc_gazebo = read_gazebo_assign('/Itarg.csv')
assign_simple, tc_simple = read_gazebo_assign('/Itarg_pn.csv')
# ###############################################################
# |						Bellow are plots						|
# ###############################################################

# compare command velocity
compare_cmdv(ts, cmd_gazebo, cmd_simple)

# velocity and command velocity
velocity_response(ts, cmd_gazebo, states_gazebo)

# trajectory of gazebo and command velocity
# print(ts)
compare_traj_and_v(ts, states_gazebo, assign_simple, cap_gazebo, params, name='simple')
compare_traj_and_v(ts, states_gazebo, assign_gazebo, cap_gazebo, params, name='gazebo')

# plot assignment and efficiencies
plot_assign(assign_gazebo, cap_gazebo)
plot_assign(assign_simple, cap_gazebo)
