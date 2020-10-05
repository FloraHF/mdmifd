import numpy as np

import Envs.scenarios as scenarios
from Envs.environment import MultiAgentEnv

from reader import resid, read_gazebo_param, read_gazebo_state, read_gazebo_cmd, read_gazebo_cap, read_gazebo_assign
from replay import replay_follow
from plotter import plot_assign, compare_cmdv, velocity_response, compare_traj_and_v

# read parameters
states_gazebo, tmin, tmax  = read_gazebo_state()
cmd_gazebo, tmin, tmax  = read_gazebo_cmd(tmin=tmin, tmax=tmax)
cap_gazebo, tmax = read_gazebo_cap(tmax=tmax)
params = read_gazebo_param()

# replay the gazebo simulation
scenario = scenarios.load('game_mdmi').Scenario()
world = scenario.make_world(r=params['r'], nd=params['nd'], ni=params['ni'], 
							vi=params['vi'], vd=params['vd'],
   	                 		Rd=params['Rt'], Ri=params['Ro'],
   	                 		xds=params['xds'], xis=params['xis'],
   	                 		resid=resid)
env = MultiAgentEnv(world, scenario.reset_world, 
					observation_callback=scenario.observation,
					state_callback=scenario.state, 
                    info_callback=None, 
                    done_callback=scenario.done_callback_defender)

ts = np.linspace(tmin, tmax, 50)
cmd_simple = replay_follow(env, np.linspace(tmin, tmax, 50), 
							scenario.dstrategy, states_gazebo)

# ###############################################################
# |						Bellow are plots						|
# ###############################################################

# compare command velocity
# compare_cmdv(np.linspace(tmin, tmax, 50), cmd_gazebo, cmd_simple)

# velocity and command velocity
# velocity_response(np.linspace(tmin, tmax, 50), cmd_gazebo, states_gazebo)

# trajectory of gazebo and command velocity
# compare_traj_and_v(ts, states_gazebo, cmd_simple, cap_gazebo, params)

# plot assignment and efficiencies
assign_gazebo, tc_gazebo = read_gazebo_assign('/Itarg.csv')
assign_simple, tc_simple = read_gazebo_assign('/Itarg_pn.csv')
# plot_assign(assign_gazebo, cap_gazebo)
plot_assign(assign_simple, cap_gazebo)
