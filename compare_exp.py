import argparse
import numpy as np

import Envs.scenarios as scenarios
from Envs.environment import MultiAgentEnv

from reader import Reader
from replay import replay_fromstart
from plotter import Plotter

parser = argparse.ArgumentParser()
parser.add_argument('datafile', default='data_00', type=str, help='data file name')
args = parser.parse_args()
res_path = '/home/flora/crazyflie_mdmifd_expdata/' + args.datafile + '/'


data = Reader(res_path)
params 					= data.read_exp_param()
states_exp, tmin, tmax  = data.read_state()
# print(tmin, tmax)
cmd_exp, tmin, tmax  	= data.read_cmd(tmin=tmin, tmax=tmax)
# print(tmin, tmax)
cap_exp, tmax 			= data.read_cap(tmax=tmax)
# print(tmin, tmax)
assign_exp, tc_exp 		= data.read_assign('/Itarg.csv')
plot = Plotter(res_path, params)

# replay the experiment using simple dynamics
scenario = scenarios.load('game_mdmi').Scenario()
world = scenario.make_world(r=params['r'], nd=params['nd'], ni=params['ni'], 
							xt=params['x0targ'], yt=params['y0targ'], Rtarg=params['Rtarg'],
							vi=params['vi'], vd=params['vd'],
   	                 		Rt=params['Rt'], Ro=params['Ro'],
   	                 		xds=params['xds'], xis=params['xis'],
                    		datadir=res_path, mode='exp')
env = MultiAgentEnv(world, scenario.reset_world, 
					observation_callback=scenario.observation,
					state_callback=scenario.state, 
                    info_callback=None, 
                    done_callback=scenario.done_callback_defender)

states_simple, tmax_s, cap_simple = replay_fromstart(env, scenario.dstrategy, tmin=tmin)
assign_simple, tc_simple = data.read_assign('/Itarg_pn.csv', toffset=tmin)

################# plot trajectories #################
plot.compare_traj(np.linspace(tmin, tmax, 100), states_exp, assign_exp, cap_exp,
			 np.linspace(tmin, tmax_s, 30), states_simple, assign_simple, cap_simple)

# plot.velocity_response(np.linspace(tmin, tmax, 100), cmd_exp, states_exp)
plot.plot_assign(assign_exp, cap_exp)
plot.plot_assign(assign_simple, cap_simple)