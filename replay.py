import numpy as np
from scipy.interpolate import interp1d

from Envs.scenarios.game_mdmi.astrategy import knapsack_assign, negotiate_assign, extended_negotiation

def replay_follow(env, ts, dstrategy, states_gazebo):
	# interpolate states_gazebo into xy_gazebo_re according to ts,
	# compute the command velocity and write to file the assignment results of the simple simulation

	defenders = ['D'+str(d) for d in range(env.world.nd)]
	intruders = ['I'+str(i) for i in range(env.world.ni)]

	cmd_simple = {d:{'vx':[], 'vy':[]} for d in defenders}

	for t in ts:
		# print(t)
		xds = [np.array([states_gazebo[d]['x'](t), states_gazebo[d]['y'](t)]) for d in defenders]
		xis = [np.array([states_gazebo[i]['x'](t), states_gazebo[i]['y'](t)]) for i in intruders]
		actives = [states_gazebo[i]['z'](t)>0.5 for i in intruders]

		env.reset(t=t, xds=xds, xis=xis, actives=actives)

		_, _ = negotiate_assign(env.world, firstassign=True)

		for d in env.world.defenders:
			action = dstrategy(d, env.world) # scenario.dstrategy writes the assignment result to file
			cmd_simple['D'+str(d.id)]['vx'].append(action[0])
			cmd_simple['D'+str(d.id)]['vy'].append(action[1])

	return cmd_simple


def replay_fromstart(env, dstrategy, tmin):
	# interpolate states_gazebo into xy_gazebo_re according to ts,
	# compute the command velocity and write to file the assignment results of the simple simulation

	defenders = ['D'+str(d) for d in range(env.world.nd)]
	intruders = ['I'+str(i) for i in range(env.world.ni)]

	states_simple = {p:{'x':[], 'y':[]} for p in defenders+intruders}
	ts = [env.world.t + tmin]

	state = env.get_state() # make sure the order of players in state and players match
	for i, p in enumerate(defenders+intruders): 
		states_simple[p]['x'].append(state[2*i])
		states_simple[p]['y'].append(state[2*i+1])

	for k in range(100):

		_, _ = negotiate_assign(env.world, firstassign=True)
		actions = [dstrategy(d, env.world) for d in env.world.defenders]
		_, _, done_n, _ = env.step(actions)
		
		state = env.get_state() # make sure the order of players in state and players match
		ts.append(env.world.t + tmin)
		for i, p in enumerate(defenders+intruders): 
			states_simple[p]['x'].append(state[2*i])
			states_simple[p]['y'].append(state[2*i+1])

		if all(done_n):
			break
	# print('replay:', ts)

	return { p: {k:interp1d(ts, states_simple[p][k]) 
			 	for k in ['x', 'y']} 
			 	for p in defenders+intruders}, ts[-1]