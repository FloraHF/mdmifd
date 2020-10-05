import numpy as np
from geometries import dist, DominantRegion
from math import pi, atan2
import itertools

from geometries import dist, norm

class DstrategyMinDR(object):
	"""docstring for DstrategyMinDREffort"""
	def __init__(self):
		self.reset()

	def __call__(self, agent, world):

		assert 'D' in agent.name
		action = Action()

		if agent.state.o is None:
			speed = norm(agent.state.p_vel)
			action.u = agent.state.p_vel/speed if speed > 0 else np.array([0, 0])
		else:
			orders = [order for order in itertools.permutations(list(agent.state.o))]
			values = [world.value_order(list(order), Dind) for order in orders]
			k = values.index(max(values))
			agent.state.o = orders[k][0]

			vi = world.agents[-1].max_speed
			vd = agent.max_speed
			xi = world.intruders[i].state.p_pos
			xd = agent.state.p_pos

			dr = DominantRegion(world.target.size, vd/vi, xi, [xd], offset=0)
			xw = world.target.deepest_point_in_dr(dr)
			dx = xw - xd
			action.u = agent.u_range*dx/norm(dx)
		return action

	def reset(self, agent):
		agent.state.o = None