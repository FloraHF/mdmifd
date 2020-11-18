# Flora Fu  @20200709
import numpy as np
import itertools
from math import pi, sin, cos
from copy import deepcopy

from Envs.scenarios.game_mdmi.geometries import CircleTarget, DominantRegion, dist, norm
from Envs.scenario import BaseScenario
from Envs.core import Game, Agent, Landmark, Action

from Envs.scenarios.game_mdmi.utils import state_to_prefstring
# xds = [np.array([3., -2.]), np.array([2.5, 1.])]
# xds = [np.array([3., -2.])]
# xis = [np.array([0., 0.]),  np.array([0., 1.]), np.array([1., 2.])]
# vd = 1.
# vi = .8
dcolors = [np.array([0.1, 1., 0.05]),
		   np.array([0.4, 0.05, 1.]),
		   np.array([0.8, 0.05, 1.]),
		   np.array([0.05, 0.5, 0.1]),
		   np.array([.7, 1., .5])]

class Scenario(BaseScenario):
	def make_world(self, r=.6, 
					nd=1, ni=1, 
					vd=1., vi=.8, 
					Rt=5., Ro=5., 
					xds=None, xis=None, 
					resid=1, 
					iselect_mode='value',
					mode='gazebo',
					overlap=0.,
					tht=[.5, 1.]):

		world = Game()
		world.mode = mode
		# world properties
		world.nd = nd
		world.ni = ni
		world.vd = vd
		world.vi = vi
		world.nl = 1 

		# initial location generation
		world.overlap = overlap
		world.tht_lb = tht[0]
		world.tht_ub = tht[1]

		# add agents
		world.agents = [Agent(d) for d in range(world.nd)] + [Agent(i) for i in range(world.ni)]
		# for a in world.agents:
		# 	print('in world', a.size)		
		for i, agent in enumerate(world.agents):
			# print(agent)
			if i < world.nd:
				agent.name = 'agent D%d' % i
				agent.iselect_mode = iselect_mode
				agent.r = r
				agent.done_callback = self.done_callback_defender
				agent.silent = False
				agent.Rd = Rt
				agent.Ri = Ro
			else:
				agent.name = 'agent I%d' % (i-world.nd)
				agent.done_callback = self.done_callback_intruder
				agent.enter_callback = self.is_enter
				agent.capture_callback = self.is_capture				
				agent.silent = True		
				# agent.Rd = Ro
				# agent.Ri = Rt
			# agent.size = 0

		world.r = world.intruders[0].size + world.defenders[0].r
		world.target = CircleTarget(1.25)
		world.target.name = 'target 0'
		world.target.color = np.array([.0, .0, .9])
		world.landmarks = [world.target]
		# print(xds)
		self.datadir = '/home/flora/mdmi_data/' + str(resid) + '/'
		self.reset_world(world, xds=xds, xis=xis)

		if world.mode == 'gazebo':
			for d in world.defenders:
				with open(self.datadir + '/D' + str(d.id) + '/Itarg_pn.csv', 'w') as f:
					f.write('t,i,e,pref\n')
				with open(self.datadir + '/D' + str(d.id) + '/Itarg_op.csv', 'w') as f:
					f.write('t,i,e,pref\n')

		return world

	def reset_world(self, world, t=0., xds=None, xis=None, actives=None, evend=False):
		
		world.t = t
		# print(actives)

		for i, agent in enumerate(world.agents):
			if 'D' in agent.name:
				agent.color = dcolors[i%5]
				agent.max_speed = world.vd
				agent.u_range = world.vd
				agent.state.a = True
				agent.iprev = None
				
				if xds is not None:
					agent.state.p_pos = xds[agent.id]
				else:
					collide = True
					while collide:
						# print(i, 'new position due to collision')
						k = i if evend else None
						# agent.state.p_pos = np.random.uniform(low=0, high=5, size=(2,))
						if i == 0:
							agent.state.p_pos, r = self.generate_player_pos(world, 1., 2+world.overlap, k=k)
						else:
							if evend: agent.state.p_pos, r = self.generate_player_pos(world, 1., 2+world.overlap, r=r, k=k)
							else: agent.state.p_pos, r = self.generate_player_pos(world, 1., 2+world.overlap, k=k)

						for other in world.agents[:i]: # agents 0-i are all defenders
							# print('generating world', other.size)
							collide = self.is_collision(other, agent)
							if collide: 
								# print(other.state.p_pos, agent.state.p_pos)
								break
						if i == 0: collide = False
			else:				
				agent.color = np.array([0.1, 1., 1.])
				agent.max_speed = world.vi
				agent.u_range = world.vi
				agent.action_callback = self.istrategy
				agent.state.e = False
				agent.mem.e = False

				if xis is not None:
					agent.state.p_pos = xis[agent.id]
				else:
					collide = True
					while collide:
						# print(i, 'new position due to collision')
						agent.state.p_pos, _ = self.generate_player_pos(world, 2., 4.)
						# agent.state.p_pos = np.random.uniform(low=0, high=5, size=(2,))
						for other in world.agents[:i]:
							# print(other.size)
							collide = self.is_inrange(agent, other) if 'D' in other.name else self.is_collision(agent, other)
							if collide: break

				if actives is not None:
					agent.state.a = actives[agent.id]
				else:
					agent.state.a = True

			agent.state.p_vel = np.zeros(world.dim_p)
			agent.state.c = np.zeros(world.dim_c)
			agent.state.n = []
			agent.mem.n = []
			agent.mem.init_p_pos = np.array([x for x in agent.state.p_pos])
			
			# agent.size = .1
			agent.initial_mass = 1.

			# print(agent.name, agent.max_speed)

		for defender in world.defenders:
			world.update_neighbours_defender(defender)
		for intruder in world.intruders:
			world.update_neighbours_intruder(intruder)

	# def set_eselect_mode(self, world, mode):
		

	def generate_player_pos(self, world, lb, ub, r=None, k=None):
		# world.target.state.p_pos
		if r is None: r = np.random.uniform(lb*world.target.size, ub*world.target.size)
		if k is None: 
			tht = np.random.uniform(world.tht_lb*pi, world.tht_ub*pi)
		else:
			tht = world.tht_lb*pi + k*(world.tht_ub - world.tht_lb)*pi/(world.nd-1)

		return world.target.state.p_pos + np.array([r*cos(tht), r*sin(tht)]), r

	def done_callback_intruder(self, agent, world):
		# print(agent.state.e or agent.state.n)
		# print(agent.name, agent.state.e)
		return agent.state.e or bool(len(agent.state.n))

	def done_callback_defender(self, agent, world):
		# allcap = True
		for intruder in world.intruders:
			if intruder.state.a:
				return False
		# print('allcap = ', allcap)
		return True

	def is_enter(self, agent, world):
		tlevel = world.target.level(agent.state.p_pos)
		return True if tlevel < 0 else False

	def is_inrange(self, intruder, defender):
		delta_pos = intruder.state.p_pos - defender.state.p_pos
		dist = np.sqrt(np.sum(np.square(delta_pos)))
		dist_min = intruder.size + defender.r
		# print(intruder.state.p_pos)
		# print(dist_min)
		return True if dist < dist_min else False

	def is_capture(self, agent, world):
		ds = []
		for defender in world.defenders:
			if self.is_inrange(agent, defender):
				# return True
				ds.append(defender.id)
		return ds

	def is_collision(self, agent1, agent2):
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.sqrt(np.sum(np.square(delta_pos)))
		dist_min = agent1.size + agent2.size
		# print(agent1.name, agent1.size, agent2.name, agent2.size)
		return True if dist < dist_min else False

	def dist_calc(self, agent1, agent2):
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.sqrt(np.sum(np.square(delta_pos)))
		dist_min = agent1.size + agent2.size;
		return dist, dist_min

	def value_order(self, defender, iorder, world):
		vd = defender.max_speed
		vi = world.intruders[0].max_speed
		def recurse(xis, actives, xd):
			if sum(actives) == 0:
				return xis, actives, xd

			for i in range(len(xis)):
				if actives[i]:
					dr = DominantRegion(world.r, vd/vi, xis[i], [xd], offset=0)
					xw = world.target.deepest_point_in_dr(dr)
					dt = dist(xw, xis[i])/vi
					xd = xd + dt*vd*(xw - xd)/dist(xw, xd)
					xis[i] = np.array([x for x in xw])
					actives[i] = 0
					break
			for j in range(i+1,len(xis)):
				if actives[j]:
					dr = DominantRegion(world.r, vd/vi, xis[j], [xd], offset=vd*dt)
					xw = world.target.deepest_point_in_dr(dr)
					e = (xw - xis[j])/dist(xw, xis[j])
					xi = xis[j] + e*vi*dt
					xis[j] = np.array([x for x in xi])
			return recurse(xis, actives, xd)

		xis = [np.array([x for x in world.intruders[i].state.p_pos]) for i in iorder]
		actives = [int(world.intruders[i].state.a) for i in iorder]
		# print(actives)
		xd = np.array([x for x in defender.state.p_pos])

		xis, _, _ = recurse(xis, actives, xd)

		return min([world.target.level(xw) for xw in xis])

	def value(self, world):
		# for i in world.intruders:
		# 	assert not i.state.a
		return min([world.target.level(i.state.p_pos) for i in world.intruders])

	def dstrategy(self, agent, world):

		assert 'D' in agent.name
		# print(agent.state.o)

		if not agent.state.o:
			dx = agent.mem.init_p_pos - agent.state.p_pos
			dis = norm(dx)
			# u = 0.3*agent.u_range*dx/dis - agent.state.p_vel if dis > 0 else np.array([0., 0.])
			u = 0.5*agent.u_range*dx/dis if dis > 0 else np.array([0., 0.])
			
			return u

		# select the current intruder by value
		if agent.iselect_mode == 'value':
			orders = [order for order in itertools.permutations(agent.state.o)]
			values = [self.value_order(agent, order, world) for order in orders]
			icurr = orders[values.index(max(values))][0]

		elif agent.iselect_mode == 'emin':
			if agent.iprev not in agent.state.o: # include self.iprev == None
				maxe_ind = agent.state.f.index(min(agent.state.f))
				icurr = agent.state.o[maxe_ind]
				agent.iprev = icurr
			else:
				icurr = agent.iprev

		elif agent.iselect_mode == 'emax':
			if agent.iprev not in agent.state.o: # include self.iprev == None
				maxe_ind = agent.state.f.index(max(agent.state.f))
				icurr = agent.state.o[maxe_ind]
				agent.iprev = icurr
			else:
				icurr = agent.iprev

		eff = agent.state.f[agent.state.o.index(icurr)]


		if world.mode == 'gazebo':
			if agent.state.s == 'pwn':
				with open(self.datadir + '/D' + str(agent.id) + '/Itarg_pn.csv', 'a') as f:
					f.write('%.2f,%s,%.6f,%s\n'%(world.t, 'I'+str(icurr), eff, state_to_prefstring(agent.state)))
			if agent.state.s == 'opt':
				with open(self.datadir + '/D' + str(agent.id) + '/Itarg_op.csv', 'a') as f:
					f.write('%.2f,%s,%.6f,%s\n'%(world.t, 'I'+str(icurr), eff, state_to_prefstring(agent.state)))

		# compute the heading to capture intruder icurr
		xi = world.intruders[icurr].state.p_pos
		xd = agent.state.p_pos
		vi = world.agents[-1].max_speed
		vd = agent.max_speed
		dr = DominantRegion(world.r, vd/vi, xi, [xd], offset=0)
		xw = world.target.deepest_point_in_dr(dr)
		dx = xw - xd
		return agent.u_range*dx/norm(dx)

	def istrategy(self, agent, world):

		assert 'I' in agent.name
		action = Action()
		action.u = np.array([0., 0.])

		xds = []
		for D in world.agents:
			if 'D' in D.name:
				xds.append(np.array([x for x in D.state.p_pos]))
		# print(xds)
		# for D in agent.neigh_d:
		# 	xds.append(np.array([x for x in world.defenders[D].state.p_pos]))
		vd = world.agents[0].max_speed

		if agent.state.a:
			dr = DominantRegion(world.r, vd/agent.max_speed, agent.state.p_pos, xds, offset=0)
			xw = world.target.deepest_point_in_dr(dr)
			dx = xw - agent.state.p_pos
			dist = norm(dx)
			if dist > 1e-6:
				action.u = dx/dist
			# speed = norm(agent.state.p_vel)
			# action.u = -agent.state.p_vel/speed if speed > 0 else np.array([0, 0])
		action.u *= agent.u_range
		return action

	def reward_test(self, agent, world):
		return np.random.random()

	def reward_dist(self, agent, world):
		assert 'D' in agent.name
		ds = []
		for intruder in world.intruders:
			dmin = min([self.dist_calc(d, intruder)[0] for d in world.defenders])
			# print(dmin)
			ds.append(-dmin/5)
		return np.mean(ds)

	def reward_team(self, agent, world):
		assert 'D' in agent.name
		r = 0
		for intruder in world.intruders:
			r -= (intruder.state.e - intruder.mem.e)
			r += (int(bool(len(intruder.state.n))) - int(bool(len(intruder.mem.n))))
		return r

	def reward_ind(self, agent, world):
		assert 'D' in agent.name
		r = -.01
		for i in agent.neigh_i:
			r -= (world.intruders[i].state.e - world.intruders[i].mem.e)
		return r + 2*(len(agent.state.n) - len(agent.mem.n))

	def observation(self, agent, world):
		other_pos = []
		other_vel = []
		for other in world.agents:
			if other is agent: continue
			other_pos.append(other.state.p_pos - agent.state.p_pos)
			other_vel.append(other.state.p_vel - agent.state.p_vel)
		return np.concatenate([agent.state.p_pos]+[agent.state.p_vel]+other_pos+other_vel+world.target.state.p_pos)

	def observation_swarm(self, agent, world):
		dstate = []
		for other in world.defenders:
			if other is agent: continue
			dstate.append(other.state.p_pos - agent.state.p_pos)
			dstate.append(other.state.p_vel - agent.state.p_vel)
			if not other.silent:
				dstate.append(other.state.c)
			dstate.append([int(other.id in agent.neigh_d)])
		istate = []
		for other in world.intruders:
			istate.append(other.state.p_pos - agent.state.p_pos)
			istate.append(other.state.p_vel - agent.state.p_vel)
			istate.append([int(other.id in agent.neigh_i)])

		return np.concatenate([world.target.state.p_pos-agent.state.p_pos] + 
								[agent.state.p_pos] + [agent.state.p_vel] +
								dstate + istate)

	def state(self, world):
		dstate = []
		for d in world.defenders:
			dstate.append(d.state.p_pos)

		istate = []
		active = []
		for i in world.intruders:
			istate.append(i.state.p_pos)
			active.append([i.state.a])

		return np.concatenate(dstate+istate+active)
			






