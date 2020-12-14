import os
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from Envs.scenarios.game_mdmi.utils import prefstring_to_list


class Reader(object):
	"""docstring for Reader"""
	def __init__(self, res_path):

		# result path
		# self.res_path = '/home/flora/crazyflie_mdmifd_expdata/' + resid + '/'
		self.res_path = res_path
		
		# find out players recorded, and sort by their id
		players = [p for p in next(os.walk(self.res_path))[1]]
		self.defenders = sorted([p for p in players if 'D' in p], key=lambda x: int(x[1:]))
		self.intruders = sorted([p for p in players if 'I' in p], key=lambda x: int(x[1:]))
		self.players = self.defenders + self.intruders

	# read player parameters
	def read_gazebo_param(self):
		params = dict()
		for p in self.players:
			data = pd.read_csv(self.res_path + p + '/param.csv')
			data = data.set_index('param').T
			params[p] = data.to_dict('records')[0]
			for k in params[p]:
				if 'target:' in k:
					if 'type' in k:
						params[p][k] = params[p][k]
					else:
						params[p][k] = float(params[p][k])
				elif k != 'id' and k != 'iselect_mode':
					if k == 'ni' or k == 'nd':
						params[p][k] = int(params[p][k])
					else:
						params[p][k] = float(params[p][k])
				else:
					params[p][k] = params[p][k]

		xds=[np.array([params[d]['x0'], params[d]['y0']]) for d in self.defenders]
		xis=[np.array([params[i]['x0'], params[i]['y0']]) for i in self.intruders]

		return {'r': params['D0']['r'] 	,
				'nd': params['D0']['nd'],	# same for all players
				'ni': params['D0']['ni'],	# same for all players
				'Rt': params['D0']['Rteam'],
				'Ro': params['D0']['Roppo'],
				'vd': params['D0']['vmax'],	# same for all defenders
				'vi': params['I0']['vmax'],	# same for all intruders
				'xds': xds,
				'xis': xis,
				'x0targ': params['D0']['target:x0'],
				'y0targ': params['D0']['target:y0'],
				'Rtarg': params['D0']['target:R'],
				'iselect_mode': params['D0']['iselect_mode']}

	# read player parameters, experiment
	def read_exp_param(self):
		params = dict()
		for p in self.players:
			data = pd.read_csv(self.res_path + p + '/param.csv')
			data = data.set_index('param').T
			params[p] = data.to_dict('records')[0]
			# print(params[p])
			for k in params[p]:
				if 'target:' in k:
					if 'type' in k:
						params[p][k] = params[p][k]
					else:
						params[p][k] = float(params[p][k])
				elif k != 'id' and k != 'iselect_mode' and 'dict' not in k:
					if k == 'ni' or k == 'nd':
						params[p][k] = int(params[p][k])
					else:
						params[p][k] = float(params[p][k])
				else:
					params[p][k] = params[p][k]

		cf_dict = dict()
		for r_cf in params[p]['cf_dict'].split('!'):
			r_cf = r_cf.split('_')
			cf_dict.update({r_cf[0]:r_cf[1]})

		xds=[np.array([params[d]['x0'], params[d]['y0']]) for d in self.defenders]
		xis=[np.array([params[i]['x0'], params[i]['y0']]) for i in self.intruders]

		return {'r': params['D0']['r'] 	,
				'nd': params['D0']['nd'],	# same for all players
				'ni': params['D0']['ni'],	# same for all players
				'Rt': params['D0']['Rteam'],
				'Ro': params['D0']['Roppo'],
				'vd': params['D0']['vmax'],	# same for all defenders
				'vi': params['I0']['vmax'],	# same for all intruders
				'xds': xds,
				'xis': xis,
				'x0targ': params['D0']['target:x0'],
				'y0targ': params['D0']['target:y0'],
				'Rtarg': params['D0']['target:R'],
				'cf_dict': cf_dict,
				'iselect_mode': params['D0']['iselect_mode']}

	# read players states
	def read_state(self, tmin=0., tmax=10e10):
		states = dict()
		for p in self.players:
			data = pd.read_csv(self.res_path + p + '/State.csv')
			t = data['t'].to_numpy()
			# tmin_temp = min(t)
			# t = t - tmin_temp
			tmin = max(min(t), tmin)
			tmax = min(max(t), tmax)
			# print(max(data['x']))
			# tmax = min(t[-1], tmax)
			states[p] = {k:interp1d(t, data[k].to_numpy()) for k in ['x', 'y', 'z', 'vx', 'vy']}
			states[p].update({'tmin': t[0]})
			# print(min(states_gazebo[p]['x'](t)), max(states_gazebo[p]['x'](t)))
			# print(t[-1], tmax)
			# print('from reading', p, tmin, tmax)

		return states, tmin, tmax

	# read player commands
	def read_cmd(self, tmin=0., tmax=10e10):
		cmd = dict()
		for p in self.players:
			data = pd.read_csv(self.res_path + p + '/Command.csv')
			t = data['t'].to_numpy()
			# tmin_temp = min(t)
			# t = t - tmin_temp
			tmin = max(min(t), tmin)
			tmax = min(max(t), tmax)
			# tmax = min(t[-1], tmax)
			cmd[p] = {k:interp1d(t, data[k].to_numpy()/40) for k in ['vx', 'vy']}
		return cmd, tmin, tmax


	def read_cap(self, tmax=10e10):
		cap = {i:{'dcap': None, 'tcap': np.inf, 'tent': np.inf} for i in self.intruders}
		maxte = 0
		ncap, nent = 0, 0
		for i in self.intruders:
			cap_data = pd.read_csv(self.res_path + i + '/Dcap.csv')
			ent_data = pd.read_csv(self.res_path + i + '/Tent.csv')
			if not ent_data['t'].empty:
				# print(i, 'enters at', ent_data['t'].values[-1])
				cap[i]['tent'] = ent_data['t'].values[-1]
				maxte = max(maxte, ent_data['t'].values[-1])
				nent += 1
			else:
				if not cap_data['t'].empty:
					# print(i, 'is captured at', cap_data['t'].values[-1])
					cap[i]['dcap'] = cap_data['d'].values[-1]
					cap[i]['tcap'] = cap_data['t'].values[-1]
					# print(i, cap[i]['tcap'])
					maxte = max(maxte, cap_data['t'].values[-1])
					ncap += 1

		print('ncap: ', ncap, 'nent: ', nent)

		return cap, min(maxte, tmax)


	def read_assign(self, data_file, toffset=0):
		assign = dict()
		tc = 0
		for d in self.defenders:
			data = pd.read_csv(self.res_path + d + data_file)
			if not data.empty:	# print(data)
				t = data['t'].to_numpy() + toffset
				# t = t - min(t)
				i = np.array([int(ii[1:]) for ii in data['i'].to_list()])
				e = data['e'].to_numpy()
				pref = [prefstring_to_list(pstr) for pstr in data['pref']]
				# print(pref)
				assign[d] = {'t': t,
							 'i': i,
							 'e': e,
							 'pref': pref,
							 'approx': interp1d(t, i)}
				tc = max(tc, t[-1])
		return assign, tc

if __name__ == '__main__':
	# p = read_gazebo_param()
	# # print(p)
	# s, tmin, tmax = read_gazebo_state()
	# s, tmin, tmax = read_gazebo_cmd()
	# cap = read_gazebo_cap()
	ass, tc = read_gazebo_assign('/Itarg_pn.csv')
	for d, a in ass.items():
		print(a['t'])


