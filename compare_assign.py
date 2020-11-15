#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 6th, 2020
@author: Flora Fu

"""

import os
from copy import deepcopy
import numpy as np
from math import cos, sin, pi

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rc
rc('text', usetex=True)

import pandas as pd
from sklearn.linear_model import LinearRegression

# import cv2
import pickle

from Envs.environment import MultiAgentEnv
import Envs.scenarios as scenarios
from Envs.scenarios.game_mdmi.astrategy import knapsack_assign, negotiate_assign, augmented_negotiation


MAX_EPISODES = 200
MAX_EP_STEPS = 100

PATH = './Logs/Geometric/'
scenario = scenarios.load('game_mdmi').Scenario()


def evaluate_assignment(r, nd, ni, vd, vi, log_PATH=PATH, n_ep=MAX_EPISODES, overlap=.5, tht=[0.1, 1.9]):
	for Rt in [1, 2, 3, 5., 7, 10]:
	# for Rt in [3]:
		# nstep = int(5/.5)
		# for Ro in np.linspace(1, 1+.5*nstep, nstep+1):
		# for Ro in [2.5]:
		for Ro in [rr for rr in [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 9] if rr<Rt+7. ]:
			log_path = os.path.join(log_PATH, 'D%dI%d'%(nd, ni), 'assign', 'Rd=%d_Ri=%d'%(Rt*100, Ro*100))
			if not os.path.exists(log_path): 
				os.makedirs(log_path)
			log_file = log_path + '/eff.csv'
			
			for i in range(n_ep):

				print(Rt, Ro, i, log_file)

				world = scenario.make_world(r=r, Rt=Rt, Ro=Ro, nd=nd, ni=ni, vd=vd, vi=vi, 
											mode='simple', overlap=overlap, tht=tht)
				env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_team,
									scenario.observation, state_callback=scenario.state,
									done_callback=scenario.done_callback_defender)
				env.reset(evend=False)	
				env_ = deepcopy(env)
				env__ = deepcopy(env)

				_, ek = knapsack_assign(env.world)
				_, en = negotiate_assign(env_.world)
				_, ea = augmented_negotiation(env__.world)
				# print(ek, en)
				state = list(map(str, env.get_state()))

				if not os.path.exists(log_file):
					with open(log_file, 'w') as f:
						f.write(','.join(['D%s:x,D%s:y'%(dd, dd) for dd in range(env.world.nd)] +\
										 ['I%s:x,I%s:y'%(ii, ii) for ii in range(env.world.ni)] +\
										 ['I%s:active'%ii for ii in range(env.world.ni)]+ \
										 ['ek'] + ['en'] + ['ea']) + '\n')		
										 				
				with open(log_file, 'a') as f:
					f.write(','.join(state + ['%.5f'%ek] + ['%.5f'%en] + ['%.5f'%ea])+'\n')


def plot_assign_statistics(res_path=PATH, nd=3, ni=12):

	res_path = os.path.join(res_path, 'D%dI%d'%(nd, ni), 'assign')
	colors = ['c', 'b', 'g', 'k', 'r', 'm']
	
	Rds, Ris, ns, kwins, nwins, ekave, enave = [], [], [], [], [], [], []
	# k = 0
	for i in next(os.walk(res_path))[1]:
		Rd, Ri = i.split('_')
		Rd = float(Rd.split('=')[-1])/100
		Ri = float(Ri.split('=')[-1])/100			

		if Rd not in Rds:
			Rds.append(Rd)
			Ris.append([])
			ns.append([])
			kwins.append([])
			nwins.append([])
			ekave.append([])
			enave.append([])
			# print(Rd)
			# k += 1

		data = pd.read_csv(res_path + '/' + i + '/eff.csv')

		Ris[Rds.index(Rd)].append(Ri)
		# print(Rd, Ris[k])

		ek_ = data['en'].to_list()
		en_ = data['ea'].to_list()
		# ea_ = data['ea'].to_list()

		ek, en = [], []
		n, kwin, nwin = 0, 0, 0
		for ek__, en__ in zip(ek_, en_):
			# print(ek__, en__)
			if ek__*en__ != 0:
				n += 1
				ek.append(ek__)
				en.append(en__)
				if ek__ > en__:
					kwin += 1
				elif ek__ < en__:
					nwin += 1
					# print(ek__, en__)
		# print(n, kwin, nwin)

		ns[Rds.index(Rd)].append(n)
		kwins[Rds.index(Rd)].append(kwin)
		nwins[Rds.index(Rd)].append(nwin)
		ekave[Rds.index(Rd)].append(sum(ek_)/n)
		enave[Rds.index(Rd)].append(sum(en_)/n)

	# print(Rds, kwins)

	ns = [x for _, x in sorted(zip(Rds, ns))]
	Ris = [x for _, x in sorted(zip(Rds, Ris))]
	kwins = [x for _, x in sorted(zip(Rds, kwins))]
	nwins = [x for _, x in sorted(zip(Rds, nwins))]
	ekave = [x for _, x in sorted(zip(Rds, ekave))]
	enave = [x for _, x in sorted(zip(Rds, enave))]
	Rds = sorted(Rds)

	# print(Rds, kwins)

	######## the percentage that ties with the optimal solution
	plt.figure(figsize=(18, 6))
	for i, (Rd, Ri, n, kwin, nwin, ek, en, c) in enumerate(zip(Rds, Ris, ns, kwins, nwins, ekave, enave, colors)):
			# winning rate	
			if i > -1:
				# print(Rd)
				temp = [(r, 1- kwin_/n_) for r, kwin_, n_ in zip(Ri, kwin, n)]
				temp = sorted(temp, key=lambda x: x[0])
				# print(min([x[1] for x in temp]))
				plt.plot([d[0] for d in temp], [d[1] for d in temp], 
						color=c, linestyle='solid', linewidth=2.5,
						label=r'$R^c=%s$'%str(Rd))
			# values			
	fs = 36
	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.legend(ncol=2, fontsize=fs*0.8)
	plt.xlabel(r'$R^d(m)$', fontsize=fs)
	plt.ylabel(r'$P(e_{PN}=e_{GN})$', fontsize=fs)
	plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
	plt.show()

	######### the amount that lose to the optimal solution
	plt.figure(figsize=(18, 6))
	for i, (Rd, Ri, n, kwin, nwin, ek, en, c) in enumerate(zip(Rds, Ris, ns, kwins, nwins, ekave, enave, colors)):
	# print(sum([sum(n) for n in ns]))				
			if i > -1:
				# print(ek)
				temp = [(r, (ek_ - en_)/ek_) for r, ek_, en_ in zip(Ri, ek, en)]
				temp = sorted(temp, key=lambda x: x[0])
				# print(min([x[1] for x in temp]))
				plt.plot([d[0] for d in temp], [d[1] for d in temp], 
						color=c, linestyle='solid', linewidth=2.5,
						label=r'$R^c=%s$'%str(Rd))

	fs = 36
	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.legend(ncol=2, fontsize=fs*0.8, loc='upper right')
	plt.xlabel(r'$R^d(m)$', fontsize=fs)
	plt.ylabel(r'$(\bar{e}_{PN}-\bar{e}_{GN})$', fontsize=fs)
	plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
	plt.show()
		

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('nd', type=int, help='the number of defenders')
	parser.add_argument('ni', type=int, help='the number of intruders')
	parser.add_argument('lb', type=float, help='lower bound of theta, for initial location generation')
	parser.add_argument('ub', type=float, help='upper bound of theta, for initial location generation')

	args = parser.parse_args()

	# evaluate_assignment(r=.3, nd=args.nd, ni=args.ni, vd=1., vi=.8, tht=[args.lb, args.ub])
	plot_assign_statistics(nd=args.nd, ni=args.ni)