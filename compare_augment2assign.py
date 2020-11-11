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
from Envs.scenarios.game_mdmi.astrategy import negotiate_assign, augmented_negotiation


MAX_EPISODES = 5
MAX_EP_STEPS = 10

PATH = './Logs/Geometric'
scenario = scenarios.load('game_mdmi').Scenario()


def evaluate_assignment(r, nd, ni, vd, vi, log_PATH=PATH, n_ep=MAX_EPISODES):
	for Rt in [1., 3., 5.]:
		nstep = int((Rt-1)/.25)
		for Ro in np.linspace(1., 1+.25*nstep, nstep+1):
			log_path = os.path.join(log_PATH, 'D%dI%d'%(nd, ni), 'augment2assign', 'Rd=%d_Ri=%d'%(Rt*100, Ro*100))
			if not os.path.exists(log_path): 
				os.makedirs(log_path)
			log_file = log_path + '/eff.csv'
			
			for i in range(n_ep):

				print(Rt, Ro, i, log_file)

				world = scenario.make_world(r=r, Rt=Rt, Ro=Ro, nd=nd, ni=ni, vd=vd, vi=vi, mode='simple')
				env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_team,
									scenario.observation, state_callback=scenario.state,
									done_callback=scenario.done_callback_defender)
				# for p in env.world.agents:
				# 	print(p.size)				
				env.reset(evend=True)	
				env_ = deepcopy(env)

				_, ea = augmented_negotiation(env.world)
				_, en = negotiate_assign(env_.world)

				state = list(map(str, env.get_state()))

				if not os.path.exists(log_file):
					with open(log_file, 'w') as f:
						f.write(','.join(['D%s:x,D%s:y'%(dd, dd) for dd in range(env.world.nd)] +\
										 ['I%s:x,I%s:y'%(ii, ii) for ii in range(env.world.ni)] +\
										 ['I%s:active'%ii for ii in range(env.world.ni)]+ \
										 ['ea'] + ['en']) + '\n')		
										 				
				with open(log_file, 'a') as f:
					f.write(','.join(state + ['%.5f'%ea] + ['%.5f'%en])+'\n')


def plot_assign_statistics(res_path=PATH, nd=3, ni=12):
	
	res_path = os.path.join(res_path, 'D%dI%d'%(nd, ni), 'augment2assign')
	# print(res_path)
	colors = ['r', 'b', 'g', 'k', 'c', 'm']
	
	Rds, Ris, ns, kwins, nwins, eaave, enave = [], [], [], [], [], [], []
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
			eaave.append([])
			enave.append([])
			# print(Rd)
			# k += 1

		print(res_path + '/' + i + '/eff.csv')
		data = pd.read_csv(res_path + '/' + i + '/eff.csv')

		Ris[Rds.index(Rd)].append(Ri)
		# print(Rd, Ris[k])

		ea_ = data['ea'].to_list()
		en_ = data['en'].to_list()
		ea, en = [], []
		n, kwin, nwin = 0, 0, 0
		for ea__, en__ in zip(ea_, en_):
			if ea__*en__ != 0:
				n += 1
				ea.append(ea__)
				en.append(en__)
				if ea__ > en__:
					kwin += 1
				elif ea__ < en__:
					nwin += 1
		# print(n, kwin, nwin)

		ns[Rds.index(Rd)].append(n)
		kwins[Rds.index(Rd)].append(kwin)
		nwins[Rds.index(Rd)].append(nwin)
		eaave[Rds.index(Rd)].append(sum(ea_)/n)
		enave[Rds.index(Rd)].append(sum(en_)/n)


	######## the percentage that ties with the optimal solution
	plt.figure(figsize=(18, 6))
	for i, (Rd, Ri, n, kwin, nwin, ea, en, c) in enumerate(zip(Rds, Ris, ns, kwins, nwins, eaave, enave, colors)):
			# winning rate	
			if i > -1:
				temp = [(r, 1- kwin_/n_) for r, kwin_, n_ in zip(Ri, kwin, n)]
				temp = sorted(temp, key=lambda x: x[0])
				print(min([x[1] for x in temp]))
				plt.plot([d[0] for d in temp], [d[1] for d in temp], 
						color=c, linestyle='solid', linewidth=2.5,
						label=r'$R_c=%s$'%str(Rd))
			# values			
	fs = 36
	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.legend(ncol=2, fontsize=fs*0.8)
	plt.xlabel(r'$R_d(m)$', fontsize=fs)
	plt.ylabel(r'$P(e_{PN}=e_{opt})$', fontsize=fs)
	plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
	plt.show()

	######### the amount that lose to the optimal solution
	# plt.figure(figsize=(18, 6))
	# for i, (Rd, Ri, n, kwin, nwin, ea, en, c) in enumerate(zip(Rds, Ris, ns, kwins, nwins, eaave, enave, colors)):
	# # print(sum([sum(n) for n in ns]))				
	# 		if i > -1:
	# 			temp = [(r, (ea_ - en_)/ea_) for r, ea_, en_ in zip(Ri, ea, en)]
	# 			temp = sorted(temp, key=lambda x: x[0])
	# 			# print(min([x[1] for x in temp]))
	# 			plt.plot([d[0] for d in temp], [d[1] for d in temp], 
	# 					color=c, linestyle='solid', linewidth=2.5,
	# 					label=r'$R_c=%s$'%str(Rd))

	# fs = 36
	# plt.grid()
	# plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	# plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	# plt.legend(ncol=2, fontsize=fs*0.8)
	# plt.xlabel(r'$R_d(m)$', fontsize=fs)
	# plt.ylabel(r'$(\bar{e}_{opt}-\bar{e}_{PN})$', fontsize=fs)
	# plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
	# plt.show()
		

if __name__ == '__main__':
	evaluate_assignment(r=.1, nd=3, ni=14, vd=1., vi=.8)
	# plot_assign_statistics(nd=3, ni=12)