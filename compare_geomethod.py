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
# from sklearn.linear_model import LinearRegression

import cv2
import pickle

from Envs.environment import MultiAgentEnv
import Envs.scenarios as scenarios
from Envs.scenarios.game_mdmi.astrategy import knapsack_assign, negotiate_assign


MAX_EPISODES = 10
MAX_EP_STEPS = 100

PATH = './Logs/Geometric'
scenario = scenarios.load('game_mdmi').Scenario()


def evaluate_assignment(r, nd, ni, vd, vi, log_PATH=PATH, render_every=1e5):

	# create environment and base log path 
	world = scenario.make_world(r=r, nd=nd, ni=ni, vd=vd, vi=vi)
	env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_team,
						scenario.observation, state_callback=scenario.state,
						done_callback=scenario.done_callback_defender)
	for Rd in [2,  3., 4., 5.]:
		nstep = int((Rd-1)/.25)
		for Ri in np.linspace(1, 1+.25*nstep, nstep+1):

			log_path = os.path.join(log_PATH, 'Rd=%d_Ri=%d'%(Rd*100, Ri*100))
			if not os.path.exists(log_path): os.makedirs(log_path)
			existings = [int(i) for i in next(os.walk(log_path))[1]]
			i0 = -1 if not existings else max(existings) # the last existing sample

			for i in range(MAX_EPISODES):

				render = True if i%render_every == 0 else False

				i += i0 + 1
				root_path = os.path.join(log_path, str(i))
				if not os.path.exists(root_path): os.makedirs(root_path)

				with open(root_path+'/config.pickle', 'wb') as f:
					pickle.dump(env, f)
				print('>>>>>>>>>>>> simulating episode %s >>>>>>>>>>>'%i)

				# ---------------- prepare two environments and directories -------------#
				s = env.reset(Rd=Rd, Ri=Ri, evend=True)
				env_ = deepcopy(env)

				path_ = os.path.join(root_path, 'negotiate')
				path = os.path.join(root_path, 'knapsack')
				if not os.path.exists(path_):
					os.makedirs(path_)
				if not os.path.exists(path):
					os.makedirs(path)

				# ------------------- simulation under knapsack_assign -----------------#
				imgArray = []
				print('>>>>>>>>>>>> knapsack_assign')
				et = 0.
				for j in range(MAX_EP_STEPS):

					if render:
						imgdata = env.render()
						if j == 0: height, width, layers = imgdata[0].shape
						imgArray.append(cv2.cvtColor(imgdata[0], cv2.COLOR_BGR2RGB))		

					_, e = knapsack_assign(env.world, firstassign=(j == 0))
					et += e
					# print('!!!!!!!!', e)
					actions = [scenario.dstrategy(d, env.world) for d in env.world.defenders]
					obs_n, reward_n, done_n, info_n = env.step(actions)

					with open(path+'/'+f'No#{i}_traj_k.csv', 'a') as f:
						if j == 0:
							f.write(','.join(['t'] + ['D%s:x,D%s:y'%(dd, dd) for dd in range(env.world.nd)] +\
											 ['I%s:x,I%s:y'%(ii, ii) for ii in range(env.world.ni)] +\
											 ['D%s:vx,D%s:vy'%(dd, dd) for dd in range(env.world.nd)] + \
											 ['I%s:vx,I%s:vy'%(ii, ii) for ii in range(env.world.ni)]+ \
											 ['eff']) + '\n')
						f.write(','.join([str(env.world.t)]+list(map(str, env.get_state())) + ['%.5f'%e])+'\n')

					if all(done_n):
						break
				e_ave = et/j
				# print('env step:', j)

				with open(path+'/'+f'No#{i}_info_k.csv', 'a') as f:
					f.write('i,tc,te,capD\n')
					for I in env.world.intruders:
						d = I.state.n[0] if I.state.n else -1
						f.write(','.join(list(map(str, [I.id, I.state.tc, I.state.te, d])))+'\n')
					
				if render:
					out = cv2.VideoWriter(path+'/'+f'No#{i}_traj_k.mp4',
										  cv2.VideoWriter_fourcc(*'mp4v'), 5, (width,height))
					for k in range(len(imgArray)):
						out.write(imgArray[k])
					out.release()
				env.close()					

				# -------------------- simulation under negotiate_assign --------------------#
				imgArray = []
				print('>>>>>>>>>>>> negotiate_assign')
				et_ = 0
				for j in range(MAX_EP_STEPS):

					if render:
						imgdata = env_.render()
						if j == 0: height, width, layers = imgdata[0].shape
						imgArray.append(cv2.cvtColor(imgdata[0], cv2.COLOR_BGR2RGB))

					_, e = negotiate_assign(env_.world, firstassign=(j == 0))
					et_ += e
					actions = [scenario.dstrategy(d, env_.world) for d in env_.world.defenders]
					obs_n, reward_n, done_n, info_n = env_.step(actions)

					with open(path_+'/'+f'No#{i}_traj_n.csv', 'a') as f:
						if j == 0:
							f.write(','.join(['t'] + ['D%s:x,D%s:y'%(dd, dd) for dd in range(env_.world.nd)] +\
											 ['I%s:x,I%s:y'%(ii, ii) for ii in range(env_.world.ni)] +\
											 ['D%s:vx,D%s:vy'%(dd, dd) for dd in range(env_.world.nd)] + \
											 ['I%s:vx,I%s:vy'%(ii, ii) for ii in range(env_.world.ni)] + \
											 ['eff'])+'\n')
						f.write(','.join([str(env_.world.t)]+list(map(str, env_.get_state())) + ['%.5f'%e])+'\n')

					if all(done_n):
						break
				e_ave_ = et_/j

				with open(path_+'/'+f'No#{i}_info_n.csv', 'a') as f:
					f.write('i,tc,te,capD\n')
					for I in env_.world.intruders:
						d = I.state.n[0] if I.state.n else -1
						f.write(','.join(list(map(str, [I.id, I.state.tc, I.state.te, d])))+'\n')

				if render:
					out = cv2.VideoWriter(path_+'/'+f'No#{i}_traj_n.mp4',
										  cv2.VideoWriter_fourcc(*'mp4v'), 5, (width,height))
					for k in range(len(imgArray)):
						out.write(imgArray[k])
					out.release()
				env_.close() 

				# --------------------------- record statistics -----------------------#
				with open(log_path+'/statistics.csv', 'a') as f:
					if i == 0:
						f.write('tc:k,tc:n,tlevel:k,tlevel:n,e_ave:k,e_ave:n\n')
					f.write(','.join(list(map(str, [env.world.t, 
													env_.world.t, 
													scenario.value(env.world), 
													scenario.value(env_.world),
													e_ave,
													e_ave_])))+'\n')

def plot_traj(Rd, Ri, i, cases, res_path=PATH):

	res_path = os.path.join(res_path, 'Rd=%d_Ri=%d'%(Rd*100, Ri*100), str(i))
	with open(res_path+'/config.pickle', 'rb') as f:
		env = pickle.load(f)

	def plot_oneset(case, env=env, res_path=res_path, i=i):
		linestyle = (0,(5, 5)) if 'negotiate' in case else 'solid'
		tfile = os.path.join(res_path, case) + '/' + f'No#{i}_traj_' + case[0] + '.csv'
		ifile = os.path.join(res_path, case) + '/' + f'No#{i}_info_' + case[0] + '.csv'
		traj = pd.read_csv(tfile)
		info = pd.read_csv(ifile)
		print(info)

		dxs = [name for name in traj.columns if 'D' in name and 'x' in name and 'v' not in name]
		ixs = [name for name in traj.columns if 'I' in name and 'x' in name and 'v' not in name]
		dys = [name for name in traj.columns if 'D' in name and 'y' in name and 'v' not in name]
		iys = [name for name in traj.columns if 'I' in name and 'y' in name and 'v' not in name]

		dvxs = [name for name in traj.columns if 'D' in name and 'vx' in name]
		ivxs = [name for name in traj.columns if 'I' in name and 'vx' in name]
		dvys = [name for name in traj.columns if 'D' in name and 'vy' in name]
		ivys = [name for name in traj.columns if 'I' in name and 'vy' in name]
		
		for d, (dx, dy) in enumerate(zip(dxs, dys)):
			label = r'$D_'+str(d)+'$'   if case=='negotiate' else ''
			plt.plot(traj[dx][0], traj[dy][0], color=env.world.defenders[d].color, marker='o', label=label)
			plt.plot(traj[dx], traj[dy], color=env.world.defenders[d].color, linestyle=linestyle)

		for i, (ix, iy) in enumerate(zip(ixs, iys)):
			label = r'$I$' if case=='negotiate' and i==0 else ''
			plt.plot(traj[ix][0], traj[iy][0], color=env.world.intruders[i].color, marker='>', label=label)
			plt.plot(traj[ix], traj[iy], color=env.world.intruders[i].color, linestyle=linestyle)

			if info['tc'][i] is not None:
				k = int(info['tc'][i]/env.world.dt)-1
				d = info['capD'][i]
				plt.plot(traj[ix][k], traj[iy][k], color=env.world.intruders[i].color, marker='>')
				plt.plot(traj[dxs[d]][k], traj[dys[d]][k], color=env.world.defenders[d].color, marker='o')
				circle = Circle((traj[dxs[d]][k], traj[dys[d]][k]), env.world.defenders[d].r, color=env.world.defenders[d].color, alpha=0.2)
				plt.gca().add_patch(circle)

	if not isinstance(cases, list): cases = [cases]
	for case in cases:
		plot_oneset(case)

	xt, yt, rt = 5., 0., 1.25
	tht = np.linspace(0, 2*pi, 50)
	target = Circle((xt, yt), rt, color='blue', alpha=0.2)
	plt.gca().add_patch(target)
	plt.plot([xt+rt*cos(t) for t in tht], [yt+rt*sin(t) for t in tht], linewidth=2, label='target')

	fs = 14
	plt.axis("equal")
	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.legend(fontsize=fs)
	plt.xlabel(r'$x_D(m)$', fontsize=fs)
	plt.ylabel(r'$y_D(m)$', fontsize=fs)
	plt.show()


def plot_statistics(res_path=PATH):

	Rds, Ris, tcs_n, tcs_k, tls_n, tls_k, rcs_n, rcs_k, es_n, es_k = [], [], [], [], [], [], [], [], [], []
	colors = ['r', 'b', 'g', 'k', 'c', 'm']

	k = -1
	for i in next(os.walk(res_path))[1]:
		# print(i)
		Rd, Ri = i.split('_')
		Rd = float(Rd.split('=')[-1])/100
		Ri = float(Ri.split('=')[-1])/100

		# print(Rd, Ri)
		# print(res_path + '/' + i + '/statistics.csv')
		data = pd.read_csv(res_path + '/' + i + '/statistics.csv')
		# print(data.columns)
		tc_n = data['tc:n'].to_numpy()
		tc_k = data['tc:k'].to_numpy()
		tl_n = data['tlevel:n'].to_numpy()
		tl_k = data['tlevel:k'].to_numpy()
		e_n = data['e_ave:n'].to_numpy()
		e_k = data['e_ave:k'].to_numpy()

		print(i, len(tl_n))

		# print(len(np.where(tl_n<0)[0]))
		# print(len(np.where(tl_k<0)[0]))

		if Rd not in Rds:
			Rds.append(Rd)
			Ris.append([])
			tcs_n.append([])
			tcs_k.append([])
			tls_n.append([])
			tls_k.append([])			
			rcs_n.append([])
			rcs_k.append([])
			es_n.append([])	
			es_k.append([])	
			k += 1

		Ris[k].append(Ri)
		tcs_n[k].append(tc_n.mean())
		tcs_k[k].append(tc_k.mean())
		tls_n[k].append(tl_n.mean())
		tls_k[k].append(tl_k.mean())
		es_n[k].append(e_n.mean())
		es_k[k].append(e_k.mean())
		rcs_n[k].append(len(np.where(tl_n>0)[0])/len(tl_n))
		rcs_k[k].append(len(np.where(tl_k>0)[0])/len(tl_k))

	# for i in range(len(Rds)):
	# 	Rds[i] = [Rds[i]]*len(Ris[i])

	# print(Rds, Ris)

	for i, (Rd, Ri, tc_n, tc_k, tl_n, tl_k, rc_n, rc_k, e_n, e_k, c) in enumerate(zip(Rds, Ris, tcs_n, tcs_k, tls_n, tls_k, rcs_n, rcs_k, es_n, es_k, colors)):
		# print(i)
		if i == 1:
			# plt.plot(Ri, tc_n, color=c, marker='o', markevery=1, linestyle='solid', label='Rd='+str(Rd)+',negotiate')
			# plt.plot(Ri, tc_k, color=c, marker='o', markevery=1, linestyle='dashed', label='Rd='+str(Rd)+',knapsack')
			# plt.plot(Ri, tl_n, color=c, linestyle='solid', label='Rd='+str(Rd)+',negotiate')
			# plt.plot(Ri, tl_k, color=c, linestyle='dashed', label='Rd='+str(Rd)+',knapsack')
			plt.plot(Ri, rc_n, color=c, linestyle='solid', label='Rd='+str(Rd)+',negotiate')
			plt.plot(Ri, rc_k, color=c, linestyle='dashed', label='Rd='+str(Rd)+',knapsack')

	plt.grid()
	plt.legend()
	plt.show()


def plot_correlate(res_path=PATH):

	Rds, Ris, tcs_n, tcs_k, tls_n, tls_k, rcs_n, rcs_k, es_n, es_k = [], [], [], [], [], [], [], [], [], []
	colors = ['r', 'b', 'g', 'k', 'c', 'm']

	k = -1
	for i, f in enumerate(next(os.walk(res_path))[1]):
		# print(i)
		# Rd, Ri = i.split('_')
		# Rd = float(Rd.split('=')[-1])/100
		# Ri = float(Ri.split('=')[-1])/100

		# print(Rd, Ri)
		# print(res_path + '/' + i + '/statistics.csv')
		data = pd.read_csv(res_path + '/' + f + '/statistics.csv')
		# print(data.columns)
		if i == 0:
			tc_n = data['tc:n'].to_numpy()
			tc_k = data['tc:k'].to_numpy()
			tl_n = data['tlevel:n'].to_numpy()
			tl_k = data['tlevel:k'].to_numpy()
			e_n = data['e_ave:n'].to_numpy()
			e_k = data['e_ave:k'].to_numpy()
		else:
			tc_n = np.concatenate((tc_n, data['tc:n'].to_numpy()), axis=0)
			tc_k = np.concatenate((tc_k, data['tc:k'].to_numpy()), axis=0)
			tl_n = np.concatenate((tl_n, data['tlevel:n'].to_numpy()), axis=0)
			tl_k = np.concatenate((tl_k, data['tlevel:k'].to_numpy()), axis=0)
			e_n = np.concatenate((e_n, data['e_ave:n'].to_numpy()), axis=0)
			e_k = np.concatenate((e_k, data['e_ave:k'].to_numpy()), axis=0)
			# rcs_n[k].append(len(np.where(tl_n>0)[0])/len(tl_n))
			# rcs_k[k].append(len(np.where(tl_k>0)[0])/len(tl_k))
	
	diff = pd.DataFrame({ 'e': e_k - e_n,
							'tc': tc_k - tc_n,
							'tl': tl_k - tl_n})
	diff = diff.sort_values(by=['e'])


	# plot linear regression
	# model = LinearRegression()
	# model.fit(diff['e'].to_numpy().reshape(-1, 1), diff['tl'])
	# r_sq = model.score(diff['e'].to_numpy().reshape(-1, 1), diff['tl'])

	# b = model.intercept_
	# a = model.coef_
	# plt.plot(diff['e'], diff['tl'], '.', label='data')
	# plt.plot([-10, 8], [-10*a + b, a*8 + b], linewidth=2, label='regression')

	# fs = 14
	# plt.grid()
	# plt.text(-5, -2.3, 'coefficient of determination = %.2f'%r_sq, fontsize=fs)
	# plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	# plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	# plt.xlabel('Average efficiency (opt - PN)', fontsize=fs)
	# plt.ylabel('Final taget level (opt - PN)', fontsize=fs)
	# plt.legend(fontsize=fs, loc='upper left')
	# plt.show()


	# plot histogram
	plt.hist(diff['tl'], bins=10)
	fs = 14
	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.xlabel('Final target level (opt - PN)', fontsize=fs)
	plt.ylabel('Number of cases', fontsize=fs)
	plt.show()

if __name__ == '__main__':
	# evaluate_assignment(r=.3, nd=3, ni=6, vd=1., vi=.8, render_every=50)
	# plot_traj(4, 1.75, 0, ['negotiate', 'knapsack'])	
	# print('?????????')
	# plot_statistics()	
	plot_correlate()