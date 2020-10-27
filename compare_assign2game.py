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
from Envs.scenarios.game_mdmi.astrategy import knapsack_assign, negotiate_assign


MAX_EPISODES = 50
MAX_EP_STEPS = 100

PATH = './Logs/Geometric/game'
scenario = scenarios.load('game_mdmi').Scenario()

def evaluate_game(r, nd, ni, vd, vi, log_PATH=PATH, render_every=1e5, n_ep=MAX_EPISODES):

	for Rt in [2,  3., 4., 5.]:
		nstep = int((Rt-1)/.25)
		for Ro in np.linspace(1, 1+.25*nstep, nstep+1):

			log_path = os.path.join(log_PATH, 'Rd=%d_Ri=%d'%(Rt*100, Ro*100))
			if not os.path.exists(log_path): os.makedirs(log_path)
			existings = [int(i) for i in next(os.walk(log_path))[1]]
			i0 = -1 if not existings else max(existings) # the last existing sample

			for i in range(n_ep):

				# render = True if i%render_every == 0 else False
				render = False

				i += i0 + 1
				root_path = os.path.join(log_path, str(i))
				if not os.path.exists(root_path): os.makedirs(root_path)

				# with open(root_path+'/config.pickle', 'wb') as f:
				# 	pickle.dump(env, f)
				print('>>>>>>>>>>>> simulating episode %s (Rt=%.2f, Ro=%.2f) >>>>>>>>>>>'%(i, Rt, Ro))

				# ---------------- prepare two environments and directories -------------#
				world = scenario.make_world(r=r, Rt=Rt, Ro=Ro, nd=nd, ni=ni, vd=vd, vi=vi, mode='simple')
				env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_team,
									scenario.observation, state_callback=scenario.state,
									done_callback=scenario.done_callback_defender)
				env.reset(evend=True)				
				env_ = deepcopy(env)
				with open(root_path+'/config.pickle', 'wb') as f:
					pickle.dump(env, f)
				
				path = os.path.join(root_path, 'knapsack')
				path_ = os.path.join(root_path, 'negotiate')
				if not os.path.exists(path):
					os.makedirs(path)
				if not os.path.exists(path_):
					os.makedirs(path_)

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
											 ['I%s:active'%ii for ii in range(env.world.ni)]+ \
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
											 ['I%s:active'%ii for ii in range(env.world.ni)]+ \
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
		# print(info)

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
			plt.plot(traj[dx][0], traj[dy][0], color=env.world.defenders[d].color, 
						marker='o', linewidth=2, markersize=10,
						label=label)
			plt.plot(traj[dx], traj[dy], color=env.world.defenders[d].color, 
					linewidth=2, markersize=10,
					linestyle=linestyle)

		for i, (ix, iy) in enumerate(zip(ixs, iys)):
			label = r'$I$' if case=='negotiate' and i==0 else ''
			plt.plot(traj[ix][0], traj[iy][0], color=env.world.intruders[i].color, 
						marker='>', markersize=10, linewidth=2, label=label)
			plt.plot(traj[ix], traj[iy], color=env.world.intruders[i].color, 
						linewidth=2, markersize=10, linestyle=linestyle)

			# print(info['tc'][i], info['tc'][i] != 'None')
			if info['tc'][i] != 'None':
				# print('??????????????')
				print(info['tc'][i])
				k = int(float(info['tc'][i])/env.world.dt)-1
				d = info['capD'][i]
				plt.plot(traj[ix][k], traj[iy][k], color=env.world.intruders[i].color, 
						 marker='>', markersize=10)
				plt.plot(traj[dxs[d]][k], traj[dys[d]][k], color=env.world.defenders[d].color, 
						 marker='o', markersize=10)
				circle = Circle((traj[dxs[d]][k], traj[dys[d]][k]), env.world.defenders[d].r, color=env.world.defenders[d].color, alpha=0.2)
				plt.gca().add_patch(circle)

	if not isinstance(cases, list): cases = [cases]
	plt.figure(figsize=(10, 10))
	for case in cases:
		plot_oneset(case)

	xt, yt, rt = env.world.target.state.p_pos[0], env.world.target.state.p_pos[1], env.world.target.size
	tht = np.linspace(0, 2*pi, 50)
	target = Circle((xt, yt), rt, color='blue', alpha=0.2)
	plt.gca().add_patch(target)
	plt.plot([xt+rt*cos(t) for t in tht], [yt+rt*sin(t) for t in tht], linewidth=2, label='target')

	fs = 34
	plt.axis("equal")
	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.legend(ncol=1, fontsize=fs*.8)
	plt.xlabel(r'$x_D(m)$', fontsize=fs)
	plt.ylabel(r'$y_D(m)$', fontsize=fs)
	plt.show()

def plot_game_statistics(res_path=PATH):

	Rds, Ris, tcs_n, tcs_k, tls_n, tls_k, rcs_n, rcs_k, es_n, es_k = [], [], [], [], [], [], [], [], [], []
	colors = ['r', 'b', 'g', 'k', 'c', 'm']

	# k = -1
	for i in next(os.walk(res_path))[1]:
		if i != 'assign':
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

			# print(i, len(tl_n))

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
				# k += 1

			Ris[Rds.index(Rd)].append(Ri)
			tcs_n[Rds.index(Rd)].append(tc_n.mean())
			tcs_k[Rds.index(Rd)].append(tc_k.mean())
			tls_n[Rds.index(Rd)].append(tl_n.mean())
			tls_k[Rds.index(Rd)].append(tl_k.mean())
			es_n[Rds.index(Rd)].append(e_n.mean())
			es_k[Rds.index(Rd)].append(e_k.mean())
			rcs_n[Rds.index(Rd)].append(len(np.where(tl_n>0)[0])/len(tl_n))
			rcs_k[Rds.index(Rd)].append(len(np.where(tl_k>0)[0])/len(tl_k))

	# for i in range(len(Rds)):
	# 	Rds[i] = [Rds[i]]*len(Ris[i])

	# print(Rds, Ris)

	plt.figure(figsize=(18, 6))
	for i, (Rd, Ri, tc_n, tc_k, tl_n, tl_k, rc_n, rc_k, e_n, e_k, c) in enumerate(zip(Rds, Ris, tcs_n, tcs_k, tls_n, tls_k, rcs_n, rcs_k, es_n, es_k, colors)):
		# print(Rd, Ri)
		if i > -1:
			# plt.plot(Ri, tc_n, color=c, marker='o', markevery=1, linestyle='solid', label='Rd='+str(Rd)+',negotiate')
			# plt.plot(Ri, tc_k, color=c, marker='o', markevery=1, linestyle='dashed', label='Rd='+str(Rd)+',knapsack')
			# plt.plot(Ri, tl_n, color=c, linestyle='solid', label='Rd='+str(Rd)+',negotiate')
			# plt.plot(Ri, tl_k, color=c, linestyle='dashed', label='Rd='+str(Rd)+',knapsack')
			temp = [(r, rc_n_) for r, rc_n_ in zip(Ri, tc_n)]
			temp = sorted(temp, key=lambda x: x[0])
			plt.plot([d[0] for d in temp], [d[1] for d in temp], 
						color=c, linestyle='solid', linewidth=2, 
						label=r'$R_c=$'+str(Rd)+',PN')
			temp = [(r, rc_k_) for r, rc_k_ in zip(Ri, tc_k)]
			temp = sorted(temp, key=lambda x: x[0])
			plt.plot([d[0] for d in temp], [d[1] for d in temp], 
						color=c, linestyle='dashed', linewidth=2, 
						label=r'$R_c=$'+str(Rd)+',opt')
			# plt.plot(Ri, rc_k, color=c, linestyle='dashed', label='Rd='+str(Rd)+',knapsack')

	fs = 36
	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.legend(ncol=2, fontsize=fs*0.8)
	plt.xlabel(r'$R_d(m)$', fontsize=fs)
	# plt.ylabel(r'$(\bar{e}_{opt}-\bar{e}_{PN})$', fontsize=fs)
	# plt.ylabel(r'$\bar{g}$', fontsize=fs)
	plt.ylabel(r'$T_c$', fontsize=fs)
	plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
	plt.show()


def plot_correlate(res_path=PATH):

	Rds, Ris, tcs_n, tcs_k, tls_n, tls_k, rcs_n, rcs_k, es_n, es_k = [], [], [], [], [], [], [], [], [], []
	colors = ['r', 'b', 'g', 'k', 'c', 'm']

	k = -1
	for i, f in enumerate(next(os.walk(res_path))[1]):
		# if f != 'assign':
		print(f)
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
	model = LinearRegression()
	model.fit(diff['e'].to_numpy().reshape(-1, 1), diff['tc'])
	r_sq = model.score(diff['e'].to_numpy().reshape(-1, 1), diff['tc'])

	b = model.intercept_
	a = model.coef_
	plt.figure(figsize=(12, 12))
	plt.plot(diff['e'], diff['tc'], '.', markersize=10, markevery=10, label='data')
	plt.plot([-10, 8], [-10*a + b, a*8 + b], linewidth=2, label='regression')

	fs = 34
	plt.grid()
	plt.text(-5, -2.3, 'coefficient of determination = %.2f'%r_sq, fontsize=fs)
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.xlabel(r'$\bar{e}_{opt}-\bar{e}_{PN}$', fontsize=fs)
	plt.ylabel(r'$\bar{Tc}_{opt}-\bar{Tc}_{PN}$', fontsize=fs)
	plt.legend(fontsize=fs*.8, loc='upper left')
	plt.show()


	# plot histogram
	# plt.hist(diff['tc'], bins=10)
	# fs = 14
	# plt.grid()
	# plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	# plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	# plt.xlabel('Final target level (opt - PN)', fontsize=fs)
	# plt.ylabel('Number of cases', fontsize=fs)
	# plt.show()

if __name__ == '__main__':
	evaluate_game(r=.3, nd=3, ni=12, vd=1., vi=.8, render_every=1e10)
	# plot_traj(5, 4, 218, ['negotiate', 'knapsack'])	
	# plot_game_statistics()
	# plot_correlate()