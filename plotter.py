import numpy as np
from math import sqrt, pi, sin, cos
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle

from Envs.scenarios.game_mdmi import dcolors

def plot_assign(assign, cap):
	
	lw = 2.
	ms = 10.
	fs = 14
	skip = 2
	circscale = 2.
	circoffset = 4.

	for d, (p, ass) in enumerate(assign.items()):
		for k, (t, pref_dict, itarg) in enumerate(zip(ass['t'], ass['pref'], ass['i'])):
			if k%skip == 0:
				for i, e in pref_dict.items():# preferred intruder
					plt.plot(t, circoffset+circscale*int(i[1:]), 'C0o', 
							 markersize=ms, alpha=0.2, color=dcolors[d])
				
				plt.plot(t, circoffset+circscale*itarg, 'C0o', 
						 markersize=ms, color=dcolors[d]) # targeted intruder
			
			# for i, data in cap.items(): # captured intruder
			# 	if t > data['tcap']:
			# 		plt.plot(t, circoffset+circscale*int(i[1:]), 'C0x', 
			# 				 markersize=ms, color=dcolors[int(data['dcap'][1:])])

		plt.plot(ass['t'], ass['e'], color=dcolors[d], linewidth=lw)

	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.ylabel('y(m)', fontsize=fs)
	plt.xlabel('x(m)', fontsize=fs)
	plt.show()

def compare_traj(tsg, tss, states_gazebo, states_simple, cap, param):

	# print('in plot traj:', ts)
	lw = 2.
	fs = 14.

	dcap = {'D'+str(d):{'t':[], 'i':[]} for d in range(param['nd'])}
	for i, info in cap.items():
		dcap[info['dcap']]['t'].append(info['tcap'])
		dcap[info['dcap']]['i'].append(i)

	twpts = [tsg[0]] + [data['tcap'] for i, data in cap.items()]

	# plot trajectories in gazebo
	for p, s in states_gazebo.items():
		c = 'r' if 'I' in p else dcolors[int(p[1:])]
		plt.plot([s['x'](t) for t in tsg], [s['y'](t) for t in tsg], color=c, label=p, linewidth=lw)
		for tc in twpts:
			plt.plot(s['x'](tc), s['y'](tc), color=c, marker='o', markersize=6., linewidth=lw)
			if 'D' in p:
				for t in dcap[p]['t']:
					picon = plt.Circle([s['x'](t), s['y'](t)], radius=param['r'], color=c, alpha=.1)
					plt.gca().add_patch(picon)

	# plot trajectories of the simple simulation
	for p, s in states_simple.items():
		c = 'r' if 'I' in p else dcolors[int(p[1:])]
		plt.plot([s['x'](t) for t in tss], [s['y'](t) for t in tss], color=c, label=p, linewidth=lw, linestyle='dashed')
		# for tc in twpts:
		# 	plt.plot(s['x'](tc), s['y'](tc), color=c, marker='o', markersize=6., linewidth=lw)
		# 	if 'D' in p:
		# 		for t in dcap[p]['t']:
		# 			picon = plt.Circle([s['x'](t), s['y'](t)], radius=param['r'], color=c, alpha=.1)
		# 			plt.gca().add_patch(picon)

	ks = np.linspace(.5*pi, 1.5*pi, 25)
	plt.plot(param['x0targ']+param['Rtarg']*np.cos(ks),
			 param['y0targ']+param['Rtarg']*np.sin(ks),
			 color='k', linewidth=lw)

	plt.axis('equal')
	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.xlabel('x(m)', fontsize=fs)
	plt.ylabel('y(m)', fontsize=fs)
	plt.show()


def compare_traj_and_v(ts, states_gazebo, cmd_simple, cap, param):

	lw = 2.
	fs = 14.
	arrowscale = 0.4
	aw = .03

	dcap = {'D'+str(d):{'t':[], 'i':[]} for d in range(param['nd'])}
	for i, info in cap.items():
		if info['dcap'] is not None:
			dcap[info['dcap']]['t'].append(info['tcap'])
			dcap[info['dcap']]['i'].append(i)
		# elif info[]

	twpts = [ts[0]] + [data['tcap'] for i, data in cap.items() if data['dcap'] is not None]

	# plot trajectories
	for p, s in states_gazebo.items():
		c = 'r' if 'I' in p else dcolors[int(p[1:])]
		plt.plot([s['x'](t) for t in ts], [s['y'](t) for t in ts], color=c, label=p, linewidth=lw)
		for tc in twpts:
			plt.plot(s['x'](tc), s['y'](tc), color=c, marker='o', markersize=6., linewidth=lw)
			if 'D' in p:
				for t in dcap[p]['t']:
					picon = plt.Circle([s['x'](t), s['y'](t)], radius=param['r'], color=c, alpha=.1)
					plt.gca().add_patch(picon)

	for i, t in enumerate(ts):
		for d, cmd in cmd_simple.items():
			if i%8 == 0:
				plt.arrow(states_gazebo[d]['x'](t), states_gazebo[d]['y'](t), # locations on the gazebo traj
						  arrowscale*cmd['vx'][i], arrowscale*cmd['vy'][i], 
						  color=dcolors[int(d[1:])], linewidth=lw,
						  head_width=aw) # command of simple sim

	ks = np.linspace(.5*pi, 1.5*pi, 25)
	plt.plot(param['x0targ']+param['Rtarg']*np.cos(ks),
			 param['y0targ']+param['Rtarg']*np.sin(ks),
			 color='k', linewidth=lw)

	plt.axis('equal')
	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.xlabel('x(m)', fontsize=fs)
	plt.ylabel('y(m)', fontsize=fs)
	plt.show()

def compare_cmdv(ts, cmd_gazebo, cmd_simple):

	lw = 2.
	fs = 14.

	for d in cmd_gazebo:
		if 'D' in d:
			# plt.plot(ts, [sqrt(vx**2 + vy**2) for (vx, vy) in zip(cmd_gazebo[d]['vx'](ts), cmd_gazebo[d]['vy'](ts))], color=dcolors[int(d[1:])])
			# plt.plot(ts, [sqrt(vx**2 + vy**2) for (vx, vy) in zip(cmd_simple[d]['vx'], cmd_simple[d]['vy'])], color=dcolors[int(d[1:])])
			plt.plot(ts, cmd_gazebo[d]['vx'](ts), color=dcolors[int(d[1:])], linewidth=lw, label=d)
			plt.plot(ts, cmd_simple[d]['vx'], color=dcolors[int(d[1:])], linestyle='dashed', linewidth=lw)
			# plt.plot(ts, cmd_gazebo[d]['vy'](ts), color=dcolors[int(d[1:])], linewidth=lw)
			# plt.plot(ts, cmd_simple[d]['vy'], color=dcolors[int(d[1:])], linestyle='dashed', linewidth=lw)

	plt.grid()
	plt.xlabel('t(s)', fontsize=fs)
	plt.ylabel('cmdv(m/s)', fontsize=fs)
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.legend(fontsize=fs)
	plt.show()	


def velocity_response(ts, cmd_gazebo, states_gazebo):

	lw = 2.
	fs = 14.

	for d in cmd_gazebo:
		if 'D' in d:
			# plt.plot(ts, [sqrt(cmd_gazebo[d]['vx'](t)**2+cmd_gazebo[d]['vy'](t)**2) for t in ts], color=dcolors[int(d[1:])], 
			# 		 linewidth=lw, label=d)
			plt.plot(ts, [sqrt(states_gazebo[d]['vx'](t)**2+states_gazebo[d]['vy'](t)**2) for t in ts], color=dcolors[int(d[1:])], 
					 linestyle='dashed', linewidth=lw)	
			# plt.plot(ts, cmd_gazebo[d]['vy'](ts), color=dcolors[int(d[1:])], 
			# 		 linewidth=lw, label=d)
			# plt.plot(ts, states_gazebo[d]['vy'](ts), color=dcolors[int(d[1:])], 
			# 		 linestyle='dashed', linewidth=lw)	

	plt.grid()
	plt.xlabel('t(s)', fontsize=fs)
	plt.ylabel('velocity(m/s)', fontsize=fs)
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.legend(fontsize=fs)
	plt.show()
