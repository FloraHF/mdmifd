import numpy as np
from math import sqrt, pi, sin, cos
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerTuple
from matplotlib import rc
rc('text', usetex=True)

from Envs.scenarios.game_mdmi import dcolors
from reader import res_path

lw = 3.
ms = 12.
ms_traj = 12.
fs = 34.
traj_size = (10,10)
plot_size = (13,4)

#####################################################################
# |							helper funcs							|
#####################################################################
def process_cap(ts, cap, param, tmin=0):
	# tmin: to shift cap by tmin. Because gazebo simulation starts from t>0, 
								# But     simple simulation starts from t=0.

	# when an intruder is captured, highlight locations of all the players
	dcap = {'D'+str(d):{'t':[], 'i':[]} for d in range(param['nd'])}
	for i, info in cap.items():
		if info['dcap'] is not None:
			dcap[info['dcap']]['t'].append(info['tcap']+tmin)
			dcap[info['dcap']]['i'].append(i)

	# extract time from dcap
	twpts = [ts[0]] + [data['tcap']+tmin for i, data in cap.items() if data['dcap'] is not None] + [ts[-1]]

	# print('dcap\n', dcap)
	# print('twpts\n', twpts)
	return dcap, twpts

def process_assign(ts, assign, param):
	# rearrange assignment in the intruders perspective
	# print('assign\n', assign)
	iass = {'I'+str(i):{'t':[], 'd':[]} for i in range(param['ni'])}
	for t in ts:
		for d, ass in assign.items():
			# print(ass['approx'](t))
			try: # t could beyond the upper bound of ass['approx'](interp1), 
				 # because no assignment exists when some intruders are captured
				i = 'I%d'%np.round(ass['approx'](t))
				iass[i]['t'].append(t)
				iass[i]['d'].append(d)
			except:
				pass

	# print('assign\n', iass)
	# rearrange iass by time segment for different assign
	iass_seg = {'I'+str(i):{'t':[], 'd':[]} for i in range(param['ni'])}
	for i, ass in iass.items():
		if ass['t']:
			iass_seg[i]['t'].append([ass['t'][0]])
			iass_seg[i]['d'].append(ass['d'][0])
			for t, d in zip(ass['t'][1:], ass['d'][1:]):
				iass_seg[i]['t'][-1].append(t)
				if d != iass_seg[i]['d'][-1]:
					iass_seg[i]['t'].append([t])
					iass_seg[i]['d'].append(d)
					
	# print('iass_seg\n', iass_seg)
	return iass_seg

def plot_traj(ts, states, iass_seg, dcap, twpts, param, linestyle='solid'):
	# print('ploting requrest', ts[-1])
	lgd = []
	for p, s in states.items():
		# print(s)
		if 'D' in p: # defenders' trajectories
			dicon, = plt.plot([s['x'](t) for t in [s['tmin']]+list(ts)], 
					 [s['y'](t) for t in [s['tmin']]+list(ts)], 
					 color=dcolors[int(p[1:])], label=p, lw=lw, ls=linestyle,
					 marker='o', markersize=ms_traj, markevery=1000)
			lgd.append(dicon)
		else:
			# print(p)
			if iass_seg[p]['t']: # if intruder p has been assigned to some defenders
				# first segment
				tseg = [s['tmin']] + \
					   [t for t in ts if t < iass_seg[p]['t'][0][0]+0.1] # in case not assigned at the beginning
				iicon, = plt.plot([s['x'](t) for t in tseg], [s['y'](t) for t in tseg], 
						 color=np.array([0.1, 1., 1.]), linewidth=lw, linestyle=linestyle,
						 label=p, marker='>', markersize=ms_traj, markevery=1000)
				# the rest of the segments
				for tseg, d in zip(iass_seg[p]['t'], iass_seg[p]['d']):
					plt.plot([s['x'](t) for t in tseg], [s['y'](t) for t in tseg], 
							 color=dcolors[int(d[1:])], linewidth=lw, linestyle=linestyle,
							 label=p)
			else: # if the intruder is not assigned at all
				iicon, = plt.plot([s['x'](t) for t in ts], [s['y'](t) for t in ts], 
						 color=np.array([0.1, 1., 1.]), linewidth=lw, linestyle=linestyle,
						 label=p, marker='>', markersize=ms_traj, markevery=1000)

		if twpts is not None:		
			for tc in twpts:
				if 'D' in p:
					plt.plot(s['x'](tc), s['y'](tc), color=dcolors[int(p[1:])], marker='o', markersize=ms_traj, linewidth=lw)
					for t in dcap[p]['t']:
						tempx, tempy, capr = s['x'](t), s['y'](t), param['r']
						picon = plt.Circle([tempx, tempy], radius=capr, color=dcolors[int(p[1:])], 
											alpha=.02, linestyle=linestyle, lw=lw)
						plt.gca().add_patch(picon)
						temptht = np.linspace(0, 2*pi, 50)
						plt.plot([tempx+capr*cos(tht) for tht in temptht],
								 [tempy+capr*sin(tht) for tht in temptht],
								 ls=linestyle, lw=lw, color=dcolors[int(p[1:])],
								 alpha = 0.1)
				else:
					if iass_seg[p]['t']:
						# first segment
						tseg = [s['tmin']] + \
							   [t for t in ts if t < iass_seg[p]['t'][0][0]+0.1] # in case not assigned at the beginning
						if tseg[0] <= tc <= tseg[-1]:
							ttemp = [abs(tc-tt) for tt in tseg]
							tc = tseg[ttemp.index(min(ttemp))] # this is to make sure that the markers are on the trajectory
							plt.plot(s['x'](tc), s['y'](tc), color=np.array([0.1, 1., 1.]), 
									marker='>', markersize=ms_traj, linewidth=lw)
						# other segments				
						for tseg, d in zip(iass_seg[p]['t'], iass_seg[p]['d']):
							if tseg[0] <= tc <= tseg[-1]:
								ttemp = [abs(tc-tt) for tt in tseg]
								tc = tseg[ttemp.index(min(ttemp))] # this is to make sure that the markers are on the trajectory
								plt.plot(s['x'](tc), s['y'](tc), color=dcolors[int(d[1:])], marker='>', markersize=ms_traj, linewidth=lw)
					else:
						plt.plot(s['x'](tc), s['y'](tc), color=np.array([0.1, 1., 1.]), marker='>', markersize=ms_traj, linewidth=lw)

	return tuple(lgd), (iicon)

def plot_target(param):
	xt, yt, rt = param['x0targ'], param['y0targ'], param['Rtarg']
	tht = np.linspace(0, 2*pi, 50)
	target = plt.Circle((xt, yt), rt, color='blue', alpha=0.2)
	plt.gca().add_patch(target)
	plt.plot([xt+rt*cos(t) for t in tht], [yt+rt*sin(t) for t in tht], linewidth=lw, label='target')

#####################################################################
# |							plot figures							|
#####################################################################
def plot_assign(assign, cap):
	
	skip = 4
	circscale = 4.
	circoffset = 6.

	plt.figure(figsize=plot_size)
	lgp, lga = [], []
	for d, (p, ass) in enumerate(assign.items()):
		icon_pref, = plt.plot(-1, -1, 'C0o', 
							 markersize=ms, alpha=0.2, color=dcolors[d])
		icon_ass, = plt.plot(-1, -1, 'C0o', 
							 markersize=ms, color=dcolors[d])
		# icon_e, = plt.plot([0, 1], [-10, -12], color=dcolors[d], linewidth=lw)
		lgp.append(icon_pref)
		lga.append(icon_ass)
		# lge.append(icon_e)
		for k, (t, pref_dict, itarg) in enumerate(zip(ass['t'], ass['pref'], ass['i'])):
			if k%skip == 0:
				for i, e in pref_dict.items():# preferred intruder
					plt.plot(t, circoffset+circscale*int(i[1:]), 'C0o', 
							 markersize=ms, alpha=0.2, color=dcolors[d])
				
				plt.plot(t, circoffset+circscale*itarg, 'C0o', 
						 markersize=ms, color=dcolors[d]) # targeted intruder

		# plt.plot(ass['t'], ass['e'], color=dcolors[d], linewidth=lw)

	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.ylim(0, 36)
	plt.xlim(19, 31)
	plt.subplots_adjust(left=.09, bottom=.3, right=.9, top=.9)
	plt.yticks([circoffset+circscale*i for i in range(7) if i%2==1], 
				[r'$'+str(i)+'$' for i in range(7) if i%2==1])
	plt.ylabel(r'$I_i$', fontsize=fs)
	plt.xlabel(r'$t(s)$', fontsize=fs)

	ax2 = plt.gca().twinx()
	lge = []
	for d, (p, ass) in enumerate(assign.items()): 
		e, = ax2.plot(ass['t'], ass['e'], color=dcolors[d], linewidth=lw)
		lge.append(e)

	ax2.tick_params(axis='both', which='major', labelsize=fs)
	ax2.tick_params(axis='both', which='minor', labelsize=fs)
	ax2.set_ylabel(r'$e$', fontsize=fs)
	ax2.set_ylim(-.1, 20.)	
	plt.legend([tuple(lgp), tuple(lga), tuple(lge)], [r'preferred', r'assigned', r'$e$'],
				fontsize=fs*0.75, loc='lower right',
				handler_map={tuple: HandlerTuple(ndivide=None)})			
	plt.show()

# def compare_traj(tsg, tss, states_gazebo, states_simple, cap, param):
def compare_traj(ts_gazebo, states_gazebo, assign_gazebo, cap_gazebo,
				 ts_simple, states_simple, assign_simple, cap_simple,
				 param):

	iass_seg_gazebo = process_assign(ts_gazebo, assign_gazebo, param)
	dcap_gazebo, twpts_gazebo = process_cap(ts_gazebo, cap_gazebo, param)

	iass_seg_simple = process_assign(ts_gazebo, assign_simple, param)
	dcap_simple, twpts_simple = process_cap(ts_simple, cap_simple, param, tmin=ts_simple[0])
	
	plt.figure(figsize=traj_size)
	lgd, lgi = plot_traj(ts_gazebo, states_gazebo, iass_seg_gazebo, dcap_gazebo, twpts_gazebo, param, linestyle='solid')
	# _,   _   = plot_traj(ts_simple, states_simple, iass_seg_simple, dcap_simple, twpts_simple, param, linestyle='dashed')
	plot_target(param)

	plt.axis('equal')
	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.xlabel(r'$x(m)$', fontsize=fs)
	plt.ylabel(r'$y(m)$', fontsize=fs)
	plt.xlim(-2, 2)
	plt.ylim(-1, 2)
	plt.legend([lgd, lgi], [r'$D$', r'$I$'],
			fontsize=fs*0.75, loc='upper right',
			handler_map={tuple: HandlerTuple(ndivide=None)})		
	plt.show()
	# plt.savefig(res_path+'trajectory_samestart.jpg')
	# plt.close()


def compare_traj_and_v(ts, states, assign, cap, param, name='gazebo'):

	iass_seg = process_assign(ts, assign, param)
	dcap, twpts = process_cap(ts, cap, param)
	
	plt.figure(figsize=traj_size)
	plot_traj(ts, states, iass_seg, dcap, twpts, param)
	plot_target(param)

	plt.axis('equal')
	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.xlabel(r'$x(m)$', fontsize=fs)
	plt.ylabel(r'$y(m)$', fontsize=fs)
	# plt.legend(fontsize=fs)
	plt.show()
	# plt.savefig(res_path+'trajectory_'+name+'.jpg')
	# plt.close()

def compare_cmdv(ts, cmd_gazebo, cmd_simple):

	plt.figure(figsize=plot_size)

	for d in cmd_gazebo:
		if 'D' in d:
			plt.plot(ts, [sqrt(vx**2 + vy**2) for (vx, vy) in zip(cmd_gazebo[d]['vx'](ts), cmd_gazebo[d]['vy'](ts))], 
						color=dcolors[int(d[1:])], linewidth=lw)
			plt.plot(ts, [sqrt(vx**2 + vy**2) for (vx, vy) in zip(cmd_simple[d]['vx'], cmd_simple[d]['vy'])], 
						color=dcolors[int(d[1:])], linewidth=lw, linestyle='dashed')

	plt.grid()
	plt.xlabel('t(s)', fontsize=fs)
	plt.ylabel('cmdv(m/s)', fontsize=fs)
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.legend(fontsize=fs)
	# plt.show()	
	plt.savefig(res_path+'cmdv_compare_speed.jpg')
	plt.close()


	########## vx
	plt.figure(figsize=plot_size)

	for d in cmd_gazebo:
		if 'D' in d:
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
	# plt.show()	
	plt.savefig(res_path+'cmdv_compare_vx.jpg')
	plt.close()


def velocity_response(ts, cmd_gazebo, states_gazebo):

	plt.figure(figsize=plot_size)
	# print('!!!!!!!!')

	for d in cmd_gazebo:
		if 'D' in d:
			plt.plot(ts, [sqrt(cmd_gazebo[d]['vx'](t)**2+cmd_gazebo[d]['vy'](t)**2) for t in ts], color=dcolors[int(d[1:])], 
					 linewidth=lw, label= r'$D_%d'%int(d[1:])+'$')
			plt.plot(ts, [sqrt(states_gazebo[d]['vx'](t)**2+states_gazebo[d]['vy'](t)**2) for t in ts], color=dcolors[int(d[1:])], 
					 linestyle='dashed', linewidth=lw)	

	plt.grid()
	plt.xlabel(r't(s)', fontsize=fs)
	plt.ylabel(r'velocity(m/s)', fontsize=fs)
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.legend(fontsize=fs)
	plt.savefig(res_path+'velocity_response_speed.jpg')
	plt.close()

	########### vx
	plt.figure(figsize=plot_size)
	lgc, lgr = [], []
	for d in cmd_gazebo:
		if 'D' in d:
			iconc, = plt.plot(ts, cmd_gazebo[d]['vx'](ts), color=dcolors[int(d[1:])], 
					 linewidth=lw, label=r'$D_%d'%int(d[1:])+'$')
			iconr, = plt.plot(ts, states_gazebo[d]['vx'](ts), color=dcolors[int(d[1:])], 
					 linestyle='dashed', linewidth=lw)	
			lgc.append(iconc)
			lgr.append(iconr)


	plt.grid()
	plt.ylim(-.26, .3)
	plt.xlabel(r'$t(s)$', fontsize=fs)
	plt.ylabel(r'$v_x(m/s)$', fontsize=fs)
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.subplots_adjust(left=.16, bottom=.3, right=.96, top=.9)
	plt.legend([tuple(lgc), tuple(lgr)], [r'command', r'actual'],
			fontsize=fs*0.75, loc='upper left',
			handler_map={tuple: HandlerTuple(ndivide=None)})	
	# plt.legend(fontsize=fs*0.8)
	# plt.savefig(res_path+'velocity_response_vx.jpg')
	# plt.close()
	plt.show()
