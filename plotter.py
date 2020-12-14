import numpy as np
from math import sqrt, pi, sin, cos
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerTuple
from matplotlib import rc
rc('text', usetex=True)

from Envs.scenarios.game_mdmi import dcolors
# from reader import res_path

class Plotter(object):
	"""docstring for Plotter"""
	def __init__(self, res_path, param):

		self.res_path = res_path
		self.ni = param['ni']
		self.nd = param['nd']
		self.r  = param['r']
		self.xt = param['x0targ']
		self.yt = param['y0targ']
		self.rt = param['Rtarg']

		self.lw = 3.
		self.ms = 12.
		self.ms_traj = 12.
		self.fs = 34.
		self.traj_size = (10,10)
		self.plot_size = (13,4)

	#####################################################################
	# |							helper funcs							|
	#####################################################################
	def process_cap(self, ts, cap, tmin=0):
		# tmin: to shift cap by tmin. Because gazebo simulation starts from t>0, 
									# But     simple simulation starts from t=0.

		# when an intruder is captured, highlight locations of all the players
		dcap = {'D'+str(d):{'t':[], 'i':[]} for d in range(self.nd)}
		for i, info in cap.items():
			if info['dcap'] is not None:
				dcap[info['dcap']]['t'].append(info['tcap']+tmin)
				dcap[info['dcap']]['i'].append(i)

		# extract time from dcap
		twpts = [ts[0]] + [data['tcap']+tmin for i, data in cap.items() if data['dcap'] is not None] + [ts[-1]]

		# print('dcap\n', dcap)
		# print('twpts\n', twpts)
		return dcap, twpts

	def process_assign(self, ts, assign):
		# rearrange assignment in the intruders perspective
		# print('assign\n', assign)
		iass = {'I'+str(i):{'t':[], 'd':[]} for i in range(self.ni)}
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
		iass_seg = {'I'+str(i):{'t':[], 'd':[]} for i in range(self.ni)}
		for i, ass in iass.items():
			if ass['t']:
				iass_seg[i]['t'].append([ass['t'][0]])
				iass_seg[i]['d'].append(ass['d'][0])
				for t, d in zip(ass['t'][1:], ass['d'][1:]):
					iass_seg[i]['t'][-1].append(t)
					if d != iass_seg[i]['d'][-1]:
						iass_seg[i]['t'].append([t])
						iass_seg[i]['d'].append(d)
						
		# print(iass_seg['I1'])
		return iass_seg

	def plot_traj(self, ts, states, iass_seg, dcap, twpts, linestyle='solid'):
		# print('ploting requrest', ts[-1])
		lgd = []
		for p, s in states.items():
			# print(s)
			if 'D' in p: # defenders' trajectories
				dicon, = plt.plot([s['x'](t) for t in [s['tmin']]+list(ts)], 
						 [s['y'](t) for t in [s['tmin']]+list(ts)], 
						 color=dcolors[int(p[1:])], label=p, lw=self.lw, ls=linestyle,
						 marker='o', ms=self.ms_traj, markevery=1000)
				lgd.append(dicon)
			else:
				# print('---------', p, '-------------')
				if iass_seg[p]['t']: # if intruder p has been assigned to some defenders
					# first segment
					tseg = [s['tmin']] + \
						   [t for t in ts if t < iass_seg[p]['t'][0][0]+0.1] # in case not assigned at the beginning
					# print(min(tseg), max(tseg))
					iicon, = plt.plot([s['x'](t) for t in tseg], [s['y'](t) for t in tseg], 
							 color=np.array([0.1, 1., 1.]), lw=self.lw, linestyle=linestyle,
							 label=p, marker='>', ms=self.ms_traj, markevery=1000)
					# the rest of the segments
					for tseg, d in zip(iass_seg[p]['t'], iass_seg[p]['d']):
						plt.plot([s['x'](t) for t in tseg], [s['y'](t) for t in tseg], 
								 color=dcolors[int(d[1:])], lw=self.lw, linestyle=linestyle,
								 label=p)
				else: # if the intruder is not assigned at all
					iicon, = plt.plot([s['x'](t) for t in ts], [s['y'](t) for t in ts], 
							 color=np.array([0.1, 1., 1.]), lw=self.lw, linestyle=linestyle,
							 label=p, marker='>', ms=self.ms_traj, markevery=1000)

			if twpts is not None:		
				for tc in twpts:
					if 'D' in p:
						plt.plot(s['x'](tc), s['y'](tc), color=dcolors[int(p[1:])], marker='o', ms=self.ms_traj, lw=self.lw)
						for t in dcap[p]['t']:
							tempx, tempy, capr = s['x'](t), s['y'](t), self.r
							picon = plt.Circle([tempx, tempy], radius=capr, color=dcolors[int(p[1:])], 
												alpha=.02, ls=linestyle, lw=self.lw)
							plt.gca().add_patch(picon)
							temptht = np.linspace(0, 2*pi, 50)
							plt.plot([tempx+capr*cos(tht) for tht in temptht],
									 [tempy+capr*sin(tht) for tht in temptht],
									 ls=linestyle, lw=self.lw, color=dcolors[int(p[1:])],
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
										marker='>', ms=self.ms_traj, lw=self.lw)
							# other segments				
							for tseg, d in zip(iass_seg[p]['t'], iass_seg[p]['d']):
								if tseg[0] <= tc <= tseg[-1]:
									ttemp = [abs(tc-tt) for tt in tseg]
									tc = tseg[ttemp.index(min(ttemp))] # this is to make sure that the markers are on the trajectory
									plt.plot(s['x'](tc), s['y'](tc), color=dcolors[int(d[1:])], marker='>', ms=self.ms_traj, lw=self.lw)
						else:
							plt.plot(s['x'](tc), s['y'](tc), color=np.array([0.1, 1., 1.]), marker='>', ms=self.ms_traj, lw=self.lw)

		return tuple(lgd), (iicon)

	def plot_target(self):
		# xt, yt, rt = param['x0targ'], param['y0targ'], param['Rtarg']
		tht = np.linspace(0, 2*pi, 50)
		target = plt.Circle((self.xt, self.yt), self.rt, color='blue', alpha=0.2)
		plt.gca().add_patch(target)
		plt.plot([self.xt+self.rt*cos(t) for t in tht], [self.yt+self.rt*sin(t) for t in tht], linewidth=self.lw, label='target')

	#####################################################################
	# |							plot figures							|
	#####################################################################
	def plot_assign(self, assign, cap):
		
		skip = 4
		circscale = 4.
		circoffset = 6.

		plt.figure(figsize=self.plot_size)
		lgp, lga = [], []
		for d, (p, ass) in enumerate(assign.items()):
			icon_pref, = plt.plot(-1, -1, 'C0o', 
								 ms=self.ms, alpha=0.2, color=dcolors[d])
			icon_ass, = plt.plot(-1, -1, 'C0o', 
								 ms=self.ms, color=dcolors[d])
			# icon_e, = plt.plot([0, 1], [-10, -12], color=dcolors[d], linewidth=lw)
			lgp.append(icon_pref)
			lga.append(icon_ass)
			# lge.append(icon_e)
			tmin, tmax = 0, 1e10
			for k, (t, pref_dict, itarg) in enumerate(zip(ass['t'], ass['pref'], ass['i'])):
				tmin = max(tmin, t.min())
				tmax = min(tmax, t.max())
				# print(t)
				if k%skip == 0:
					for i, e in pref_dict.items():# preferred intruder
						plt.plot(t, circoffset+circscale*int(i[1:]), 'C0o', 
								 ms=self.ms, alpha=0.2, color=dcolors[d])
					
					plt.plot(t, circoffset+circscale*itarg, 'C0o', 
							 ms=self.ms, color=dcolors[d]) # targeted intruder

			# plt.plot(ass['t'], ass['e'], color=dcolors[d], linewidth=lw)

		plt.grid()
		plt.gca().tick_params(axis='both', which='major', labelsize=self.fs)
		plt.gca().tick_params(axis='both', which='minor', labelsize=self.fs)
		# plt.ylim(0, 36)
		plt.xlim(tmax, tmin)
		plt.subplots_adjust(left=.09, bottom=.3, right=.9, top=.9)
		plt.yticks([circoffset+circscale*i for i in range(4) if i%1==0], 
					[r'$'+str(i)+'$' for i in range(4) if i%1==0])
		plt.ylabel(r'$I_i$', fontsize=self.fs)
		plt.xlabel(r'$t(s)$', fontsize=self.fs)

		ax2 = plt.gca().twinx()
		lge = []
		for d, (p, ass) in enumerate(assign.items()): 
			e, = ax2.plot(ass['t'], ass['e'], color=dcolors[d], lw=self.lw)
			lge.append(e)

		ax2.tick_params(axis='both', which='major', labelsize=self.fs)
		ax2.tick_params(axis='both', which='minor', labelsize=self.fs)
		ax2.set_ylabel(r'$e$', fontsize=self.fs)
		ax2.set_ylim(-.1, 20.)	
		plt.legend([tuple(lgp), tuple(lga), tuple(lge)], [r'preferred', r'assigned', r'$e$'],
					fontsize=self.fs*0.75, loc='lower right',
					handler_map={tuple: HandlerTuple(ndivide=None)})			
		plt.show()

	# def compare_traj(tsg, tss, states_gazebo, states_simple, cap, param):
	def compare_traj(self, ts_gazebo, states_gazebo, assign_gazebo, cap_gazebo,
					 ts_simple, states_simple, assign_simple, cap_simple):

		iass_seg_gazebo = self.process_assign(ts_gazebo, assign_gazebo)
		dcap_gazebo, twpts_gazebo = self.process_cap(ts_gazebo, cap_gazebo)

		iass_seg_simple = self.process_assign(ts_gazebo, assign_simple)
		dcap_simple, twpts_simple = self.process_cap(ts_simple, cap_simple, tmin=ts_simple[0])
		
		plt.figure(figsize=self.traj_size)
		lgd, lgi = self.plot_traj(ts_gazebo, states_gazebo, iass_seg_gazebo, dcap_gazebo, twpts_gazebo, linestyle='solid')
		_,   _   = self.plot_traj(ts_simple, states_simple, iass_seg_simple, dcap_simple, twpts_simple, linestyle='dashed')
		self.plot_target()

		plt.axis('equal')
		plt.grid()
		plt.gca().tick_params(axis='both', which='major', labelsize=self.fs)
		plt.gca().tick_params(axis='both', which='minor', labelsize=self.fs)
		plt.xlabel(r'$x(m)$', fontsize=self.fs)
		plt.ylabel(r'$y(m)$', fontsize=self.fs)
		plt.xlim(-2, 2)
		plt.ylim(-1, 2)
		plt.legend([lgd, lgi], [r'$D$', r'$I$'],
				fontsize=self.fs*0.75, loc='upper right',
				handler_map={tuple: HandlerTuple(ndivide=None)})		
		plt.show()
		# plt.savefig(self.res_path+'trajectory_samestart.jpg')
		# plt.close()


	def compare_traj_and_v(self, ts, states, assign, cap, name='gazebo'):

		iass_seg = self.process_assign(ts, assign)
		dcap, twpts = self.process_cap(ts, cap)
		
		plt.figure(figsize=self.traj_size)
		self.plot_traj(ts, states, iass_seg, dcap, twpts)
		self.plot_target()

		plt.axis('equal')
		plt.grid()
		plt.gca().tick_params(axis='both', which='major', labelsize=self.fs)
		plt.gca().tick_params(axis='both', which='minor', labelsize=self.fs)
		plt.xlabel(r'$x(m)$', fontsize=self.fs)
		plt.ylabel(r'$y(m)$', fontsize=self.fs)
		# plt.legend(fontsize=fs)
		plt.show()
		# plt.savefig(self.res_path+'trajectory_'+name+'.jpg')
		# plt.close()

	def compare_cmdv(self, ts, cmd_gazebo, cmd_simple):

		plt.figure(figsize=self.plot_size)

		for d in cmd_gazebo:
			if 'D' in d:
				plt.plot(ts, [sqrt(vx**2 + vy**2) for (vx, vy) in zip(cmd_gazebo[d]['vx'](ts), cmd_gazebo[d]['vy'](ts))], 
							color=dcolors[int(d[1:])], lw=self.lw)
				plt.plot(ts, [sqrt(vx**2 + vy**2) for (vx, vy) in zip(cmd_simple[d]['vx'], cmd_simple[d]['vy'])], 
							color=dcolors[int(d[1:])], lw=self.lw, linestyle='dashed')

		plt.grid()
		plt.xlabel('t(s)', fontsize=self.fs)
		plt.ylabel('cmdv(m/s)', fontsize=self.fs)
		plt.gca().tick_params(axis='both', which='major', labelsize=self.fs)
		plt.gca().tick_params(axis='both', which='minor', labelsize=self.fs)
		plt.legend(fontsize=self.fs)
		# plt.show()	
		plt.savefig(res_path+'cmdv_compare_speed.jpg')
		plt.close()


		########## vx
		plt.figure(figsize=self.plot_size)

		for d in cmd_gazebo:
			if 'D' in d:
				plt.plot(ts, cmd_gazebo[d]['vx'](ts), color=dcolors[int(d[1:])], lw=self.lw, label=d)
				plt.plot(ts, cmd_simple[d]['vx'], color=dcolors[int(d[1:])], ls='dashed', lw=self.lw)
				# plt.plot(ts, cmd_gazebo[d]['vy'](ts), color=dcolors[int(d[1:])], linewidth=lw)
				# plt.plot(ts, cmd_simple[d]['vy'], color=dcolors[int(d[1:])], linestyle='dashed', linewidth=lw)

		plt.grid()
		plt.xlabel('t(s)', fontsize=self.fs)
		plt.ylabel('cmdv(m/s)', fontsize=self.fs)
		plt.gca().tick_params(axis='both', which='major', labelsize=self.fs)
		plt.gca().tick_params(axis='both', which='minor', labelsize=self.fs)
		plt.legend(fontsize=self.fs)
		# plt.show()	
		plt.savefig(self.res_path+'cmdv_compare_vx.jpg')
		plt.close()


	def velocity_response(self, ts, cmd_gazebo, states_gazebo):

		plt.figure(figsize=self.plot_size)
		# print('!!!!!!!!')

		for d in cmd_gazebo:
			if 'D' in d:
				plt.plot(ts, [sqrt(cmd_gazebo[d]['vx'](t)**2+cmd_gazebo[d]['vy'](t)**2) for t in ts], color=dcolors[int(d[1:])], 
						 lw=self.lw, label= r'$D_%d'%int(d[1:])+'$')
				plt.plot(ts, [sqrt(states_gazebo[d]['vx'](t)**2+states_gazebo[d]['vy'](t)**2) for t in ts], color=dcolors[int(d[1:])], 
						 linestyle='dashed', lw=self.lw)	

		plt.grid()
		plt.xlabel(r't(s)', fontsize=self.fs)
		plt.ylabel(r'velocity(m/s)', fontsize=self.fs)
		plt.gca().tick_params(axis='both', which='major', labelsize=self.fs)
		plt.gca().tick_params(axis='both', which='minor', labelsize=self.fs)
		plt.legend(fontsize=self.fs)
		plt.savefig(self.res_path+'velocity_response_speed.jpg')
		plt.close()

		########### vx
		plt.figure(figsize=self.plot_size)
		lgc, lgr = [], []
		for d in cmd_gazebo:
			if 'D' in d:
				iconc, = plt.plot(ts, cmd_gazebo[d]['vx'](ts), color=dcolors[int(d[1:])], 
						 lw=self.lw, label=r'$D_%d'%int(d[1:])+'$')
				iconr, = plt.plot(ts, states_gazebo[d]['vx'](ts), color=dcolors[int(d[1:])], 
						 linestyle='dashed', lw=self.lw)	
				lgc.append(iconc)
				lgr.append(iconr)


		plt.grid()
		plt.ylim(-.26, .3)
		plt.xlabel(r'$t(s)$', fontsize=self.fs)
		plt.ylabel(r'$v_x(m/s)$', fontsize=self.fs)
		plt.gca().tick_params(axis='both', which='major', labelsize=self.fs)
		plt.gca().tick_params(axis='both', which='minor', labelsize=self.fs)
		plt.subplots_adjust(left=.16, bottom=.3, right=.96, top=.9)
		plt.legend([tuple(lgc), tuple(lgr)], [r'command', r'actual'],
				fontsize=self.fs*0.75, loc='upper left',
				handler_map={tuple: HandlerTuple(ndivide=None)})	
		# plt.legend(fontsize=fs*0.8)
		# plt.savefig(self.res_path+'velocity_response_vx.jpg')
		# plt.close()
		plt.show()
