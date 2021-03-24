import os
import argparse
import rosbag
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('datafile', default='data_00', type=str, help='data file name')
args = parser.parse_args()
# fdir = '~/home/flora/crazyflie_ws/src/crazyflie_mdmifd/data/' + args.datafile
fnames = args.datafile.split(' ')

for fname in fnames:

	fdir = '/home/flora/crazyflie_mdmifd_expdata/' + fname
	print('>>>>>> reading from' + fdir)

	# read the most recent bag
	imax, tmax, dmax = 0, 0, ''
	for i, t in enumerate(next(os.walk(fdir))[2]): # upstream dir, dir, files
		ttemp = int(''.join([s for s in t.split('.')[0].split('-')]))
		if ttemp > tmax:
			imax = i
			tmax  = ttemp
			dmax = t
	bag = rosbag.Bag(fdir+'/'+dmax)

	# parameters
	param = pd.read_csv(fdir + '/D0/param.csv').set_index('param').T.to_dict('records')[0]
	temp, cf_dict = param['cf_dict'].split('!'), dict()
	for rcf in temp:
		r_cf = rcf.split('_')
		cf_dict.update({r_cf[0]:r_cf[1]}) # role: cf

	# read the data
	# df_dict = {p:[None, None] for p in cf_dict}

	for p, cf in cf_dict.items():

		# locations and velocities
		df = pd.DataFrame(columns=['t', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
		for topic, msg, ros_t in bag.read_messages(topics=['/'+cf+'/mocap']):

			t = ros_t.secs + ros_t.nsecs * 1e-9

			x = msg.position[0]
			y = msg.position[1]
			z = msg.position[2]

			vx = msg.velocity[0]
			vy = msg.velocity[1]
			vz = msg.velocity[2]

			df = df.append(
		    	{'t': t,
				 'x': x,
				 'y': y,
				 'z': z,
				 'vx': vx,
				 'vy': vy,
				 'vz': vz},
				 ignore_index=True)

		# df_dict[p][0] = df
		out_file = fdir+'/'+p+'/State.csv'
		if os.path.exists(out_file): 
			os.remove(out_file) 

		df.to_csv(out_file)

		# velocity command
		df = pd.DataFrame(columns=['t', 'vx', 'vy'])
		for topic, msg, ros_t in bag.read_messages(topics=['/'+cf+'/cmdV']):

			t = ros_t.secs + ros_t.nsecs * 1e-9

			vx = msg.linear.x
			vy = msg.linear.y

			df = df.append(
		    	{'t': t,
				 'vx': vx,
				 'vy': vy},
				 ignore_index=True)

		# df_dict[p][1] = df
		out_file = fdir+'/'+p+'/Command.csv'
		if os.path.exists(out_file): 
			os.remove(out_file) 
			
		df.to_csv(out_file)	

	# tmin = min([min(min(data[0]['t']), min(data[1]['t'])) for p, data in df_dict.items()])
	# for p, data in df_dict.items():
	# 	data[0]['t'] = data[0]['t'] - tmin
	# 	data[1]['t'] = data[1]['t'] - tmin

	# 	out_file = fdir+'/'+p+'/State.csv'
	# 	if os.path.exists(out_file): 
	# 		os.remove(out_file) 
	# 	data[0].to_csv(out_file)

	# 	out_file = fdir+'/'+p+'/Command.csv'
	# 	if os.path.exists(out_file): 
	# 		os.remove(out_file) 
	# 	data[1].to_csv(out_file)			