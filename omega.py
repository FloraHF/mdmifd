import numpy as np

from math import sin, cos, atan2, tan, pi
import matplotlib.pyplot as plt

# from matplotlib import rc
# rc("text", usetex=True)

a = 1.2
Omega_ = 1
# kmax = .9
kmax = 1


def beta_omega(beta):
	# return Omega_*(a*cos(beta)-1)/(a*sin(beta))**3
	return a*sin(beta)*(a*cos(beta)-1)/(1+a**2-2*a*cos(beta))/(1-a**2)


def beta_theta(beta):
	# a = 1.2
	# # beta = 0.2*pi
	# Omega = 1.25

	Omega = beta_omega(beta)

	cphi = a*sin(beta)
	sphi = 1 - a*cos(beta)
	phi = atan2(sphi, cphi)

	cpsi = 1 - a*cos(beta)
	spsi = a*sin(beta)
	psi = atan2(spsi, cpsi)

	stht = Omega
	ctht = Omega/tan(psi) + 1
	tht = atan2(stht, ctht)

	return tht

def det(gmm, beta=1.2):
	theta = beta_theta(beta)

	# print(theta, gmm)
	return sin(gmm + theta)

def res_kmin(gmm, beta=1.2):
	Omega = beta_omega(beta)
	return - (sin(gmm) + Omega)

def res_kmax(gmm, beta=1.2):
	Omega = beta_omega(beta)
	theta = beta_theta(beta)
	return kmax*Omega*sin(gmm + theta) - (sin(gmm) + Omega)

def judgement(gmm, d, r_kmin, r_kmax):

	# get separated segments where d>0 and d<0
	d_sign = {'True':[], 'False':[]} # True: d>0, False: d<0
	for i, dd in enumerate(d):
		ispos = dd > 0
		if i == 0: 
			previous = ispos
			d_sign[str(ispos)].append([i])
		else:
			if ispos != previous:
				d_sign[str(ispos)].append([])
				previous = ispos
			d_sign[str(ispos)][-1].append(i)

	# if d>0, use r_max
	r_kmax_dpos = {'gmm':[], 'res':[]}
	nseg_dpos = 0
	for seg in d_sign['True']:
		nseg_dpos += 1
		rseg = []
		gseg = []
		for idx in seg:
			rseg.append(r_kmax[idx])
			gseg.append(gmm[idx])
		r_kmax_dpos['gmm'].append(gseg)
		r_kmax_dpos['res'].append(rseg)

	# if d<0, use k_min
	r_kmin_dneg = {'gmm':[], 'res':[]}
	nseg_dneg = 0
	for seg in d_sign['False']:
		nseg_dneg += 1
		rseg = []
		gseg = []
		for idx in seg:
			rseg.append(r_kmin[idx])
			gseg.append(gmm[idx])
		r_kmin_dneg['gmm'].append(gseg)
		r_kmin_dneg['res'].append(rseg)

	r_kmax_out = []	
	for i in range(nseg_dpos):
		r_kmax_out.append({'gmm':np.array(r_kmax_dpos['gmm'][i]),
							'res':np.array(r_kmax_dpos['res'][i])}) 
	r_kmin_out = []	
	for i in range(nseg_dneg):
		r_kmin_out.append({'gmm':np.array(r_kmin_dneg['gmm'][i]),
							'res':np.array(r_kmin_dneg['res'][i])}) 

	return r_kmax_out, r_kmin_out
	# print(d_sign)
############## beta vs Omega
# beta = np.linspace(0.001, pi-0.001, 50)
# theta = np.array([beta_theta(b) for b in beta])

# plt.plot(-beta, theta)
# plt.grid()
# plt.show()
		
############## gmm vs sin(gmm+theta), gmm vs - sin(gmm) - Omega
cs = ['r', 'b', 'm', 'k']
for b, c in zip([pi/1.5], cs):
	gmm = np.linspace(0, 2*pi, 60)
	d = np.array([det(g, beta=b) for g in gmm])

	r_kmin = np.array([res_kmin(g, beta=b) for g in gmm])
	r_kmax = np.array([res_kmax(g, beta=b) for g in gmm])

	r_kmax_, r_kmin_ = judgement(gmm, d, r_kmin, r_kmax)

	for seg in r_kmax_:
		plt.plot(seg['gmm']*180/pi, seg['res'], '--', c=c, label='r_kmax')
	for seg in r_kmin_:
		plt.plot(seg['gmm']*180/pi, seg['res'], '-.', c=c, label='r_kmin')
	plt.plot(gmm*180/pi, d, c=c, label='delt')

	gmm_min = (pi/2-b+pi)*180/pi
	gmm_max = (pi/2-b+2*pi)*180/pi
	plt.plot([gmm_min, gmm_min], [-1.5, 1.1], '--', c=c)
	plt.plot([gmm_max, gmm_max], [-1.5, 1.1], '--', c=c)

	plt.plot(gmm*180/pi, r_kmin, '--', c=c, label='r_kmin')
	plt.plot(gmm*180/pi, r_kmax, '-.', c=c, label='r_kmax')
	

	print()

plt.grid()
# plt.legend()
plt.show()