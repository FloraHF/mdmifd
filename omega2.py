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

def left(gmm, beta=pi/3):

	Omega = beta_omega(beta)
	theta = beta_theta(beta)
	
	return max(0, kmax*Omega*sin(theta+gmm))

def right(gmm, beta=pi/3):

	Omega = beta_omega(beta)

	return sin(gmm) + Omega

def wrap_angle(angle):
	if angle > 2*pi:
		return angle - 2*pi
	if angle < 0.:
		return angle + 2*pi
	return angle

		
############## gmm vs sin(gmm+theta), gmm vs - sin(gmm) - Omega
cs = ['g', 'y', 'r', 'm', 'b', 'k']
bmin = atan2(a, 1)
bs = np.linspace(bmin, (bmin+pi)/2, 6)


for b, c in zip(bs, cs):
	gmm = np.linspace(0, 2*pi, 60)

	d = np.array([left(g, beta=b)-right(g, beta=b) for g in gmm])
	plt.plot(gmm*180/pi, d, '--', c=c, label=str(b))

	gmm_min = wrap_angle(-pi/2-b)*180/pi
	gmm_max = wrap_angle(pi/2-b)*180/pi

	# print(gmm_min, gmm_max)

	d_ = []
	gmm_ = []
	for i, g in enumerate(gmm):
		if gmm_min <= g*180/pi <= gmm_max or gmm_max <= g*180/pi <= gmm_min:
			d_.append(d[i])
			gmm_.append(gmm[i]*180/pi)
	plt.plot(gmm_, d_, c=c)
	
	# plt.plot([gmm_min, gmm_min], [-1.5, 1.1], '--', c=c)
	# plt.plot([gmm_max, gmm_max], [-1.5, 1.1], '--', c=c)

plt.grid()
plt.legend()
plt.show()