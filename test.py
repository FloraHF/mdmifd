#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 6th, 2020
@author: Flora Fu

"""

# import os
# from copy import deepcopy
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rc
rc('text', usetex=True)

from Envs.scenarios.game_mdmi.geometries import DominantRegion, CircleTarget, norm

n = 50
xd = np.array([3, 2.5])
r = .3
vd = 1. 
vi = .8
target = CircleTarget(1.25)

xs, ys = np.linspace(0, 5, n), np.linspace(0, 5, n)
X, Y = np.meshgrid(xs, ys)
tl = np.zeros(X.shape)
dist = np.zeros(X.shape)
eff = np.zeros(X.shape)
# print(eff)

for i, (xx, yy) in enumerate(zip(X, Y)):
	for j, (x, y) in enumerate(zip(xx, yy)):
		# if 
		xi = np.array([x, y])
		dr = DominantRegion(r, vd/vi, xi, [xd], offset=0)
		xw = target.deepest_point_in_dr(dr)
		tl[i,j] = target.level(xw)
		dist[i,j] = norm(xw - xd)
		eff[i,j] = tl[i,j]/dist[i,j]

fs = 48
plt.figure(figsize=(11,10))

cp = plt.contourf(X, Y, eff, [-2, 0, 1, 1.5, 2, 5, 10, 15])
# cp = plt.contourf(X, Y, tl, [-2, 0, 1, 2, 3, 4, 5])
# cp = plt.contourf(X, Y, dist, [0, .5, 1., 1.5, 2, 3, 5])
cbar = plt.colorbar(cp)
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=fs)

target = Circle(target.state.p_pos, target.size, color='slategrey', alpha=0.99)
plt.gca().add_patch(target)
plt.xlim(0, 5.01)

plt.grid()
plt.gca().tick_params(axis='both', which='major', labelsize=fs)
plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
plt.xlabel(r'$x(m)$', fontsize=fs)
plt.ylabel(r'$y(m)$', fontsize=fs, labelpad=20)
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
plt.show()