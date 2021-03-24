import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

# data copied from stat.csv
Rd = [1., 2., 3., 4]
tl_exp = [0.237, 0.684, 0.612, 0.5105]
tl_sim = [0.468] + [1.092]*3
tc_exp = [17.488, 11.987, 13.0365, 15.565] 
tc_sim = [11.8] + [7.8]*3

# target level
plt.figure(figsize=(13,4))
plt.scatter(Rd, tl_exp, edgecolors='b', facecolors='b', 
			s=300, linewidth=2.5, label='experiment')
plt.scatter(Rd, tl_sim, edgecolors='b', facecolors='None',
			s=300, linewidth=2.5, label='simulation')

fs = 36

plt.grid(zorder=0)
plt.gca().tick_params(axis='both', which='major', labelsize=fs)
plt.gca().tick_params(axis='both', which='minor', labelsize=fs)

plt.xlim(.5, 5.7)
plt.ylim(0, 1.4)
plt.xlabel(r'$R_d(m)$', fontsize=fs)
plt.ylabel(r'$J_D^1(m)$', fontsize=fs)
plt.legend(fontsize=fs*0.8, loc='upper right')
plt.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.3)

plt.show()

# capture time
plt.figure(figsize=(13,4))
plt.scatter(Rd, tc_exp, edgecolors='b', facecolors='b', 
			s=300, linewidth=2.5, label='experiment')
plt.scatter(Rd, tc_sim, edgecolors='b', facecolors='None',
			s=300, linewidth=2.5, label='simulation')

fs = 36

plt.grid(zorder=0)
plt.gca().tick_params(axis='both', which='major', labelsize=fs)
plt.gca().tick_params(axis='both', which='minor', labelsize=fs)

plt.xlim(.5, 5.7)
plt.ylim(5., 19.)
plt.xlabel(r'$R_d(m)$', fontsize=fs)
plt.ylabel(r'$J_D^2(s)$', fontsize=fs)
plt.legend(fontsize=fs*0.8, loc='upper right')
plt.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.3)

plt.show()