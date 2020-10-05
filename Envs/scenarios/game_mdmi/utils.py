def state_to_prefstring(state):
	datalist = []
	for (p, e) in zip(state.o, state.f):
		datalist.append('I' + str(p) + '=%.6f'%e)
	return '_'.join(list(map(str, datalist)))

def prefstring_to_list(pref):
	preflist = dict()
	for data in pref.split('_'):
		p_e = data.split('=')
		p = p_e[0]
		e = float(p_e[1])
		preflist.update({p:e})
	return preflist