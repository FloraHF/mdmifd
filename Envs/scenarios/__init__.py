import imp
import os.path as osp


def load(name):
	try:
		pathname = osp.join(osp.dirname(__file__), name+'.py')
		return imp.load_source('', pathname)
	except:
		pathname = osp.dirname(__file__)	
		f, path, desc = imp.find_module(name, [pathname])
		return imp.load_module(name, f, path, desc)