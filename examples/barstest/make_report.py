#
#  Lincense: Academic Free License (AFL) v3.0
#
import numpy as np
import tables as tb
import matplotlib.pyplot as plt

outputdir = 'output/'+'dsc_run_tvem.py.2015-12-16+17:04/'
ofile = outputdir+'result.h5'

fh =tb.open_file(ofile,'r')

W_all = fh.root.W.read()
H = W_all.shape[-1]
D = W_all.shape[-2]
psz = np.int(np.sqrt(D))

epochs = W_all.shape[0]

cscale='local'
for e in range(epochs):
	minwg = -np.max(np.abs(W_all[e])) 
	maxwg = -minwg 
	for h in range(H):
		this_W = W_all[e,:,h]
		this_W=this_W.reshape((psz,psz))
		minwl = -np.max(np.abs(this_W)) 
		maxwl = -minwl 
		if cscale == 'global':
			plt.imshow(this_W,interpolation='nearest',vmin=minwg,vmax=minwg)
		elif cscale == 'local':
			plt.imshow(this_W,interpolation='nearest',vmin=minwl,vmax=minwl)
		plt.axis('off')
		plt.savefig(outputdir+'_images'+'W_e_{:03}_h_{:03}.png'.format(e,h))