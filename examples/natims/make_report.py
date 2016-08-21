#
#  Lincense: Academic Free License (AFL) v3.0
#
import numpy as np
import tables as tb
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 26
mpl.rcParams['font.size'] = 26
import matplotlib.pyplot as plt
import os
import tqdm

outputdir = 'output/'+'dsc_run.py.d3981/' #dDSC
outputdir = 'output/'+'dsc_run.py.d3985/' #tDSC
outputdir = 'output/'+'dsc_run.py.d4097/' #bDSC 
ofile = outputdir+'result.h5'

fh =tb.open_file(ofile,'r')

sigma_all = fh.root.sigma.read().squeeze()
pi_all = fh.root.pi.read().squeeze()
states = fh.root.states.read().squeeze()
K =states.shape[0]
K_0 =np.arange(K)[states==0]
W_all = fh.root.W.read()
H = W_all.shape[-1]
D = W_all.shape[-2]
psz = np.int(np.sqrt(D))
fh.close()
epochs = W_all.shape[0]
statemin = np.min(states)
statemax = np.max(states)
statengmin = np.min(states[states!=0])
statengmax = np.max(states[states!=0])
cscale='local'
if not os.path.exists(outputdir+'montage_images'):
	os.mkdir(outputdir+'montage_images')
if not os.path.exists(outputdir+'_images'):
	os.mkdir(outputdir+'_images')
for e in tqdm.tqdm(range(epochs)[::-1],'epochs'):
	minwg = -np.max(np.abs(W_all[e])) 
	maxwg = -minwg 
	minpig = 0 
	maxpig = np.max(pi_all) 
	maxping = np.max(pi_all[:,states!=0])*1.1


	if not os.path.exists('{}montage_images/pi_broken_{:03}.jpg'.format(outputdir, e)):
		f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
		ax.bar(states,pi_all[e,:],align='center')	
		ax2.bar(states,pi_all[e,:],align='center')	
		yl = pi_all[e,np.argsort(pi_all[e,:])[:-1]].sum()*1.25
		print yl,np.argsort(pi_all[e,:])[:-1]
		ax.set_ylim(1.-yl ,1.)	
		ax2.set_ylim(0,yl)	
		# hide the spines between ax and ax2
		ax.spines['bottom'].set_visible(False)
		ax2.spines['top'].set_visible(False)
		ax.xaxis.tick_top()
		ax.tick_params(labeltop='off')  # don't put tick labels at the top
		ax2.xaxis.tick_bottom()

		d = 0.015 # how big to make the diagonal lines in axes coordinates
		# arguments to pass plot, just so we don't keep repeating them
		kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
		ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
		ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

		kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
		ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
		ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

		ax2.set_xticks(states)
		ax2.tick_params(axis='x', which='major', length=1)
		ax2.tick_params(axis='x', which='minor', length=1)
		f.savefig(outputdir+'montage_images/pi_broken_{:03}.jpg'.format(e))
		plt.close(f)

	if not os.path.exists('{}montage_images/W_e_{:03}.jpg'.format(outputdir, e)):
		fig = plt.figure(figsize=(30,10))
		for h in tqdm.tqdm(range(H),'Hidden Basis',nested=True):
			this_W = W_all[e,:,h]
			this_W=this_W.reshape((psz,psz))
			minwl = -np.max(np.abs(this_W)) 
			maxwl = -minwl 
			if cscale == 'global':
				maxw, minw = maxwg, minwg
			elif cscale == 'local':
				maxw, minw = maxwl, minwl
			ax = fig.add_subplot(10,30,h+1)
			ax.imshow(this_W,interpolation='nearest',vmin=minwl,vmax=maxwl)
			ax.axis('off')

		fig = plt.figure(figsize=(30,10))
		fig.savefig("{}montage_images/W_e_{:03}.jpg".format(outputdir,e))
		plt.close(fig)
	if not os.path.exists('{}montage_images/hist_W_e_{:03}.jpg'.format(outputdir, e)):
		fig = plt.figure(figsize=(30,10))
		ax = fig.add_subplot(111)
		ax.hist(W_all[e,:,:].reshape(-1),bins=100,normed=True)
		fig.savefig("{}montage_images/hist_W_e_{:03}.jpg".format(outputdir,e))
		plt.close(fig)
os.system("convert -delay 10 {}montage_images/pi* {}pi_training.gif".format(outputdir,outputdir))

plt.clf()
plt.plot(sigma_all,label='$\sigma$')
plt.savefig(outputdir+'sigma.jpg')
plt.clf()
plt.bar(states,pi_all[-1,:],label='$\pi$')
plt.savefig(outputdir+'pi.jpg')
plt.clf()
plt.bar(states[states!=0],pi_all[-1,:][states!=0],label='$\pi$')
plt.savefig(outputdir+'pi_nonzero.jpg')
plt.clf()
sparsity = pi_all[:,states!=0].sum(1)*H
plt.plot(sparsity,label='$\pi H$')
plt.savefig(outputdir+'sparsity.jpg')
plt.clf()
