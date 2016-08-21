#
#  Lincense: Academic Free License (AFL) v3.0
#
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
import os
import tqdm


outputdir = 'output/'+'dsc_run.py.2016-06-28+18:58/'
ofile = outputdir+'result.h5'
dfile = outputdir+'data.h5'

fh =tb.open_file(dfile,'r')
y = fh.root.y.read().squeeze()
fh.close()
fh =tb.open_file(ofile,'r')

sigma_all = fh.root.sigma.read().squeeze()
pi_all = fh.root.pi.read().squeeze()
pi_gt = fh.root.pi_gt.read().squeeze()
W_gt = fh.root.W_gt.read().squeeze()
sigma_gt = fh.root.sigma_gt.read().squeeze()
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
pi_all = pi_all
# pi_comb = np.zeros((pi_all.shape[0],2*pi_all.shape[1]))
# for i in xrange(pi_comb.shape[1]):
# 	if i%2==0:
# 		pi_comb[:,i]=pi_all[:,i]
# 	else:
# 		pi_comb[:,i]=pi_gt[i]
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
	# if not os.path.exists('{}montage_images/npi_{:03}.jpg'.format(outputdir, e)):
	# 	fig, ax = plt.subplots()
	# 	ax.bar(states[states!=0],pi_all[e,states!=0],align='center')
	# 	ax.axis([statengmin-0.5,statengmax+0.5,0,maxping])
	# 	ax.xticks(states[states!=0])
	# 	fig.savefig(outputdir+'montage_images/npi_{:03}.jpg'.format(e))
	# 	plt.close(fig) 
	if not os.path.exists('{}montage_images/pi_{:03}.jpg'.format(outputdir, e)):
		fig, ax = plt.subplots()
		width=0.2
		ax.bar(states,pi_all[e,:],width,align='edge',color='blue')	
		ax.bar(states-width,pi_gt,width,align='edge',color='green')	
		ax.axis([statemin-0.5,statemax+0.5,0,maxpig])
		ax.set_xticks(states)
		fig.savefig(outputdir+'montage_images/pi_{:03}.jpg'.format(e))
		plt.close(fig) 

	if not os.path.exists('{}montage_images/W_e_{:03}.jpg'.format(outputdir, e)):
		fig = plt.figure(figsize=(30,5))
		im=None
		for h in tqdm.tqdm(range(H),'Hidden Basis',nested=True):
			# if os.path.exists(outputdir+'_images/'+'W_e_{:03}_h_{:03}.jpg'.format(e,h)):
				# continue
			this_W = W_all[e,:,h]
			this_W=this_W.reshape((psz,psz))
			minwl = -np.max(np.abs(this_W)) 
			maxwl = -minwl 
			minwle = -np.max(np.abs(W_all[e,:,:])) 
			maxwle = -minwle 
			if cscale == 'global':
				maxw, minw = maxwg, minwg
			elif cscale == 'local':
				maxw, minw = maxwl, minwl
			ax = fig.add_subplot(1,10,h+1)
			im=ax.imshow(this_W,interpolation='nearest',vmin=minwle,vmax=maxwle)
			ax.axis('off')
		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
		# ColorbarBase(cbar_ax,boundaries=[minwle,maxwle],values=np.linspace(minwle,maxwle,2))
		# ColorbarBase(cbar_ax,values=np.linspace(minwle,maxwle))
		fig.colorbar(im,cax=cbar_ax)
		fig.savefig("{}montage_images/W_e_{:03}.jpg".format(outputdir,e))
		plt.close(fig)


#ground truth
fig = plt.figure(figsize=(30,5))
im=None
for h in tqdm.tqdm(range(H),'Hidden Basis',nested=True):
	this_W = W_gt[:,h].reshape((psz,psz))+1e-5
	minwle = -np.max(np.abs(this_W)) 
	maxwle = -minwle 
	ax = fig.add_subplot(1,10,h+1)
	im=ax.imshow(this_W,interpolation='nearest',vmin=minwle,vmax=maxwle)
	ax.axis('off')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
# ColorbarBase(cbar_ax,boundaries=[minwle,maxwle],values=np.linspace(minwle,maxwle,2))
# ColorbarBase(cbar_ax,values=np.linspace(minwle,maxwle))
fig.colorbar(im,cax=cbar_ax)
fig.savefig("{}montage_images/W_gt.jpg".format(outputdir,e))
plt.close(fig)

#data
fig = plt.figure(figsize=(30,5))
im=None
for h in tqdm.tqdm(range(H),'Hidden Basis',nested=True):
	this_y = y[h].reshape((psz,psz))+1e-5
	minwle = -np.max(np.abs(this_y)) 
	maxwle = -minwle 
	ax = fig.add_subplot(1,10,h+1)
	im=ax.imshow(this_y,interpolation='nearest',vmin=minwle,vmax=maxwle)
	ax.axis('off')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
# ColorbarBase(cbar_ax,boundaries=[minwle,maxwle],values=np.linspace(minwle,maxwle,2))
# ColorbarBase(cbar_ax,values=np.linspace(minwle,maxwle))
fig.colorbar(im,cax=cbar_ax)
fig.savefig("{}montage_images/example_y.jpg".format(outputdir,e))
plt.close(fig)
#os.system("convert -delay 10 {}montage_images/W* {}W_training.gif".format(outputdir,outputdir))
# os.system("convert -delay 10 {}montage_images/pi* {}pi_training.gif".format(outputdir,outputdir))
# os.system("convert -delay 10 {}montage_images/npi* {}npi_training.gif".format(outputdir,outputdir))
plt.figure()
plt.clf()
plt.plot(sigma_all,label='$\sigma$')
plt.axis([0,100,0,10])
plt.savefig(outputdir+'sigma.jpg')
plt.clf()
plt.plot(sigma_all,label='$\sigma$')
plt.plot(sigma_gt*np.ones_like(sigma_all),linestyle='dashed',
	label='$\sigma_{gt}$')
plt.axis([0,100,0,10])
plt.legend()
plt.savefig(outputdir+'sigma_vs_gt.jpg')
plt.clf()
plt.bar(states,pi_all[-1,:],label='$\pi$')
plt.savefig(outputdir+'pi.jpg')
plt.clf()
plt.bar(states[states!=0],pi_all[-1,:][states!=0],label='$\pi$')
plt.savefig(outputdir+'pi_nonzero.jpg')
plt.clf()
sparsity = pi_all[:,states!=0].sum(1)*H
plt.plot(sparsity,label='crowdedness')
plt.plot(2*np.ones_like(sparsity),label='ground truth')
plt.axis([0,100,0,4])
plt.legend()
plt.savefig(outputdir+'sparsity.jpg')
plt.clf()
