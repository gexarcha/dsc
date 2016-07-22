import numpy as np
import tables as tb
import matplotlib.pyplot as plt
import os


outputdir = 'output/'+'dsc_run_tvemgibbs.py.2016-05-18+20:48+1/'
outputdir = 'output/'+'dsc_run.py.d3985/'

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
for e in range(epochs)[::-1]:
	minwg = -np.max(np.abs(W_all[e])) 
	maxwg = -minwg 
	minpig = 0 
	maxpig = np.max(pi_all) 
	maxping = np.max(pi_all[:,states!=0])*1.1
	if not os.path.exists('{}montage_images/npi_{:03}.jpg'.format(outputdir, e)):
		plt.bar(states[states!=0],pi_all[e,states!=0],align='center')
		plt.axis([statengmin-0.5,statengmax+0.5,0,maxping])
		plt.xticks(states[states!=0])
		plt.savefig(outputdir+'montage_images/npi_{:03}.jpg'.format(e))
		plt.clf() 
	if not os.path.exists('{}montage_images/pi_{:03}.jpg'.format(outputdir, e)):
		plt.bar(states,pi_all[e,:],align='center')	
		plt.axis([statemin-0.5,statemax+0.5,0,maxpig])
		plt.xticks(states)
		plt.savefig(outputdir+'montage_images/pi_{:03}.jpg'.format(e))
		plt.clf() 

	if not os.path.exists('{}montage_images/W_e_{:03}.jpg'.format(outputdir, e)):
		for h in range(H):
			if os.path.exists(outputdir+'_images/'+'W_e_{:03}_h_{:03}.jpg'.format(e,h)):
				continue
			this_W = W_all[e,:,h]
			this_W=this_W.reshape((psz,psz))
			minwl = -np.max(np.abs(this_W)) 
			maxwl = -minwl 
			if cscale == 'global':
				maxw, minw = maxwg, minwg
			elif cscale == 'local':
				maxw, minw = maxwl, minwl
			plt.imshow(this_W,interpolation='nearest',vmin=minwl,vmax=maxwl)
			plt.axis('off')
			plt.savefig(outputdir+'_images/'+'W_e_{:03}_h_{:03}.jpg'.format(e,h))

			if h%30 == 0 :
				print "Finished epoch {:03} basis {:03}".format(e,h)
				print "\tPlot settings scale: '{}', min: {}, max: {}".format(cscale,minw,maxw)
			plt.clf()

		os.system("montage -trim {}_images/W_e_{:03}_h*.jpg {}montage_images/W_e_{:03}.jpg".format(outputdir,e,outputdir,e))
		os.system("rm {}_images/W_e_{:03}_h*.jpg ".format(outputdir, e))
#os.system("convert -delay 10 {}montage_images/W* {}W_training.gif".format(outputdir,outputdir))
os.system("convert -delay 10 {}montage_images/pi* {}pi_training.gif".format(outputdir,outputdir))
os.system("convert -delay 10 {}montage_images/npi* {}npi_training.gif".format(outputdir,outputdir))

plt.clf()
plt.plot(sigma_all,label='$\sigma$')
plt.savefig(outputdir+'sigma.jpg')
plt.clf()
plt.bar(states,pi_all[-1,:],label='$\pi$')
plt.savefig(outputdir+'pi.jpg')
plt.clf()
plt.bar(states[states!=0],pi_all[-1,:][states!=0],label='$\pi$')
plt.savefig(outputdir+'pi_nonzero.jpg')
