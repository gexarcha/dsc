import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rc('text', usetex=False)
import tables as tb
import os
import tqdm

#dsc_run_audio.py.2016-04-21+09:50
outputdir = 'output/'+'dsc_run_audio.py.2016-04-21+09:50/'
ofile = outputdir+'result.h5'

fh =tb.open_file(ofile,'r')

sigma_all = fh.root.sigma.read().squeeze()
pi_all = fh.root.pi.read().squeeze()

W_all = fh.root.W.read()
H = W_all.shape[-1]
D = W_all.shape[-2]
h5nodes = [f.name for f in fh.root._f_list_nodes()]
rseries=None
if 'rseries' in h5nodes:
	rseries = fh.root.rseries.read()
series=None
if 'series' in h5nodes:
	series = fh.root.series.read()
channel=None
if 'channel' in h5nodes:
	channel = fh.root.channel.read()
overlap=0.
# overlap=0.5
if 'overlap' in h5nodes:
	overlap = fh.root.overlap.read()
inf_poster=None
if 'infered_posterior' in h5nodes:
	inf_poster = fh.root.infered_posterior.read().squeeze()
inf_states=None
if 'infered_states' in h5nodes:
	inf_states = fh.root.infered_states.read().squeeze()
ry=None
if 'ry' in h5nodes:
	ry = fh.root.ry.read().squeeze()
rs=None
if 'rs' in h5nodes:
	rs = fh.root.rs.read().squeeze()
N = fh.root.N.read()[0]
psz = np.int(np.sqrt(D))
fh.close()
epochs = W_all.shape[0]

def findpeaks(a,inds,max=True):
	d=np.diff(a.squeeze())
	di=np.diff(inds.squeeze())
	p=[]

	for i in range(2,d.shape[0]):
		if max:
			if a[i-2]<a[i-1] and a[i]<a[i-1] and np.all(di[i-2:i]==1):
				p.append(i-1)
		else:
			if a[i-2]>a[i-1] and a[i]>a[i-1] and np.all(di[i-2:i]==1):
				p.append(i-1)
	p = np.array(p)
	if p.shape[0]==0:
		return np.array([])
	else:
		return inds[p]

cscale='local'
if not os.path.exists(outputdir+'montage_images'):
	os.mkdir(outputdir+'montage_images')
if not os.path.exists(outputdir+'_images'):
	os.mkdir(outputdir+'_images')
if not os.path.exists(outputdir+'reconstructions'):
	os.mkdir(outputdir+'reconstructions')
for e in range(0,epochs,5)[::-1]:
	# minwg = -np.max(np.abs(W_all[e])) 
	# maxwg = -minwg
	minwg = np.min(W_all[e]) 
	maxwg = np.max(W_all[e]) 
	meanw = np.mean(W_all) 
	if not os.path.exists('{}montage_images/W_e_{:03}.jpg'.format(outputdir, e)):
		for h in range(H):
			if os.path.exists(outputdir+'_images/'+'W_e_{:03}_h_{:03}.jpg'.format(e,h)):
				continue
			this_W = W_all[e,:,h]
			# this_W=this_W.reshape((psz,psz))
			# minwl = -np.max(np.abs(this_W)) 
			# maxwl = -minwl 
			minwl = np.min(this_W) 
			maxwl = np.max(this_W)
			meanwl = np.mean(this_W) 
			if cscale == 'global':
				maxw, minw = maxwg, minwg
			elif cscale == 'local':
				maxw, minw = maxwl, minwl
			plt.plot(np.linspace(0,D/10,num=D) ,this_W)#scale in kHz
			plt.axis([0,D/10,minwg,maxwg])
			plt.savefig(outputdir+'_images/'+'W_e_{:03}_h_{:03}.jpg'.format(e,h))

			plt.clf()
			if h%30 == 0 :
				print "Finished epoch {:03} basis {:03}".format(e,h)
				print "\tPlot settings scale: '{}', min: {}, max: {}, mean:{}".format(cscale,minwg,maxwg,meanw)
				plt.clf()

		os.system("montage -trim {}_images/W_e_{:03}_h*.jpg {}montage_images/W_e_{:03}.jpg".format(outputdir,e,outputdir,e))
		os.system("rm {}_images/W_e_{:03}_h*.jpg ".format(outputdir, e))
# os.system("convert -delay 10 {}montage_images/* {}W_training.gif".format(outputdir,outputdir))
if series is not None and rseries is not None:
	#IC = series.squeeze()[5,:].squeeze()
        series = series.squeeze()#[channel,:].squeeze()
	rseries = rseries.squeeze()
	T = rseries.shape[0]
	l = 1000
	s=0
	step=int(D*(1-overlap))
	reconstates = []
	recondata = []
	lims = []
	for n in tqdm.tqdm(range(N),'reconstructing'):
	    e = s+D
	    lims.append([s,e])
	    s+=step
	reconseries2 = np.zeros((lims[-1][1],))
	assert overlap<=0.5
	s=0
	lims2 = [0]
	rsts = [rs[0]]
	# import ipdb; ipdb.set_trace()
	for n in tqdm.tqdm(range(1,N),'reconstructing'):
		padding  = lims2[-1]-lims[n-1][0]
		if inf_poster[n,0]>inf_poster[n-1,0]:
			reconseries2[lims2[-1]:lims[n-1][1]]=ry[n-1][padding:]
			# reconseries2[lims[n-1][0]:lims[n-1][1]]=ry[n-1]
			reconseries2[lims[n][0]:lims[n][1]]=ry[n]
			lims2.append(lims[n][0])
			# rsts.append(rs[n])
			# rsts.append(rs[n])
		else:
			reconseries2[lims[n][0]:lims[n][1]]=ry[n]
			reconseries2[lims2[-1]:lims[n-1][1]]=ry[n-1][padding:]
			# rsts.append(rs[n])
			lims2.append(lims[n-1][1])
			# rsts.append(rs[n])
		if lims2[-1]==lims2[-2]:
			lims2.pop()
			rsts.pop()
		# else:
		rsts.append(rs[n])
	lims2=np.array(lims2)
	rsts=np.array(rsts)

	# reconstates = np.array(reconstates)
	# lims3 = np.array(lims3)
	mins = np.min(series)
	maxs = np.max(series)
	minrs = np.min(rseries)
	maxrs = np.max(rseries)
	minic = np.min(IC)
	maxic = np.max(IC)
	minb = np.minimum(mins,minrs)
	maxb = np.maximum(maxs,maxrs)
	c=0
	# plt.clf()
	# import ipdb; ipdb.set_trace()  # breakpoint 3ef80612 //
	c=0
	for s in range(0,T-l,l):
	# for s in tqdm.tqdm(range(0,T-l,l),'plotting'):
	# for s in tqdm.tqdm(range(30000,T-l,l),'plotting'):
		fig = plt.figure(1,(10,15))
		ax1 = fig.add_subplot(3,1,1)
		orig = series[s:s+l]
		recon = reconseries2[s:s+l]
		# trsts = reconstates[s/step:s/step+l/step]
		#thisIC = IC[s:s+l]
		xdata = np.linspace(s,s+l, l)
		these_lims = lims2[lims2>=s]
		these_lims = these_lims[these_lims<s+l]
		trsts = rsts[lims2>=s]
		trsts = trsts[:these_lims.shape[0]]
		# xdata = np.linspace(s,s+l, l)
		# print xdata.shape,orig.shape,recon.shape

		ax1.plot(xdata,orig,label='original')
		ax1.plot(xdata,recon,label='reconstruction')
		ax1.axis([s,s+l,minb,maxb],fontsize=16)
		ax1.tick_params(axis='both',labelsize=16)
		# ax.yticks(fontsize=16)
		handles, labels = ax1.get_legend_handles_labels()
		lgd1 = ax1.legend(handles,labels, loc='upper right', bbox_to_anchor=(1.,1.),fontsize=9)
		ax1.grid('on')

		# plt.savefig(outputdir+'reconstructions/'+'series_{}_{}.jpg'.format(s,s+l))
		# if not os.path.exists('{}montage_images/orig_{:03}.jpg'.format(outputdir, e)):
		
		ax2 = fig.add_subplot(3,1,2)
		ax2.axis([s,s+l,0,1],fontsize=16)
		for w in range(trsts.shape[0]):
		# for w in tqdm.tqdm(range(trsts.shape[0]),'point',nested=True):
			for h in range(H):
				width = these_lims[w]+2
				height = float(h)/H
				# if these_lims[w,0]==these_lims[w,1]:
				# 	continue
				# print width,height
				# if trsts[w,h]!=0:
				ax2.text(width,height,'{}'.format(trsts[w,h]),fontsize=5)
		ax2.axis('off')
		ax3 = fig.add_subplot(3,1,3)

		ax3.plot(xdata,thisIC,label='IC')
		
		thisICg200 = np.argwhere(thisIC>200)
		# print thisICg200,thisIC[thisICg200]
		peaks = findpeaks(thisIC[thisICg200],thisICg200)
		if peaks.shape[0]>0:
			ploc = peaks
			for p in ploc:
				# print (p[0]-25.)/l,0.01,50./l,0.99
				pat = patches.Rectangle((s+p[0]-25,minic),50.,4*(maxic-minic),fill=True,alpha=0.3,color='red')
				ax3.add_patch(pat)
				# ax.axvspan(p[0]-25,p[0]+25,color='red',alpha=0.5)

		ax3.axis([s,s+l,minic,maxic],fontsize=16)
		handles, labels = ax3.get_legend_handles_labels()
		lgd2 = ax3.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.,1.),fontsize=9)
		ax3.tick_params(axis='both',labelsize=16)
		ax3.grid('on')

		for i in range(these_lims.shape[0]):
			ax1.axvline(x=these_lims[i],ymin=0,ymax=1,c="red",linewidth=.5,zorder=0, clip_on=False)
			ax2.axvline(x=these_lims[i],ymin=-0.2,ymax=1.2,c="green",linewidth=.5,zorder=0, clip_on=False)
			ax3.axvline(x=these_lims[i],ymin=0,ymax=1,c="blue",linewidth=.5,zorder=0, clip_on=False)
		fig.savefig(outputdir+'reconstructions/'+'series_{}_{}_n.jpg'.format(s,s+l), bbox_extra_artists=(lgd1,lgd2), bbox_inches = 'tight')
		plt.close(fig)
		c+=1
plt.clf()
plt.plot(sigma_all,label='$\sigma$')
plt.savefig(outputdir+'sigma.jpg')
plt.clf()
