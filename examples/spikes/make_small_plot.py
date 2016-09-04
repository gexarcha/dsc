#
#  Lincense: Academic Free License (AFL) v3.0
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)
import tables as tb
import os
# import tqdm

outputdir = 'output/'+'dsc_on_hc1_run.py.2016-03-17+11:39/'
outputdir = 'output/'+'dsc_on_hc1_run.py.2016-03-17+21:16/'
outputdir = 'output/'+'dsc_on_hc1_run.py.2016-03-23+01:56/'
outputdir = 'output/'+'dsc_on_hc1_run.py.2016-03-23+13:05/'
outputdir = 'output/'+'dsc_on_hc1_run.py.2016-03-29+12:46/'
ofile = outputdir+'result.h5'
fh =tb.open_file(ofile,'r')
sampling_rate=10000.

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
overlap=0.5
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
gamma=None
if 'gamma' in h5nodes:
	gamma = fh.root.gamma.read().squeeze()
states=None
if 'states' in h5nodes:
	states = fh.root.states.read().squeeze()
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
if not os.path.exists(outputdir+'reconstructions_small'):
	os.mkdir(outputdir+'reconstructions_small')
if not os.path.exists(outputdir+'filters'):
	os.mkdir(outputdir+'filters')


state_list = []
std_series=None
print series
print rseries
title_font_dict = {"fontsize":"12"}
marker_font_dict = {"fontsize":"16"}
title_font_dict = None # {"fontsize":"12"}
marker_font_dict = None #{"fontsize":"16"}
time_scale = 1000/sampling_rate
if series is not None and rseries is not None:
	IC = series.squeeze()[5,:].squeeze()
	series = series.squeeze()[channel,:].squeeze()
	rseries = rseries.squeeze()
	T = 64000
	l = 80
	s=0
	step=int(D*(1-overlap))
	reconstates = []
	recondata = []
	lims = []
	for n in range(N):
	    e = s+D
	    lims.append([s,e])
	    s+=step
	reconseries2 = np.zeros((lims[-1][1],))
	std_series=np.std(series[:reconseries2.shape[0]])
	assert overlap<=0.5
	s=0
	lims2 = [0]
	lims3 = [lims[0]]
	rsts = [rs[0]]
	rstssize = [0] # 0 means full, 1 means first half, 2 means second half, 3 means missing 
	# import ipdb; ipdb.set_trace() 
	# Reconstruction code
	# for n in tqdm.tqdm(range(1,N),'reconstructing'):
	for n in range(1,N):
		padding  = lims2[-1]-lims[n-1][0]

		if inf_poster[n,0]>inf_poster[n-1,0]:
			reconseries2[lims2[-1]:lims[n-1][1]]=ry[n-1][padding:]
			# reconseries2[lims[n-1][0]:lims[n-1][1]]=ry[n-1]
			reconseries2[lims[n][0]:lims[n][1]]=ry[n]
			lims2.append(lims[n][0])

			if rstssize[-1]==0:
				rstssize[-1]=1
			elif rstssize[-1]==2:
				rstssize[-1]=3
			rstssize.append(0)
			# rsts.append(rs[n])
			# rsts.append(rs[n])
		else:
			reconseries2[lims[n][0]:lims[n][1]]=ry[n]
			reconseries2[lims2[-1]:lims[n-1][1]]=ry[n-1][padding:]
			# rsts.append(rs[n])
			lims2.append(lims[n-1][1])
			# rstssize[-1]=1
			rstssize.append(2)
			# rsts.append(rs[n])
		if lims2[-1]==lims2[-2]:
			lims2.pop()
			rsts.pop()
		# else:
		rsts.append(rs[n])
	lims2.append(lims[n][1])
	rssize = np.diff(lims2)
	lims2.pop()
	lims2=np.array(lims2)
	lims=np.array(lims)
	rsts=np.array(rsts)
	mass =np.zeros((rsts.shape[0],))
	rstssize = np.array(rstssize)
	# Store how active is the neuron 
	for i in range(4):
		newinds = rstssize[rstssize!=3]
		tmass=mass[newinds==i]
		for j in range(tmass.shape[0]):
			tmass[j]=np.sum(np.mean(np.abs(W_all[-1,:,:]),0)*rsts[newinds==i][j])
		# mass[newinds==i]=np.sum(np.abs(rsts[newinds==i]),1)
		# 
		mass[newinds==i]=tmass
	if rstssize[rstssize!=3].shape[0]!=rsts.shape[0]:
		import ipdb; ipdb.set_trace()  # breakpoint 6783ebf8 //
	nzero = np.sum(rsts>0,1)
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
	# for s in range(0,T-l,l):
	for s in range(63000,T-l,l):
		if os.path.exists(outputdir+'reconstructions_small/'+'series_decomp_{}_{}.eps'.format(s,s+l)) and os.path.exists(outputdir+'reconstructions_small/'+'series_rec_{}_{}.eps'.format(s,s+l)):
			continue
	# for s in tqdm.tqdm(range(30000,T-l,l),'plotting'):
		lim_x1=s*time_scale
		lim_x2=(s+l)*time_scale
		fig_decomp = plt.figure(1,(8,20))
		ax_orig_recon_1 = plt.subplot2grid((4,1),(0,0))
		orig = series[s:s+l]
		recon = reconseries2[s:s+l]
		# trsts = reconstates[s/step:s/step+l/step]
		diff_std = np.std( reconseries2 - series[:reconseries2.shape[0]] )
		thisIC = IC[s:s+l]
		xdata = np.linspace(s,s+l, l)*time_scale #convert to ms
		these_lims = lims2[lims2>=s]
		these_lims = these_lims[these_lims<s+l]
		trsts = rsts[lims2>=s]
		trsts = trsts[:these_lims.shape[0]]
		trssize = rssize[lims2>=s]
		trssize = trssize[:these_lims.shape[0]]
		tmass = mass[lims2>=s]
		tmass = tmass[:these_lims.shape[0]]
		tnzero = nzero[lims2>=s]
		tnzero = tnzero[:these_lims.shape[0]]
		# xdata = np.linspace(s,s+l, l)
		# print xdata.shape,orig.shape,recon.shape
		ax_orig_recon_1.plot(xdata,orig,label='Original EC',color='blue')
		ax_orig_recon_1.plot(xdata,recon,label='Reconstruction EC',color='red')
		ax_orig_recon_1.axis([lim_x1,lim_x2,minb,maxb],fontsize=12)
		ax_orig_recon_1.tick_params(axis='both',labelsize=12)
		# ax.yticks(fontsize=16)
		handles, labels = ax_orig_recon_1.get_legend_handles_labels()
		lgd1 = ax_orig_recon_1.legend(handles,labels, loc='upper right', bbox_to_anchor=(1.,1.),fontsize=9)
		# ax_orig_recon_1.set_xlabel("ms")
		ax_orig_recon_1.set_ylabel("mV [400-4000]Hz")
		ax_orig_recon_1.set_title("A.", marker_font_dict,loc='left')
		ax_orig_recon_1.set_title(r"Original and Reconstructed Signal", fontdict=title_font_dict,loc='center')
		# ax_orig_recon_1.grid('on')

		# plt.savefig(outputdir+'reconstructions/'+'series_{}_{}.jpg'.format(s,s+l))
		# if not os.path.exists('{}montage_images/orig_{:03}.jpg'.format(outputdir, e)):
		
		inds = np.arange(N)
		inds = inds[1==((lims[:,0]>=s) * (lims[:,1]<s+l))]
		# tlims = tlims[tlims[:,1]<s+l,:]
		
		tlims = lims[inds,:]
		ltrssize = rstssize[inds]
		ax_decomp_1=plt.subplot2grid((4,1),(1,0))
		ax_decomp_2=plt.subplot2grid((4,1),(2,0))
		ax_decomp_3=plt.subplot2grid((4,1),(3,0))
		# ax_decomp_1=fig_decomp.add_subplot(4,1,2)
		# O1=300
		O2=np.max(np.abs(W_all[-1,:,:]))*3
		for ind in inds:
			h=0
			for hp,b in enumerate(rs[ind]):
				if b==0:
					continue


				width = lims[ind,0]*time_scale
				height = None

				full_xdata = np.linspace(lims[ind,0],lims[ind,1],D)*time_scale
				half_xdata1 = np.linspace(lims[ind,0],lims[ind,0]+(D/2),D/2)*time_scale
				half_xdata2 = np.linspace(lims[ind,0]+(D/2),lims[ind,1],D/2)*time_scale
				if ind%3==2:
					height=np.max( W_all[-1,:,hp] + O2*h)+2
					if rstssize[ind]==0:
						ax_decomp_1.plot(full_xdata, W_all[-1,:,hp] + O2*h,'r')
					elif rstssize[ind]==1:
						ax_decomp_1.plot(half_xdata1, W_all[-1,:D/2,hp] + O2*h,'r')
						ax_decomp_1.plot(half_xdata2, W_all[-1,D/2:,hp] + O2*h,'b')
					elif rstssize[ind]==2:
						ax_decomp_1.plot(half_xdata1, W_all[-1,:D/2,hp] + O2*h,'b')
						ax_decomp_1.plot(half_xdata2, W_all[-1,D/2:,hp] + O2*h,'r')
					elif rstssize[ind]==3:
						ax_decomp_1.plot(full_xdata, W_all[-1,:,hp] + O2*h,'b')
					ax_decomp_1.text(width,height,'{}x{}'.format(hp+1,b),fontsize=12)
				elif ind%3==1:
					height=np.max( W_all[-1,:,hp] + O2*h)+2
					# height=np.max( W_all[-1,:,hp] + O1 + O2*h)+2
					if rstssize[ind]==0:
						ax_decomp_2.plot(full_xdata, W_all[-1,:,hp] + O2*h,'r')
					elif rstssize[ind]==1:
						ax_decomp_2.plot(half_xdata1, W_all[-1,:D/2,hp] + O2*h,'r')
						ax_decomp_2.plot(half_xdata2, W_all[-1,D/2:,hp] + O2*h,'b')
					elif rstssize[ind]==2:
						ax_decomp_2.plot(half_xdata1, W_all[-1,:D/2,hp] + O2*h,'b')
						ax_decomp_2.plot(half_xdata2, W_all[-1,D/2:,hp] + O2*h,'r')
					elif rstssize[ind]==3:
						ax_decomp_2.plot(full_xdata, W_all[-1,:,hp] + O2*h,'b')
					ax_decomp_2.text(width,height,'{}x{}'.format(hp+1,b),fontsize=12)
				else:
					height=np.max( W_all[-1,:,hp] + O2*h)+2
					# height=np.max( W_all[-1,:,hp] + 2*O1 + O2*h)+2
					if rstssize[ind]==0:
						ax_decomp_3.plot(full_xdata, W_all[-1,:,hp] + O2*h,'r')
					elif rstssize[ind]==1:
						ax_decomp_3.plot(half_xdata1, W_all[-1,:D/2,hp] + O2*h,'r')
						ax_decomp_3.plot(half_xdata2, W_all[-1,D/2:,hp] + O2*h,'b')
					elif rstssize[ind]==2:
						ax_decomp_3.plot(half_xdata1, W_all[-1,:D/2,hp] + O2*h,'b')
						ax_decomp_3.plot(half_xdata2, W_all[-1,D/2:,hp] + O2*h,'r')
					elif rstssize[ind]==3:
						ax_decomp_3.plot(full_xdata, W_all[-1,:,hp] + O2*h,'b')
					ax_decomp_3.text(width,height,'{}x{}'.format(hp+1,b),fontsize=12)
				
				h+=1
		
		#make good ticks
		ticmax= np.int(np.max(np.abs(W_all[-1,:,:])))
		ticmin= -ticmax
		tloc=[]
		tlab=[]
		for g in range(gamma):
			tloc.append(ticmin+O2*g)
			tlab.append('${}$'.format(ticmin))
			tloc.append(0+O2*g)
			tlab.append('${}$'.format(0))
			tloc.append(ticmax+O2*g)
			tlab.append('${}$'.format(ticmax))

		# ax_decomp_1.bar(these_lims,tnzero,trssize)
		ax_decomp_1.axis([lim_x1,lim_x2,np.min(W_all[-1])-3,np.max(np.abs(W_all[-1]))+(gamma-1)*O2+3],fontsize=12)
		ax_decomp_1.set_yticks(tloc)
		ax_decomp_1.set_yticklabels(tlab)
		ax_decomp_1.tick_params(axis='both',labelsize=12)
		# ax_decomp_1.set_xlabel("ms")
		ax_decomp_1.set_title("B.", marker_font_dict,loc='left')
		ax_decomp_1.set_title("$n-1$", fontdict=title_font_dict,loc='center')
		# ax_decomp_1.set_ylabel("mV [400-4000]Hz")
		
		ax_decomp_2.axis([lim_x1,lim_x2,np.min(W_all[-1])-3,np.max(np.abs(W_all[-1]))+(gamma-1)*O2+3],fontsize=12)
		ax_decomp_2.set_yticks(tloc)
		ax_decomp_2.set_yticklabels(tlab)
		ax_decomp_2.tick_params(axis='both',labelsize=12)
		# ax_decomp_2.set_xlabel("ms")
		ax_decomp_2.set_title("C.", marker_font_dict,loc='left')
		ax_decomp_2.set_title("$n$", fontdict=title_font_dict,loc='center')
		# ax_decomp_2.set_ylabel("mV [400-4000]Hz")
		
		ax_decomp_3.axis([lim_x1,lim_x2,np.min(W_all[-1])-3,np.max(np.abs(W_all[-1]))+(gamma-1)*O2+3],fontsize=12)
		ax_decomp_3.set_yticks(tloc)
		ax_decomp_3.set_yticklabels(tlab)
		ax_decomp_3.tick_params(axis='both',labelsize=12)
		ax_decomp_3.set_xlabel("ms")
		ax_decomp_3.set_title("D.", marker_font_dict,loc='left')
		ax_decomp_3.set_title("$n+1$", fontdict=title_font_dict,loc='center')
		# ax_decomp_3.set_ylabel("mV [400-4000]Hz")


		# ax_IC = fig_decomp.add_subplot(4,1,3)
		for i in range(these_lims.shape[0]):
			ax_orig_recon_1.axvline(x=these_lims[i]*time_scale,ymin=0,ymax=1,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')
			ax_decomp_1.axvline(x=these_lims[i]*time_scale,ymin=0,ymax=1,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')
			ax_decomp_2.axvline(x=these_lims[i]*time_scale,ymin=0,ymax=1,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')
			ax_decomp_3.axvline(x=these_lims[i]*time_scale,ymin=0,ymax=1,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')
		# fig_decomp.tight_layout(rect=[0,0,1,0.9])
		# fig_decomp.savefig(outputdir+'reconstructions/'+'small_series_{}_{}.jpg'.format(s,s+l), bbox_extra_artists=(lgd1,), bbox_inches = 'tight')
		# sup1=fig_decomp.suptitle(r"\textbf{EC signal discretization}")
		fig_decomp.savefig(outputdir+'reconstructions_small/'+'series_decomp_{}_{}.eps'.format(s,s+l), bbox_extra_artists=(lgd1,), bbox_inches = 'tight',dpi=600)
		# fig_decomp.savefig(outputdir+'reconstructions/'+'small_series_{}_{}_n.jpg'.format(s,s+l), bbox_extra_artists=(lgd1,), bbox_inches = 'tight')
		# fig_decomp.savefig(outputdir+'reconstructions/'+'series_{}_{}_n.jpg'.format(s,s+l), bbox_extra_artists=(lgd1,lgd2), bbox_inches = 'tight')
		plt.close(fig_decomp)

		c+=1
	state_list=np.array(state_list)
	# fig_decomp = plt.figure()
	# ax = fig_decomp.add_subplot(111)
	# ax.
plt.clf()
plt.plot(sigma_all,label='$\sigma_{model}$')
plt.plot(np.arange(sigma_all.shape[0]),std_series*np.ones_like(sigma_all),label='$\sigma_{orig}$',linestyle='dashed')
print(std_series)
plt.axis([0,200,10,22])
plt.legend()
plt.savefig(outputdir+'sigma.jpg')
plt.clf()
print "state_list holds the states of interest"
