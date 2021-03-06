#
#  Lincense: Academic Free License (AFL) v3.0
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)
FONTSIZE=24
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
if not os.path.exists(outputdir+'reconstructions2'):
	os.mkdir(outputdir+'reconstructions2')
if not os.path.exists(outputdir+'filters'):
	os.mkdir(outputdir+'filters')

for e in range(0,epochs,10)[::-1]:
	# minwg = -np.max(np.abs(W_all[e]))
	# maxwg = -minwg
	minwg = np.min(W_all[e])
	maxwg = np.max(W_all[e])
	meanw = np.mean(W_all)
	if not os.path.exists('{}filters/W_e_{}.eps'.format(outputdir, e)):
	# if not os.path.exists('{}montage_images/W_e_{:03}.eps'.format(outputdir, e)):
		fig=plt.figure(2,(16,10))
		for h in range(H):
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
			ax = fig.add_subplot(5,8,h+1)
			ax.plot(np.linspace(0,D/10,num=D) ,this_W)#scale in kHz
			ax.axis([0,D/10,minwg,maxwg],fontsize=16)
			# ax.savefig(outputdir+'_images/'+'W_e_{:03}_h_{:03}.jpg'.format(e,h))
			ax.set_title("$W_{"+str(h+1)+"}$",fontsize=16)
			ax.tick_params(axis='both',labelsize=12)
			ax.axis('off')
			# ax.clf()
			if h%30 == 0 :
				print ("Finished epoch {:03} basis {:03}".format(e,h))
				print ("\tPlot settings scale: '{}', min: {}, max: {}, mean:{}".format(cscale,minwg,maxwg,meanw))
				# ax.clf()
		plt.tight_layout()
		fig.savefig(outputdir+'filters/W_e_{}.eps'.format(e), bbox_inches = 'tight',dpi=600)
		plt.close(fig)
		# os.system("montage -trim {}_images/W_e_{:03}_h*.jpg {}montage_images/W_e_{:03}.jpg".format(outputdir,e,outputdir,e))
		# os.system("rm {}_images/W_e_{:03}_h*.jpg ".format(outputdir, e))
# os.system("convert -delay 10 {}montage_images/* {}W_training.gif".format(outputdir,outputdir))
#
minwg = np.min(W_all[-1])
maxwg = np.max(W_all[-1])
meanw = np.mean(W_all)
e=-1
for h in range(H):
	fig=plt.figure(2)
	if os.path.exists(outputdir+'_images/'+'W_e_{:03}_h_{:03}.eps'.format(-1,h+1)):
		continue
	this_W = W_all[-1,:,h]
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
	ax = fig.add_subplot(111)
	ax.plot(np.linspace(0,D/10,num=D) ,this_W)#scale in kHz
	ax.axis([0,D/10,minwg,maxwg])
	# ax.savefig(outputdir+'_images/'+'W_e_{:03}_h_{:03}.jpg'.format(e,h))
	ax.set_title("$W_{"+str(h+1)+"}$")
	ax.tick_params(axis='both')

	# ax.clf()
	if h%30 == 0 :
		print ("Finished epoch {:03} basis {:03}".format(e,h+1))
		print ("\tPlot settings scale: '{}', min: {}, max: {}, mean:{}".format(cscale,minwg,maxwg,meanw))
		# ax.clf()
	plt.tight_layout()
	fig.savefig(outputdir+'_images/W_e_{:03}_h_{:03}.eps'.format(200,h+1), bbox_inches = 'tight',dpi=600)
	plt.close(fig)


for e in range(epochs)[::-1]:
	print(e)

	if not os.path.exists('{}montage_images/pi_broken_{:03}.jpg'.format(outputdir, e)):
		f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
		ax.bar(states,pi_all[e,:],align='center')
		ax2.bar(states,pi_all[e,:],align='center')
		yl = pi_all[e,np.argsort(pi_all[e,:])[:-1]].sum()*1.25
		ax.set_ylim(1.-yl ,1.)
		ax2.set_ylim(0,yl)
		# hide the spines between ax and ax2
		ax.spines['bottom'].set_visible(False)
		ax2.spines['top'].set_visible(False)
		ax.xaxis.tick_top()
		ax.tick_params(labeltop='off')  # don't put tick labels at the top
		ax2.xaxis.tick_bottom()

		d = 0.015 # how big to make the diagonal lines in axes coordinates
		# d = pi_all[e,np.argsort(pi_all[e,:])[1]]*1.2  # how big to make the diagonal lines in axes coordinates
		# arguments to pass plot, just so we don't keep repeating them
		kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
		ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
		ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

		kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
		ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
		ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

		# We change the fontsize of minor ticks label
		ax.tick_params(axis='both', which='major', labelsize=20)
		ax.tick_params(axis='both', which='minor', labelsize=20)
		ax2.set_xticks(states)
		ax2.tick_params(axis='both', which='major', labelsize=20)
		ax2.tick_params(axis='both', which='minor', labelsize=20)
		ax2.tick_params(axis='x', which='major', length=1)
		ax2.tick_params(axis='x', which='minor', length=1)
		# ax2.tick_params(axis='x', which='major', labelsize=20, length=1)
		# ax2.tick_params(axis='x', which='minor', labelsize=20, length=1)
		f.savefig(outputdir+'montage_images/pi_broken_{:03}.jpg'.format(e))
		plt.close(f)

# plt.tick_params(axis='both',labelsize=16)
# plt.clf()


state_list = []
std_series=None
print (series)
print (rseries)
title_font_dict = {"fontsize":"12"}
marker_font_dict = {"fontsize":"16"}
title_font_dict = None # {"fontsize":"12"}
marker_font_dict = None #{"fontsize":"16"}
time_scale = 1000/sampling_rate
if series is not None and rseries is not None:
	IC = series.squeeze()[5,:].squeeze()
	series = series.squeeze()[channel,:].squeeze()
	rseries = rseries.squeeze()
	T = rseries.shape[0]
	l = 500
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


	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot(sigma_all,label='$\sigma_{model}$')
	ax.plot(np.arange(sigma_all.shape[0]),std_series*np.ones_like(sigma_all),label='$\sigma_{orig}$',linestyle='dashed')
	ax.tick_params(axis='both',labelsize=20)

	# print(std_series)
	ax.axis([0,200,10,22])
	ax.legend()
	fig.savefig(outputdir+'sigma.jpg')

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
	for s in range(0,T-l,l):
		if os.path.exists(outputdir+'reconstructions2/'+'series_{}_{}_decomposition.eps'.format(s,s+l)) and os.path.exists(outputdir+'reconstructions2/'+'series_{}_{}_rec.eps'.format(s,s+l)):
			continue
	# for s in tqdm.tqdm(range(30000,T-l,l),'plotting'):
		lim_x1=s*time_scale
		lim_x2=(s+l)*time_scale
		fig_decomp = plt.figure(1,(12,20))
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
		ax_orig_recon_1.plot(xdata,orig,label='Original EC',color='black')
		ax_orig_recon_1.plot(xdata,recon,label='Reconstruction EC',color='lime')
		ax_orig_recon_1.axis([lim_x1,lim_x2,minb,maxb],fontsize=12)
		ax_orig_recon_1.tick_params(axis='both',labelsize=12)
		# ax.yticks(fontsize=16)
		handles, labels = ax_orig_recon_1.get_legend_handles_labels()
		lgd1 = ax_orig_recon_1.legend(handles,labels, loc='upper right', bbox_to_anchor=(1.,1.),fontsize=9)
		# ax_orig_recon_1.set_xlabel("ms")
		ax_orig_recon_1.set_ylabel("mV [400-4000]Hz",fontsize=FONTSIZE)
		ax_orig_recon_1.set_title("A.", marker_font_dict,loc='left',fontsize=FONTSIZE)
		ax_orig_recon_1.set_title(r"Original and Reconstructed Signal",fontsize=FONTSIZE, fontdict=title_font_dict,loc='center')
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
						ax_decomp_1.plot(full_xdata, W_all[-1,:,hp] + O2*h,'lime')
					elif rstssize[ind]==1:
						ax_decomp_1.plot(half_xdata1, W_all[-1,:D//2,hp] + O2*h,'lime')
						ax_decomp_1.plot(half_xdata2, W_all[-1,D//2:,hp] + O2*h,'r')
					elif rstssize[ind]==2:
						ax_decomp_1.plot(half_xdata1, W_all[-1,:D//2,hp] + O2*h,'r')
						ax_decomp_1.plot(half_xdata2, W_all[-1,D//2:,hp] + O2*h,'lime')
					elif rstssize[ind]==3:
						ax_decomp_1.plot(full_xdata, W_all[-1,:,hp] + O2*h,'r')
					ax_decomp_1.text(width,height,'{}x{}'.format(hp+1,b),fontsize=8)
				elif ind%3==1:
					height=np.max( W_all[-1,:,hp] + O2*h)+2
					# height=np.max( W_all[-1,:,hp] + O1 + O2*h)+2
					if rstssize[ind]==0:
						ax_decomp_2.plot(full_xdata, W_all[-1,:,hp] + O2*h,'lime')
					elif rstssize[ind]==1:
						ax_decomp_2.plot(half_xdata1, W_all[-1,:D//2,hp] + O2*h,'lime')
						ax_decomp_2.plot(half_xdata2, W_all[-1,D//2:,hp] + O2*h,'r')
					elif rstssize[ind]==2:
						ax_decomp_2.plot(half_xdata1, W_all[-1,:D//2,hp] + O2*h,'r')
						ax_decomp_2.plot(half_xdata2, W_all[-1,D//2:,hp] + O2*h,'lime')
					elif rstssize[ind]==3:
						ax_decomp_2.plot(full_xdata, W_all[-1,:,hp] + O2*h,'r')
					ax_decomp_2.text(width,height,'{}x{}'.format(hp+1,b),fontsize=8)
				else:
					height=np.max( W_all[-1,:,hp] + O2*h)+2
					# height=np.max( W_all[-1,:,hp] + 2*O1 + O2*h)+2
					if rstssize[ind]==0:
						ax_decomp_3.plot(full_xdata, W_all[-1,:,hp] + O2*h,'lime')
					elif rstssize[ind]==1:
						ax_decomp_3.plot(half_xdata1, W_all[-1,:D//2,hp] + O2*h,'lime')
						ax_decomp_3.plot(half_xdata2, W_all[-1,D//2:,hp] + O2*h,'r')
					elif rstssize[ind]==2:
						ax_decomp_3.plot(half_xdata1, W_all[-1,:D//2,hp] + O2*h,'r')
						ax_decomp_3.plot(half_xdata2, W_all[-1,D//2:,hp] + O2*h,'lime')
					elif rstssize[ind]==3:
						ax_decomp_3.plot(full_xdata, W_all[-1,:,hp] + O2*h,'r')
					ax_decomp_3.text(width,height,'{}x{}'.format(hp+1,b),fontsize=8)

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
		ax_decomp_1.set_title("B.", marker_font_dict, fontsize=FONTSIZE,loc='left')
		ax_decomp_1.set_title("$t-1$",fontsize=FONTSIZE, fontdict=title_font_dict, loc='center')
		# ax_decomp_1.set_ylabel("mV [400-4000]Hz")

		ax_decomp_2.axis([lim_x1,lim_x2,np.min(W_all[-1])-3,np.max(np.abs(W_all[-1]))+(gamma-1)*O2+3],fontsize=12)
		ax_decomp_2.set_yticks(tloc)
		ax_decomp_2.set_yticklabels(tlab)
		ax_decomp_2.tick_params(axis='both',labelsize=12)
		# ax_decomp_2.set_xlabel("ms")
		ax_decomp_2.set_title("C.", marker_font_dict, fontsize=FONTSIZE,loc='left')
		ax_decomp_2.set_title("$t$",fontsize=FONTSIZE, fontdict=title_font_dict,loc='center')
		# ax_decomp_2.set_ylabel("mV [400-4000]Hz")

		ax_decomp_3.axis([lim_x1,lim_x2,np.min(W_all[-1])-3,np.max(np.abs(W_all[-1]))+(gamma-1)*O2+3],fontsize=12)
		ax_decomp_3.set_yticks(tloc)
		ax_decomp_3.set_yticklabels(tlab)
		ax_decomp_3.tick_params(axis='both',labelsize=12)
		ax_decomp_3.set_xlabel("ms",fontsize=FONTSIZE)
		ax_decomp_3.set_title("D.", marker_font_dict, fontsize=FONTSIZE, loc='left')
		ax_decomp_3.set_title("$t+1$",fontsize=FONTSIZE, fontdict=title_font_dict,loc='center')
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
		fig_decomp.savefig(outputdir+'reconstructions2/'+'series_{}_{}_decomposition.eps'.format(s,s+l), bbox_extra_artists=(lgd1,), bbox_inches = 'tight',dpi=600)
		# fig_decomp.savefig(outputdir+'reconstructions/'+'small_series_{}_{}_n.jpg'.format(s,s+l), bbox_extra_artists=(lgd1,), bbox_inches = 'tight')
		# fig_decomp.savefig(outputdir+'reconstructions/'+'series_{}_{}_n.jpg'.format(s,s+l), bbox_extra_artists=(lgd1,lgd2), bbox_inches = 'tight')
		plt.close(fig_decomp)
		fig_rec = plt.figure(1,(12,20))


		ax_orig_recon_2 = plt.subplot2grid((4,1),(0,0))
		ax_orig_recon_2.plot(xdata,orig,label='Original EC',color='black')
		ax_orig_recon_2.plot(xdata,recon,label='Reconstruction EC',linestyle='dashed',color='lime')
		ax_orig_recon_2.axis([lim_x1,lim_x2,minb,maxb],fontsize=12)
		ax_orig_recon_2.tick_params(axis='both',labelsize=12)
		ax_orig_recon_2.plot(xdata,5*std_series*np.ones_like(xdata),linestyle='dashed',label='$5 \\times \sigma_{orig}$',color='black')
		ax_orig_recon_2.plot(xdata,-5*std_series*np.ones_like(xdata),linestyle='dashed',color='black')
		# ax.yticks(fontsize=16)
		ax_orig_recon_2.plot(xdata,5*sigma_all[-1]*np.ones_like(xdata),linestyle='dashdot',label='$5 \\times \sigma_{model}$',color='lime')
		ax_orig_recon_2.plot(xdata,-5*sigma_all[-1]*np.ones_like(xdata),linestyle='dashdot',color='lime')
		# ax.yticks(fontsize=16)
		handles, labels = ax_orig_recon_2.get_legend_handles_labels()
		lgd1 = ax_orig_recon_2.legend(handles,labels, loc='upper right', bbox_to_anchor=(1.,1.),fontsize=9)
		# ax_orig_recon_2.set_xlabel("ms")
		ax_orig_recon_2.set_ylabel(r"mV [400-4000]Hz",fontsize=FONTSIZE)
		ax_orig_recon_2.set_title(r"A.", marker_font_dict,loc='left',fontsize=FONTSIZE)
		ax_orig_recon_2.set_title(r"Original and Reconstructed signal", fontsize=FONTSIZE,fontdict=title_font_dict,loc='center')
		# ax_orig_recon_1.grid('on')

		ax_diff = plt.subplot2grid((4,1),(1,0))
		ax_diff.plot(xdata,recon-orig,label='difference',color='red')
		#std
		#
		# ax_diff.plot(xdata,diff_std*np.ones_like(xdata),label='std of difference',color='black')
		# ax_diff.plot(xdata,-diff_std*np.ones_like(xdata),color='black')
		# ax_diff.plot(xdata,std_series*np.ones_like(xdata),label='std of original',color='red')
		# ax_diff.plot(xdata,-std_series*np.ones_like(xdata),color='red')
		# ax_diff.plot(xdata,sigma_all[-1]*np.ones_like(xdata),label='std of model',color='blue')
		# ax_diff.plot(xdata,-sigma_all[-1]*np.ones_like(xdata),color='blue')
		#3 x std
		# ax_diff.plot(xdata,3*diff_std*np.ones_like(xdata),linestyle='dashed',label='std of difference',color='black')
		# ax_diff.plot(xdata,-3*diff_std*np.ones_like(xdata),linestyle='dashed',color='black')
		# ax_diff.plot(xdata,3*std_series*np.ones_like(xdata),linestyle='dashed',label='$3 \\times std$ of original',color='red')
		# ax_diff.plot(xdata,-3*std_series*np.ones_like(xdata),linestyle='dashed',color='red')
		ax_diff.plot(xdata,5*std_series*np.ones_like(xdata),linestyle='dashed',label='$5 \\times \sigma_{model}$',color='black')
		ax_diff.plot(xdata,-5*std_series*np.ones_like(xdata),linestyle='dashed',color='black')
		ax_diff.plot(xdata,sigma_all[-1]*np.ones_like(xdata),linestyle='dashdot',label='$\sigma_{model}$',color='lime')
		ax_diff.plot(xdata,-sigma_all[-1]*np.ones_like(xdata),linestyle='dashdot',color='lime')
		ax_diff.axis([lim_x1,lim_x2,minb,maxb],fontsize=12)
		ax_diff.tick_params(axis='both',labelsize=12)
		ax_diff.set_title("B.", marker_font_dict, fontsize=FONTSIZE,loc='left')
		ax_diff.set_title("Difference EC-REC", fontsize=FONTSIZE, fontdict=title_font_dict,loc='center')
		# ax_diff.set_xlabel("ms")
		# ax_diff.set_ylabel("mV [400-4000]Hz")
		# ax.yticks(fontsize=16)
		handles, labels = ax_diff.get_legend_handles_labels()
		lgd2 = ax_diff.legend(handles,labels, loc='upper right', bbox_to_anchor=(1.,1.),fontsize=9)
		# ax_orig_recon_1.grid('on')



		ax_IC = plt.subplot2grid((4,1),(2,0))
		ax_IC.plot(xdata,thisIC,label='IC')
		thisICg200 = np.argwhere(thisIC>200)
		# print thisICg200,thisIC[thisICg200]
		peaks = findpeaks(thisIC[thisICg200],thisICg200)
		if peaks.shape[0]>0:
			ploc = peaks
			for p in ploc:
				# print (p[0]-25.)/l,0.01,50./l,0.99
				# pat = patches.Rectangle((s+p[0]-25,minic),50.,4*(maxic-minic),fill=True,alpha=0.3,color='red')
				left = these_lims[these_lims<=s+p[0]]#[-2]
				right = these_lims[these_lims>=s+p[0]]
				# state_list.append(trsts[these_lims<=s+p[0]][-2])
				if p[0]-20<s:
					state_list.append(trsts[0])
				else:
					state_list.append(trsts[these_lims<=s+p[0]][-1])
				# state_list.append(trsts[these_lims>=s+p[0]][0])
				# state_list.append(trsts[these_lims>=s+p[0]][1])
				if right.shape[0]>1:
					right = right[1]
				else:
					right = s+l
				if left.shape[0]>1:
					left = left[-2]
				else:
					left = s
				left*=time_scale
				right*=time_scale
				pat = patches.Rectangle((left,minic),right-left,4*(maxic-minic),fill=True,alpha=0.3,color='red')
				ax_IC.add_patch(pat)
				# ax.axvspan(p[0]-25,p[0]+25,color='red',alpha=0.5)

		ax_IC.axis([lim_x1,lim_x2,minic,maxic],fontsize=12)
		handles, labels = ax_IC.get_legend_handles_labels()
		lgd2 = ax_IC.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.,1.),fontsize=9)
		ax_IC.tick_params(axis='both',labelsize=12)
		# ax_IC.set_xlabel("ms")
		ax_IC.set_ylabel("mV",fontsize=FONTSIZE)
		ax_IC.set_title("C.", marker_font_dict, fontsize=FONTSIZE,loc='left')
		ax_IC.set_title("Time-aligned intra-cellular recording (IC)", fontsize=FONTSIZE, fontdict=title_font_dict,loc='center')
		# ax_IC.grid('on')
		ax_mass=plt.subplot2grid((4,1),(3,0))
		ax_mass.bar(these_lims*time_scale,tmass,trssize*time_scale)
		ax_mass.axis([lim_x1,lim_x2,0,gamma*np.max(states)*np.max(np.mean(np.abs(W_all[-1]),0))],fontsize=12)
		ax_mass.tick_params(axis='both',labelsize=12)
		ax_mass.set_xlabel("ms",fontsize=FONTSIZE)
		ax_mass.set_title("D.", marker_font_dict, fontsize=FONTSIZE, loc='left')
		ax_mass.set_title("Model Spike Mass", fontsize=FONTSIZE, fontdict=title_font_dict,loc='center')



		# for i in range(these_lims.shape[0]):
			# ax_orig_recon_2.axvline(x=these_lims[i],ymin=0,ymax=1,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')
			# ax_decomp_1.axvline(x=these_lims[i],ymin=-0.2,ymax=1.2,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')
			# ax_IC.axvline(x=these_lims[i],ymin=0,ymax=1,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')
			# ax_diff.axvline(x=these_lims[i],ymin=0,ymax=1,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')

		# sup1=fig_rec.suptitle(r"\textbf{Accuracy of the reconstruction}")
		fig_rec.savefig(outputdir+'reconstructions2/'+'series_{}_{}_rec.eps'.format(s,s+l), bbox_extra_artists=(lgd1,), bbox_inches = 'tight',dpi=600)
		plt.close(fig_decomp)
		plt.close(fig_rec)


		c+=1
	state_list=np.array(state_list)
	# fig_decomp = plt.figure()
	# ax = fig_decomp.add_subplot(111)
	# ax.

# print("state_list holds the states of interest")
