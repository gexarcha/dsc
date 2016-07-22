import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rc('text', usetex=True)
import tables as tb
import os
import tqdm

MEAN_HIDDEN=False
outputdir = 'output/'+'dsc_on_hc1_run.py.2016-03-17+11:39/'
outputdir = 'output/'+'dsc_on_hc1_run.py.2016-03-17+21:16/'
outputdir = 'output/'+'dsc_on_hc1_run.py.2016-03-23+01:56/'
outputdir = 'output/'+'dsc_on_hc1_run.py.2016-03-23+13:05/'
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
states=np.array([-2,-1.,0.,1.,2.])
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
if not os.path.exists(outputdir+'filters'):
        os.mkdir(outputdir+'filters')

for e in range(0,epochs)[::-1]:
    # minwg = -np.max(np.abs(W_all[e])) 
    # maxwg = -minwg
    minwg = np.min(W_all[e]) 
    maxwg = np.max(W_all[e]) 
    meanw = np.mean(W_all) 
    if not os.path.exists('{}filters/W_e_{:03}.eps'.format(outputdir, e)):
    # if not os.path.exists('{}montage_images/W_e_{:03}.eps'.format(outputdir, e)):
        fig=plt.figure(2,(20,30))
        # fig = plt.figure(2,figsize=(30,10))
        for h in range(H):
            if os.path.exists(outputdir+'_images/'+'W_e_{:03}_h_{:03}.eps'.format(e,h)):
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
            ax = fig.add_subplot(13,8,h+1)
            ax.plot(np.linspace(0,D/10,num=D) ,this_W)#scale in kHz
            ax.axis([0,D/10,minwg,maxwg],fontsize=12)
            # ax.savefig(outputdir+'_images/'+'W_e_{:03}_h_{:03}.jpg'.format(e,h))
            ax.set_title("$W_{"+str(h+1)+"}$",fontsize=12)
            ax.tick_params(axis='both',labelsize=12)

            # ax.clf()
            if h%30 == 0 :
                print "Finished epoch {:03} basis {:03}".format(e,h)
                print "\tPlot settings scale: '{}', min: {}, max: {}, mean:{}".format(cscale,minwg,maxwg,meanw)
                # ax.clf()
        plt.tight_layout()
        fig.savefig(outputdir+'filters/W_e_{:03}.eps'.format(e), bbox_inches = 'tight',dpi=600)
        plt.close(fig)
        # os.system("montage -trim {}_images/W_e_{:03}_h*.jpg {}montage_images/W_e_{:03}.jpg".format(outputdir,e,outputdir,e))
        # os.system("rm {}_images/W_e_{:03}_h*.jpg ".format(outputdir, e))
# os.system("convert -delay 10 {}montage_images/* {}W_training.gif".format(outputdir,outputdir))
state_list = []
if series is not None and rseries is not None:
    #IC = series.squeeze()[5,:].squeeze()
    series = series.squeeze()#[channel,:].squeeze()
    step = int((1-overlap)*D)
    tmp2=series[:,:step].reshape(-1)
    tmp2=np.concatenate([tmp2,series[-1,D-step:]])
    series=np.array(tmp2)

    # # Mean Hidden state
    # if MEAN_HIDDEN:
    #     rs = np.mean(inf_poster[:,:,None]*inf_states,1)
    #     ry = np.dot(rs,W_all[-1,:,:].T)
    ry = np.dot(inf_states[:,0,:],W_all[-1,:,:].T)

    rseries = rseries.squeeze()
    T = rseries.shape[0]
    l = 1000
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
    assert overlap<=0.5
    s=0
    lims2 = [0]
    lims3 = [lims[0]]
    rsts = [rs[0]]
    rstssize = [0] # 0 means full, 1 means first half, 2 means second half, 3 means missing 
    # import ipdb; ipdb.set_trace() 
    for n in tqdm.tqdm(range(1,N),'reconstructing'):
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
    #minic = np.min(IC)
    #maxic = np.max(IC)
    minb = np.minimum(mins,minrs)
    maxb = np.maximum(maxs,maxrs)
    
    from scipy.io.wavfile import write
    scaled_series = np.int16(series/np.max(np.abs(series))*32767)
    scaled_rseries = np.int16(reconseries2/np.max(np.abs(reconseries2))*32767)
    write(outputdir+'original.wav',16000,scaled_series)
    write(outputdir+'reconstruction.wav',16000,scaled_rseries)
    print "wrote audio files"
    c=0
    for s in tqdm.tqdm(range(0,T-l,l),'plotting'):
    # for s in tqdm.tqdm(range(30000,T-l,l),'plotting'):
        fig = plt.figure(1,(12,20))
        ax1 = plt.subplot2grid((5,1),(0,0),rowspan=2)
        # ax1 = fig.add_subplot(4,1,1)
        # ax1 = fig.add_subplot(4,1,1)
        orig = series[s:s+l]
        recon = reconseries2[s:s+l]
        # trsts = reconstates[s/step:s/step+l/step]
        #thisIC = IC[s:s+l]
        xdata = np.linspace(s,s+l, l)
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

        ax1.plot(xdata,orig,label='Original EC',color='blue')
        ax1.plot(xdata,recon,label='Reconstruction EC',color='red')
        ax1.axis([s,s+l,minb,maxb],fontsize=12)
        ax1.tick_params(axis='both',labelsize=12)
        # ax.yticks(fontsize=16)
        handles, labels = ax1.get_legend_handles_labels()
        lgd1 = ax1.legend(handles,labels, loc='upper right', bbox_to_anchor=(1.,1.),fontsize=9)
        # ax1.grid('on')

        # plt.savefig(outputdir+'reconstructions/'+'series_{}_{}.jpg'.format(s,s+l))
        # if not os.path.exists('{}montage_images/orig_{:03}.jpg'.format(outputdir, e)):
        
        inds = np.arange(N)
        inds = inds[1==((lims[:,0]>=s) * (lims[:,1]<s+l))]
        # tlims = tlims[tlims[:,1]<s+l,:]
        
        tlims = lims[inds,:]
        ltrssize = rstssize[inds]
        ax2=plt.subplot2grid((5,1),(2,0))
        ax3=plt.subplot2grid((5,1),(3,0))
        ax4=plt.subplot2grid((5,1),(4,0))
        # ax2=fig.add_subplot(4,1,2)
        # O1=300
        O2=np.max(np.abs(W_all[-1,:,:]))*3
        hmax=-1
        Wstd = np.std(W_all[-1,:,:])
        for ind in inds:
            hs=-1
            h=0
            for b in rs[ind]:
                hs+=1
                if b==0:
                    continue


                width = lims[ind,0]+2
                height = None


                if ind%3==2:
                    height=np.max( W_all[-1,:,b] + O2*h)+2*Wstd
                    if rstssize[ind]==0:
                        ax2.plot(np.linspace(lims[ind,0],lims[ind,1],D), W_all[-1,:,b] + O2*h,'r')
                    elif rstssize[ind]==1:
                        ax2.plot(np.linspace(lims[ind,0],lims[ind,0]+(D/2),D/2), W_all[-1,:D/2,b] + O2*h,'r')
                        ax2.plot(np.linspace(lims[ind,0]+(D/2),lims[ind,1],D/2), W_all[-1,D/2:,b] + O2*h,'b')
                    elif rstssize[ind]==2:
                        ax2.plot(np.linspace(lims[ind,0],lims[ind,0]+(D/2),D/2), W_all[-1,:D/2,b] + O2*h,'b')
                        ax2.plot(np.linspace(lims[ind,0]+(D/2),lims[ind,1],D/2), W_all[-1,D/2:,b] + O2*h,'r')
                    elif rstssize[ind]==3:
                        ax2.plot(np.linspace(lims[ind,0],lims[ind,1],D), W_all[-1,:,b] + O2*h,'b')
                    ax2.text(width,height,'{}x{}'.format(hs,b),fontsize=6)
                elif ind%3==1:
                    height=np.max( W_all[-1,:,b] + O2*h)+2*Wstd
                    # height=np.max( W_all[-1,:,b] + O1 + O2*h)+2*Wstd
                    if rstssize[ind]==0:
                        ax3.plot(np.linspace(lims[ind,0],lims[ind,1],D), W_all[-1,:,b] + O2*h,'r')
                    elif rstssize[ind]==1:
                        ax3.plot(np.linspace(lims[ind,0],lims[ind,0]+(D/2),D/2), W_all[-1,:D/2,b] + O2*h,'r')
                        ax3.plot(np.linspace(lims[ind,0]+(D/2),lims[ind,1],D/2), W_all[-1,D/2:,b] + O2*h,'b')
                    elif rstssize[ind]==2:
                        ax3.plot(np.linspace(lims[ind,0],lims[ind,0]+(D/2),D/2), W_all[-1,:D/2,b] + O2*h,'b')
                        ax3.plot(np.linspace(lims[ind,0]+(D/2),lims[ind,1],D/2), W_all[-1,D/2:,b] + O2*h,'r')
                    elif rstssize[ind]==3:
                        ax3.plot(np.linspace(lims[ind,0],lims[ind,1],D), W_all[-1,:,b] + O2*h,'b')
                    ax3.text(width,height,'{}x{}'.format(hs,b),fontsize=6)
                else:
                    height=np.max( W_all[-1,:,b] + O2*h)+2*Wstd
                    # height=np.max( W_all[-1,:,b] + 2*O1 + O2*h)+2*Wstd
                    if rstssize[ind]==0:
                        ax4.plot(np.linspace(lims[ind,0],lims[ind,1],D), W_all[-1,:,b] + O2*h,'r')
                    elif rstssize[ind]==1:
                        ax4.plot(np.linspace(lims[ind,0],lims[ind,0]+(D/2),D/2), W_all[-1,:D/2,b] + O2*h,'r')
                        ax4.plot(np.linspace(lims[ind,0]+(D/2),lims[ind,1],D/2), W_all[-1,D/2:,b] + O2*h,'b')
                    elif rstssize[ind]==2:
                        ax4.plot(np.linspace(lims[ind,0],lims[ind,0]+(D/2),D/2), W_all[-1,:D/2,b] + O2*h,'b')
                        ax4.plot(np.linspace(lims[ind,0]+(D/2),lims[ind,1],D/2), W_all[-1,D/2:,b] + O2*h,'r')
                    elif rstssize[ind]==3:
                        ax4.plot(np.linspace(lims[ind,0],lims[ind,1],D), W_all[-1,:,b] + O2*h,'b')
                    ax4.text(width,height,'{}x{}'.format(hs,b),fontsize=6)
            
                h+=1
                if h>hmax:
                    hmax=h
            
            #make good ticks
            ticmax= np.int(np.max(np.abs(W_all[-1,:,:])))
            ticmin= -ticmax
            tloc=[]
            tlab=[]
            for g in range(hmax):
                tloc.append(ticmin+O2*g)
                tlab.append(ticmin)
                tloc.append(0+O2*g)
                tlab.append(0)
                tloc.append(ticmax+O2*g)
                tlab.append(ticmax)

            # ax2.bar(these_lims,tnzero,trssize)
            ax2.axis([s,s+l,np.min(W_all[-1])-3*Wstd,np.max(np.abs(W_all[-1]))+(hmax-1)*O2+3*Wstd],fontsize=120)
            ax2.set_yticks(tloc)
            ax2.set_yticklabels(tlab)
            ax2.tick_params(axis='both',labelsize=12)
            ax3.axis([s,s+l,np.min(W_all[-1])-3*Wstd,np.max(np.abs(W_all[-1]))+(hmax-1)*O2+3*Wstd],fontsize=120)
            ax3.set_yticks(tloc)
            ax3.set_yticklabels(tlab)
            ax3.tick_params(axis='both',labelsize=12)
            ax4.axis([s,s+l,np.min(W_all[-1])-3*Wstd,np.max(np.abs(W_all[-1]))+(hmax-1)*O2+3*Wstd],fontsize=120)
            ax4.set_yticks(tloc)
            ax4.set_yticklabels(tlab)
            ax4.tick_params(axis='both',labelsize=12)


            # ax6=plt.subplot2grid((7,1),(5,0),rowspan=2)
            # ax6.bar(these_lims,tmass,trssize)
            # ax6.axis([s,s+l,0,hmax*np.max(states)*np.max(np.mean(np.abs(W_all[-1]),0))],fontsize=12)
            # ax6.tick_params(axis='both',labelsize=12)



            for i in range(these_lims.shape[0]):
                ax1.axvline(x=these_lims[i],ymin=0,ymax=1,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')
                # ax2.axvline(x=these_lims[i],ymin=-0.2,ymax=1.2,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')
                ax2.axvline(x=these_lims[i],ymin=0,ymax=1,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')
                ax3.axvline(x=these_lims[i],ymin=0,ymax=1,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')
                ax4.axvline(x=these_lims[i],ymin=0,ymax=1,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')
                # ax5.axvline(x=these_lims[i],ymin=0,ymax=1,c="green",linewidth=.5,zorder=0, clip_on=False,ls='dotted')
            fig.tight_layout()
            # fig.savefig(outputdir+'reconstructions/'+'small_series_{}_{}.jpg'.format(s,s+l), bbox_extra_artists=(lgd1,), bbox_inches = 'tight')
            fig.savefig(outputdir+'reconstructions/'+'small_series_{}_{}.eps'.format(s,s+l), bbox_extra_artists=(lgd1,), bbox_inches = 'tight',dpi=600)
            # fig.savefig(outputdir+'reconstructions/'+'small_series_{}_{}_n.jpg'.format(s,s+l), bbox_extra_artists=(lgd1,), bbox_inches = 'tight')
            # fig.savefig(outputdir+'reconstructions/'+'series_{}_{}_n.jpg'.format(s,s+l), bbox_extra_artists=(lgd1,lgd2), bbox_inches = 'tight')
            plt.close(fig)


            c+=1
    state_list=np.array(state_list)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.
plt.clf()
plt.plot(sigma_all,label='$\sigma$')
plt.savefig(outputdir+'sigma.jpg')
plt.clf()
print "state_list holds the states of interest"
