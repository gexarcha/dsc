import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# plt.rc('text', usetex=True)
import tables as tb
import os
import tqdm

MEAN_HIDDEN=False
outputdir = 'output/'+'dsc_on_hc1_run.py.2016-03-17+11:39/'
outputdir = 'output/'+'dsc_on_hc1_run.py.2016-03-17+21:16/'
outputdir = 'output/'+'dsc_on_hc1_run.py.2016-03-23+01:56/'
outputdir = 'output/'+'dsc_on_hc1_run.py.2016-03-23+13:05/'
outputdir = 'output/'+'dsc_run_audio.py.2016-04-21+09:50/'
outputdir = 'output/'+'dsc_run_audio.py.d587969/'
ofile = outputdir+'result.h5'

fh =tb.open_file(ofile,'r')

sigma_all = fh.root.sigma.read().squeeze()
pi_all = fh.root.pi.read().squeeze()

W_all = fh.root.W.read()
H = W_all.shape[-1]
D = W_all.shape[-2]
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
            minwl = -np.max(np.abs(this_W)) * 1.1 
            maxwl = -minwl
            # minwl = np.min(this_W) * 1.1
            # maxwl = np.max(this_W) * 1.1
            # meanwl = np.mean(this_W) * 1.1
            if cscale == 'global':
                maxw, minw = maxwg, minwg
            elif cscale == 'local':
                maxw, minw = maxwl, minwl
            ax = fig.add_subplot(13,8,h+1)
            plt.locator_params(nbins=6)
            ax.plot(np.linspace(0,D/10,num=D) ,this_W)#scale in kHz
            # ax.axis([0,D/10,minwg,maxwg],fontsize=16)
            ax.axis([0,D/10,minwl,maxwl],fontsize=16)
            # ax.savefig(outputdir+'_images/'+'W_e_{:03}_h_{:03}.jpg'.format(e,h))
            ax.set_title("$W_{"+str(h+1)+"}$",fontsize=16)
            ax.tick_params(axis='both',labelsize=16)




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
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.
for h in xrange(H):
    from scipy.io.wavfile import write
    scaled_W = np.int16(W_all[:,:,h]/np.max(np.abs(W_all[:,:,h]))*32767)
    write(outputdir+'filters/f{:03}.wav'.format(h),16000,scaled_W.flatten())

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sigma_all,label='$\sigma$')
fig.savefig(outputdir+'sigma.jpg')
plt.close(fig)
