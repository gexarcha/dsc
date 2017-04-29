import numpy as np
import tables as tb
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 26
mpl.rcParams['font.size'] = 26
mpl.rc('text', usetex=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
import os
import tqdm

#Gabor Parameters       [A theta frequ phi sX sY xZero yZero err]
#Mexican Hat Parameters [sigma1 sigma2 xZero yZero A1 A2]
outputdird = 'output/'+'dsc_run.py.d3981/' #dDSC
outputdirt = 'output/'+'dsc_run.py.d3985/' #tDSC
outputdirb = 'output/'+'dsc_run.py.d4097/' #bDSC 
outputdira = 'output/'+'dsc_run.py.d796458/' #aDSC
# outputdira = 'output/'+'dsc_run.py.d802072/' #aDSC


gabparamsa=np.loadtxt(outputdira+'MatchedGaborParameters.txt')
dogparamsa=np.loadtxt(outputdira+'MatchedMaxicanHatParameters.txt')
globflagsa=dogparamsa[:,-1]<gabparamsa[:,-1]
fh=tb.open_file(outputdira+'result.h5','r')
sigmaa=fh.root.sigma.read().squeeze()
pia=fh.root.pi.read().squeeze()
fh.close()

gabparamsb=np.loadtxt(outputdirb+'MatchedGaborParameters.txt')
dogparamsb=np.loadtxt(outputdirb+'MatchedMaxicanHatParameters.txt')
globflagsb=dogparamsb[:,-1]<gabparamsb[:,-1]
fh=tb.open_file(outputdirb+'result.h5','r')
sigmab=fh.root.sigma.read().squeeze()
pib=fh.root.pi.read().squeeze()
fh.close()

gabparamst=np.loadtxt(outputdirt+'MatchedGaborParameters.txt')
dogparamst=np.loadtxt(outputdirt+'MatchedMaxicanHatParameters.txt')
globflagst=dogparamst[:,-1]<gabparamst[:,-1]
fh=tb.open_file(outputdirt+'result.h5','r')
sigmat=fh.root.sigma.read().squeeze()
pit=fh.root.pi.read().squeeze()
fh.close()

gabparamsd=np.loadtxt(outputdird+'MatchedGaborParameters.txt')
dogparamsd=np.loadtxt(outputdird+'MatchedMaxicanHatParameters.txt')
globflagsd=dogparamsd[:,-1]<gabparamsd[:,-1]
fh=tb.open_file(outputdird+'result.h5','r')
sigmad=fh.root.sigma.read().squeeze()
pid=fh.root.pi.read().squeeze()
fh.close()



fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
sigmas=np.array([sigmab[-1],sigmat[-1],sigmaa[-1],sigmad[-1]])
ax.bar(np.arange(1,5),sigmas,tick_label=['BSC','tDSC','aDSC','dDSC'] )
fig.savefig(outputdirb+'sigmas.jpg')
fig.savefig(outputdirt+'sigmas.jpg')
fig.savefig(outputdird+'sigmas.jpg')
fig.savefig(outputdira+'sigmas.jpg')
plt.close(fig)

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
crowdedness=np.zeros(4)
crowdedness[0]=pib[-1,1]*300
crowdedness[1]=(pit[-1,0]+pit[-1,2])*300
crowdedness[2]=(pia[-1,0]+pia[-1,2:].sum())*300
crowdedness[3]=(pid[-1,1:].sum())*300
ax.bar(np.arange(1,5),crowdedness,tick_label=['bDSC','tDSC','aDSC','dDSC'] )
fig.savefig(outputdirb+'crowdedness.jpg')
fig.savefig(outputdirt+'crowdedness.jpg')
fig.savefig(outputdird+'crowdedness.jpg')
fig.savefig(outputdira+'crowdedness.jpg')
plt.close(fig)

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
globs=np.zeros(4)
globs[0]=globflagsb.sum()
globs[1]=globflagst.sum()
globs[2]=globflagsa.sum()
globs[3]=globflagsd.sum()
ax.bar(np.arange(1,5),globs,tick_label=['bDSC','tDSC','aDSC','dDSC'] )
fig.savefig(outputdirb+'globulars.jpg')
fig.savefig(outputdirt+'globulars.jpg')
fig.savefig(outputdird+'globulars.jpg')
fig.savefig(outputdira+'globulars.jpg')
plt.close(fig)

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
L=np.zeros((4))

Ltmpgabb=gabparamsb[:,4]*gabparamsb[:,5]*4/256
Ltmpgabb[Ltmpgabb>1]=1
Ltmpmexb=(np.max(dogparamsb[:,:2],1)**2)*4/256
Ltmpmexb[Ltmpmexb>1]=1
Ltmpb = Ltmpgabb.copy()
Ltmpb[globflagsb]=Ltmpmexb[globflagsb]

Ltmpgabt=gabparamst[:,4]*gabparamst[:,5]*4/256
Ltmpgabt[Ltmpgabt>1]=1
Ltmpmext=(np.max(dogparamst[:,:2],1)**2)*4/256
Ltmpmext[Ltmpmext>1]=1
Ltmpt = Ltmpgabt.copy()
Ltmpt[globflagst]=Ltmpmext[globflagst]

Ltmpgaba=gabparamsa[:,4]*gabparamsa[:,5]*4/256
Ltmpgaba[Ltmpgaba>1]=1
Ltmpmexa=(np.max(dogparamsa[:,:2],1)**2)*4/256
Ltmpmexa[Ltmpmexa>1]=1
Ltmpa = Ltmpgaba.copy()
Ltmpa[globflagsa]=Ltmpmexa[globflagsa]

Ltmpgabd=gabparamsd[:,4]*gabparamsd[:,5]*4/256
Ltmpgabd[Ltmpgabd>1]=1
Ltmpmexd=(np.max(dogparamsb[:,:2],1)**2)*4/256
Ltmpmexd[Ltmpmexd>1]=1
Ltmpd = Ltmpgabd.copy()
Ltmpd[globflagsd]=Ltmpmexd[globflagsd]

L[0]=Ltmpb.mean()
L[1]=Ltmpt.mean()
L[2]=Ltmpa.mean()
L[3]=Ltmpd.mean()
L/=L[0]
ax.bar(np.arange(1,5),L,tick_label=['bDSC','tDSC','aDSC','dDSC'] )
fig.savefig(outputdirb+'surface.jpg')
fig.savefig(outputdirt+'surface.jpg')
fig.savefig(outputdird+'surface.jpg')
fig.savefig(outputdira+'surface.jpg')
plt.close(fig)
