import tables as tb
import h5py
import numpy as np
from matplotlib import pyplot as plt
import os


VisualizeParams = True
VisualizeRecon = True

#Read original test_data before corruption

fh = h5py.File("/users/ml/xoex6879/data/ECoG/ec14/aEC14_B2.mat",'r')
ecog_data = fh[fh['/ecog/data'][0,0]].value
ecog_dataFs = fh['/ecog/dataFs'].value[0,0]
audio = np.squeeze(fh['/ecog/audio'].value)
audioFs = fh['/ecog/audioFs'].value[0,0]
f=fh[fh['/ecog/f'].value[0,0]].value
fh.close()

ecog_data_5_30hz = np.mean(ecog_data[:,1:4,:],1).T
audio = audio.reshape((ecog_data_5_30hz.shape[0], audioFs/ecog_dataFs))
data = np.concatenate([ecog_data_5_30hz,audio],1)

# train_data = data[:-1000,:]
test_orig_data = data[-1000:,:]
del audio, ecog_data_5_30hz, data



RUN = 'common_cause_ecog.py.2015-09-09+22:39'
fname = os.getenv("HOME")+'/workspace/pylib/examples/ecog/output/'+RUN+'/result.h5'
imDirName = os.getenv("HOME")+'/workspace/pylib/examples/ecog/output/'+RUN+'/images'
if not os.path.exists(imDirName):
    os.mkdir(imDirName,0700)
print fname
fh = tb.openFile(fname,'r')

W_all = fh.root.W.read()
y_all = fh.root.y.read()[0]
pi_all = fh.root.pi.read()
sigma_all = fh.root.sigma.read()
test_recon = None
test_data = None
if fh.root.__contains__('test_recon') and fh.root.__contains__('test_data'):
    test_recon = fh.root.test_recon[0]
    test_data = fh.root.test_data[0]
else:
    VisualizeRecon = False
fh.close()


if VisualizeParams:

    H = W_all.shape[-1]
    D = W_all.shape[-2]

    W_all_audio = W_all[:,-160:,:]
    W_all_ecog = W_all[:,:-160,:]

    W_last_audio = W_all_audio[-1]
    W_last_ecog = W_all_ecog[-1]

    y_all_audio = y_all[:,-160:]
    y_all_ecog = y_all[:,:-160]

    emax= np.max(np.abs(W_last_ecog))
    emin=-emax
    amax= np.max(np.abs(W_last_audio))
    amin=-amax
    for h in range(H):
      plt.imshow(W_last_ecog[:,h].reshape((16,16)),interpolation='nearest',vmin=emin,vmax=emax)
      plt.axis('off')
      plt.savefig("{}/field_{:0>4}_ecog.png".format(imDirName,h))
      plt.clf()
      plt.plot(np.linspace(0,10.,160),W_last_audio[:,h])
      plt.axis([0,10,amin,amax])
      plt.savefig("{}/field_{:0>4}_audio.png".format(imDirName,h))
      plt.clf()
    plt.plot(sigma_all  )
    plt.savefig("{}/sigma_ecog.png".format(imDirName))
    plt.clf()
    plt.plot(pi_all)
    plt.savefig("{}/pi_ecog.png".format(imDirName))
    plt.clf()

if VisualizeRecon:
  N = test_recon.shape[0]
  test_recon_ecog  = test_recon[:,:-160]
  test_recon_audio = test_recon[:,-160:]
  test_orig_ecog  = test_orig_data[:,:-160]
  test_orig_audio = test_orig_data[:,-160:]
  emax= np.max(np.abs(test_recon_ecog))
  emin=-emax
  amax= np.max(np.abs(test_recon_audio))
  amin=-amax
  for numDpt in range(N):
    plt.subplot(1,2,1)
    plt.imshow(test_recon_ecog[numDpt,:].reshape((16,16)),interpolation='nearest',vmin=emin,vmax=emax,label='reconstruction')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(test_orig_ecog[numDpt,:].reshape((16,16)),interpolation='nearest',vmin=emin,vmax=emax,label='original')
    plt.axis('off')
    plt.savefig("{}/test_recon_{:0>4}_ecog.png".format(imDirName,numDpt))
    plt.clf()
    plt.plot(np.linspace(0,10.,160),test_recon_audio[numDpt,:],'r',label='reconstruction')
    plt.plot(np.linspace(0,10.,160),test_orig_audio[numDpt,:],'g',label='original')
    plt.legend()
    plt.axis([0,10,amin,amax])
    plt.savefig("{}/test_recon_{:0>4}_audio.png".format(imDirName,numDpt))
    plt.clf()
