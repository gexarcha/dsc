#
#  Lincense: Academic Free License (AFL) v3.0
#
import numpy as np

fname = '../../Data/d5611/d561102'

def readdat(fname,nChannels=8):
	with open(fname+'.dat','rb') as fh:
		ar = np.fromstring(fh.read(),dtype=np.int16).astype(np.double)
		return ar.reshape((-1,nChannels)).T

def readfil(fname,nChannels=8):
	with open(fname+'.fil','rb') as fh:
		ar = np.fromstring(fh.read(),dtype=np.int16).astype(np.double)
		return ar.reshape((-1,nChannels)).T

def readxml(fname,):
	import xml.etree.ElementTree as ET
	xml = ET.parse(fname+'.xml')
	root = xml.getroot()
	d = {}
	for c in range(len(root)):
		if root[c].tag=='generalInfo':
			d['Date']=root[c][0].text
			continue

		if root[c].tag=='acquisitionSystem':
			d['nBits']=int(root[c][0].text)
			d['nChannels']=int(root[c][1].text)
			d['SampleRate']=int(root[c][2].text)
			d['SampleTime']=1e6/d['SampleRate']
			d['VoltageRange']=int(root[c][3].text)
			d['Amplification']=int(root[c][4].text)
			d['Offset']=int(root[c][5].text)
			continue

		if root[c].tag=='fieldPotentials':
			d['lfpSampleRate']=int(root[c][0].text)
			continue

		if root[c].tag=='anatomicalDescription':
			d['anatomicalDescription'] = {}
			td = d['anatomicalDescription']
			for grp  in range(len(root[c][0])):
				td['group_{}'.format(grp)]=[]
				for ch in range(len(root[c][0][grp])):
					td['group_{}'.format(grp)].append((int(root[c][0][grp][ch].text), root[c][0][grp][ch].attrib))
	return d

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
def getspikes(ar,sz=20,thr=-50,max=False):
	p=None
	if max:
		thr_a = ar[ar>thr]
		ind_a = np.argwhere(ar>thr)
		p = findpeaks(thr_a,ind_a,True)
	else:
		thr_a = ar[ar<thr]
		ind_a = np.argwhere(ar<thr)
		p = findpeaks(thr_a,ind_a,False)
	spikes = np.zeros((p.shape[0],sz))
	for i,c in enumerate(p):
		spikes[i]=ar[c-sz/2:c+sz/2]
	return spikes


def getpatches(ar,N,D,overlap=0.,start=0):
	step = int((1-overlap)*D)
	data = np.zeros((N,D))
	s=start
	for n in range(N):
		data[n] = ar[s:s+D]
		s+=step
	print "returning samples from {} to {}".format(start,s+D)
	return data


# xml=readxml(fname)
# ar=readdat(fname,xml['nChannels'])
# fil=readfil(fname,xml['nChannels'])