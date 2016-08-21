#
#  Lincense: Academic Free License (AFL) v3.0
#


import numpy as np
import tables as tb

def read_chunks(N=1000, sz=30, disc=1):
	h5f = tb.open_file('/users/ml/xoex6879/data/TIMIT/TIMIT'+str(disc)+'_TRAIN.h5','r')
	it = h5f.walk_nodes(h5f.root)
	audio_samples = []
	for node in it:
		if node._v_name=='audio':
			audio_samples.append(node)
	c=0
	data = np.zeros((N,sz))
	while c<N:
		for sample in audio_samples:
			MIN = np.random.randint(0,sample.shape[-1]-sz)
			MAX = MIN + sz
			data[c]=sample[MIN:MAX]
			c+=1
			if c==N:
				break
	return data


def read(N=1000, sz=30,overlap=0.5, disc=1):
	h5f = tb.open_file('/users/ml/xoex6879/data/TIMIT/TIMIT'+str(disc)+'_TRAIN.h5','r')
	it = h5f.walk_nodes(h5f.root)
	audio_samples = []
	for node in it:
		if node._v_name=='audio':
			audio_samples.append(node)
	c=0
	data = np.zeros((N,sz))
	for sample in audio_samples:
		samplesize = sample.shape[0]
		if c==N:
			break
		ind=0
		while (ind+sz)<samplesize: 		
			MIN = ind
			MAX = MIN+sz
			data[c]=sample[MIN:MAX]
			c+=1
			if c==N:
				break
			ind+= int((1-overlap)*sz)
	return data

