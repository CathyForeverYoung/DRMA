import numpy as np
import abc
import random
from sklearn import preprocessing

class AnchorSelector(object):
	def __init__(self, m,n):
		self.m = m
		self.n = n

	__metaclass__ = abc.ABCMeta
	@abc.abstractmethod
	def anchor_select(self):
		pass


class RWalkAnchorSelector(AnchorSelector):
	"""随机游走锚点选择"""
	def anchor_select(self,data,q,TransM):
		TransVU,TransUV = TransM
		#初始节点分布向量
		probU,probV = np.ones((self.m,1)),np.ones((self.n,1))	
		probU[:],probV[:] = 1/self.m,1/self.n
		while True:
			alpha=0.8
			probU_t = alpha*np.dot(TransVU,probV) + (1-alpha)/self.m
			probV_t = alpha*np.dot(TransUV,probU) + (1-alpha)/self.n
			residual = np.sum(abs(probU-probU_t))+np.sum(abs(probV-probV_t))
			probU,probV = probU_t,probV_t
			if abs(residual)<1e-8:
				break  

		pgu = [(i,j) for i,j in enumerate(probU)] #(id,pg_val)	
		pgv = [(i,j) for i,j in enumerate(probV)]
		uanchor = sorted(pgu,reverse=True,key=lambda s: s[1])[:q]
		vanchor = sorted(pgv,reverse=True,key=lambda s: s[1])[:q]
		random.shuffle(uanchor)
		random.shuffle(vanchor)
		anchors = []
		for m,n in zip(uanchor,vanchor): 
			anchors.append((m[0],n[0]))

		return anchors






