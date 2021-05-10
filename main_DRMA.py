import numpy as np
import multiprocessing
from sklearn import preprocessing
from math import sqrt,log,exp,ceil,floor
import time
import os
import pickle
import sys


from estimator import *
from anchorselector import *
import decomposition as decomp


class Template(object):
	def __init__(self, m, n, p, dataset):
		self.dataset = dataset
		self.m = m #用户个数
		self.n = n #物品个数
		self.para = p

	@abc.abstractmethod
	def load_data(self):
		pass

	def anchor_select(self,data,TransM,q):
		# 随机游走选择锚点
		rw_anchor_selector = RWalkAnchorSelector(self.m,self.n)
		anchors = rw_anchor_selector.anchor_select(data,q,TransM)
		return anchors


	def preprocess(self,data):
		l = self.m + self.n
		#带评分的邻接矩阵
		AdjVU,TransVU = np.zeros((self.m,self.n)),np.zeros((self.m,self.n))
		for u,v,r in data:
			AdjVU[u][v] = r
		AdjUV,TransUV = AdjVU.T,np.zeros((self.n,self.m))
		#归一化得到转移矩阵
		AdjVU_sum = AdjVU.sum(axis=0)
		AdjUV_sum = AdjUV.sum(axis=0) 
		for i in range(self.n):
			if AdjVU_sum[i]>0:
				TransVU[:,i] = AdjVU[:,i]/AdjVU_sum[i]
		for i in range(self.m):
			if AdjUV_sum[i]>0:
				TransUV[:,i] = AdjUV[:,i]/AdjUV_sum[i] 
		return (TransVU,TransUV)
	
	
	def random_walk(self,TransM,anchors):
		"""重启动随机游走"""
		print('start random walk')
		TransVU,TransUV = TransM
		l = self.m + self.n
		q = len(anchors)
		alpha = 0.5
		#初始节点分布矩阵
		probU,probV = np.zeros((self.m, q)),np.zeros((self.n, q))
		#重启动矩阵
		restartU,restartV = np.zeros((self.m,q)),np.zeros((self.n,q))
		for i in range(q):
			au,ai = anchors[i][0],anchors[i][1]
			restartU[au][i] = 1
			probU[au][i]=1
			restartV[ai][i] = 1
		
		while True:
			probU_t = alpha*np.dot(TransVU,probV) + (1-alpha)*restartU
			probV_t = np.dot(TransUV,probU) 
			residual = np.sum(abs(probU-probU_t))+np.sum(abs(probV-probV_t))
			probU,probV = probU_t,probV_t
			if abs(residual)<1e-8:
				pU = probU.copy()
				break 

		probU[:,:],probV[:,:] = 0,0
		for i in range(q):
			au,ai = anchors[i][0],anchors[i][1]
			probV[ai][i]=1
		while True:
			probV_t = alpha*np.dot(TransUV,probU) + (1-alpha)*restartV 
			probU_t = np.dot(TransVU,probV) 
			residual = np.sum(abs(probU-probU_t))+np.sum(abs(probV-probV_t))
			probU,probV = probU_t,probV_t
			if abs(residual)<1e-8:
				pV = probV.copy()
				break 

		return (pU,pV)
	

	def submatrix_const(self,prob,q,data_train,data_test):
		print('start constructing submatrices')
		#分别得到用户物品的稳态概率矩阵
		probU,probV = prob
		anchor_neighuser = {}
		anchor_neighitem = {}
		for u in range(self.m):
			index_val = [(i,j) for i,j in enumerate(probU[u])]
			index_val = sorted(index_val,key=lambda s: s[1],reverse=True)[:int(q*self.para)]
			for p in index_val:
				anchor_neighuser.setdefault(p[0],[])
				anchor_neighuser[p[0]].append(u)
		for v in range(self.n):
			index_val = [(i,j) for i,j in enumerate(probV[v])]
			index_val = sorted(index_val,key=lambda s: s[1],reverse=True)[:int(q*self.para)]
			for p in index_val:
				anchor_neighitem.setdefault(p[0],[])
				anchor_neighitem[p[0]].append(v)

		subdata_train,subdata_test = {},{}
		nargs = [(data_train,data_test,anchor_neighuser[i],anchor_neighitem[i],i) for i in range(q) if (i in anchor_neighuser.keys()) and (i in anchor_neighitem.keys())]
		multiprocessing.freeze_support()
		cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=cores - 2)
		for y in pool.imap(self.get_subdata,nargs):
			this_train, this_test, i = y
			subdata_train[i] = this_train
			subdata_test[i] = this_test
		pool.close()
		pool.join()

		return (subdata_train,subdata_test)	


	def local_train(self,q,subdata_train,subdata_test,weight_u,weight_v):
		print('start local train')
		multiprocessing.freeze_support()
		cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=cores - 2)
				
		eachpred_dict = {}
		nargs = [(subdata_train[i], subdata_test[i], q, i,weight_u[i],weight_v[i]) for i in range(q) if i in subdata_train.keys()]
		for y in pool.imap(self.train_submatrix, nargs):
			pred_rate, subdata_test_i, i = y
			#得到每个用户-物品在不同子矩阵下的打分
			self.fill_pred_dict(eachpred_dict,pred_rate, subdata_test_i, q, i)
			sys.stdout.write('have finished training for %d/%d local matrices\r' % (i+1,q))
		pool.close()
		pool.join()
		return eachpred_dict


	def local_weight(self,q,subdata_train,subdata_test):
		multiprocessing.freeze_support()
		cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=cores - 2)
		
		user_weight,item_weight = {},{}
		nargs = [(subdata_train[i],subdata_test[i], i) for i in range(q) if i in subdata_train.keys()]
		for y in pool.imap(self.get_weight, nargs):
			user_weight_i,item_weight_i,i = y
			user_weight[i] = user_weight_i
			item_weight[i] = item_weight_i
		pool.close()
		pool.join()
		return user_weight,item_weight


	def predict(self,data_test,eachpred_dict):
		#子矩阵中测试数据评分
		true_dict_test = self.get_datadic(data_test)
		pred_dict = {}
		for user in eachpred_dict:
			pred_dict.setdefault(user, {})
			for item in eachpred_dict[user]:
				ratesum = ratenum = 0
				for i in eachpred_dict[user][item]:
					if i != 0:
						ratesum += i
						ratenum += 1
				pred_dict[user][item] = ratesum / ratenum
	
		estimator = Estimator()
		mae = round(estimator.get_mae(pred_dict, true_dict_test),4)
		rmse = round(estimator.get_rmse(pred_dict, true_dict_test),4)
		print("directly average, MAE:" + str(mae) + ";RMSE:" + str(rmse))
		
		return mae,rmse


	def predict_weight_uv(self,data_test,eachpred_dict,weight_u,weight_v,q):
		#子矩阵中测试数据评分
		true_dict_test = self.get_datadic(data_test)
		pred_dict = {}
		rates = np.array([1,2,3,4,5])
		for user in eachpred_dict:
			pred_dict.setdefault(user, {})
			for item in eachpred_dict[user]:
				ratesum = weightsum = 0
				a = []
				for i in range(len(weight_u)):
					if eachpred_dict[user][item][i]!=0:
						rate = eachpred_dict[user][item][i]
						wu = np.sum(1/(abs(rates-rate)+0.5)*weight_u[i][user])
						wv = np.sum(1/(abs(rates-rate)+0.5)*weight_v[i][item])
						weight = wu*wv					
						ratesum += rate*weight
						weightsum += weight
				pred_dict[user][item] = ratesum / weightsum
	

		estimator = Estimator()
		mae = round(estimator.get_mae(pred_dict, true_dict_test),4)
		rmse = round(estimator.get_rmse(pred_dict, true_dict_test),4)
		print('Harmonic average result with weight is', "MAE:" + str(mae) + ";RMSE:" + str(rmse))
	
		return mae,rmse


	#--------------------------------------------
	def get_weight(self,args):
		"""
		工具方法：将预测的值填入字典
		"""
		subtrain,subtest,q = args
		
		u_distri_q,v_distri_q = {},{}
		for u,v,r in subtrain:
			u_distri_q.setdefault(u,np.zeros(5))
			v_distri_q.setdefault(v,np.zeros(5))
			u_distri_q[u][int(r-1)]+=1
			v_distri_q[v][int(r-1)]+=1
		for u,v,r in subtest: 
			u_distri_q.setdefault(u,np.zeros(5))
			v_distri_q.setdefault(v,np.zeros(5))

		user_weight_q,item_weight_q = {},{}
	
		for u in u_distri_q: 					
			if np.sum(u_distri_q[u])>0: 
				user_weight_q[u] = 1 + 1*u_distri_q[u]/np.sum(u_distri_q[u])
			else:
				user_weight_q[u] = 1 + u_distri_q[u]
		for v in v_distri_q:	
			if np.sum(v_distri_q[v])>0:
				item_weight_q[v] = 1 + 1.8*v_distri_q[v]/np.sum(v_distri_q[v])
			else:
				item_weight_q[v] = 1 + v_distri_q[v]
		
		return user_weight_q, item_weight_q, q


	def get_subdata(self,args):
		"""
		工具方法：找到子矩阵的点
		"""
		data_train,data_test,neighuser,neighitem,i = args
		subdata_train,subdata_test = [],[]
		for d in data_train:
			if (d[0] in neighuser) and (d[1] in neighitem):
				subdata_train.append(d)
		for d in data_test:
			if (d[0] in neighuser) and (d[1] in neighitem):
				subdata_test.append(d)
		return subdata_train,subdata_test,i


	def train_submatrix(self,args):
		"""
		工具方法：服务于并行训练
		"""
		data_train,data_test,q,i,weight_u,weight_v = args
		svd = decomp.SVD(data_train, k=10)
		svd.train(weight_u,weight_v,data_test, steps=100, gamma=0.002, Lambda=0.01, epsilon=0.0001)
		pred_rate = svd.test(data_test)
		return (pred_rate, data_test,i)


	def fill_pred_dict(self,dict_data,pred,test,len_q,q):
		"""
		工具方法：将预测的值填入字典
		"""
		for i in range(len(test)):
			dict_data.setdefault(test[i][0],{})
			dict_data[test[i][0]].setdefault(test[i][1],np.zeros(len_q))
			dict_data[test[i][0]][test[i][1]][q]=pred[i]


	def get_datadic(self,data):
		"""
		工具方法：构建满足需求的字典 
		"""
		true_dict={}
		for i in range(len(data)):
			uid = data[i][0]
			mid = data[i][1]
			rate = data[i][2]
			true_dict.setdefault(uid, {})
			true_dict[uid][mid] = rate
		return true_dict


class ML100k(Template):
	def load_data(self,fold=1):
		data_train,data_test = [],[]
		file_path_train = "ml-100k/u"+str(fold)+".base"
		file_path_test = "ml-100k/u"+str(fold)+".test"
		with open(file_path_train) as f:
			for line in f:
				a = line.split("\t")
				data_train.append((int(a[0]) - 1, int(a[1]) - 1, int(a[2])))  # 此处已经把id变成索引（-1）
		with open(file_path_test) as f:
			for line in f:
				a = line.split("\t")
				data_test.append((int(a[0]) - 1, int(a[1]) - 1, int(a[2])))  # 此处已经把id变成索引（-1）
		return data_train,data_test

class ML1m(Template):
	def load_data(self,fold=1):
		with open("ml-1m/data/train"+str(fold)+".txt",'r') as f:
			data_train = eval(f.read())
		with open("ml-1m/data/test"+str(fold)+".txt",'r') as f:
			data_test = eval(f.read())
		return (data_train,data_test)

class Ciao(Template):
	def load_data(self,fold=1):
		with open("ciao/data/train"+str(fold)+".txt",'r') as f:
			data_train = eval(f.read())
		with open("ciao/data/test"+str(fold)+".txt",'r') as f:
			data_test = eval(f.read())
		return (data_train,data_test)


if __name__=='__main__':
	p = 0.7 # 局部矩阵规模控制参数
	q = 50 # 锚点个数 
	fold = 1 # 第一折

	DataInstance = ML1m(m=6040,n=3952,p=p, dataset = "ML1M")
	

	#加载数据
	data_train,data_test = DataInstance.load_data(fold)

	#特征提取,得到状态转移矩阵
	TransM = DataInstance.preprocess(data_train)
	#选择锚点
	anchors = DataInstance.anchor_select(data_train,TransM,q)
	#随机游走寻找邻域
	anchorM = DataInstance.random_walk(TransM,anchors)
	#得到以每个锚点为中心子矩阵所包含的训练集和测试集
	subdata_train,subdata_test = DataInstance.submatrix_const(anchorM,q,data_train,data_test)
	
	#训练
	weight_u,weight_v = DataInstance.local_weight(q,subdata_train,subdata_test)
	eachpred_dict = DataInstance.local_train(q,subdata_train,subdata_test,weight_u,weight_v)

	#预测
	mae,rmse = DataInstance.predict_weight_uv(data_test,eachpred_dict,weight_u,weight_v,q)

	# 计算局部矩阵平均规模
	a=0
	for i in subdata_train:
		a = a + len(subdata_train[i])
	print(a/50)	
