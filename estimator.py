from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt,log

class Estimator(object):
    def rmse(self,prediction, ground_truth):
        """
            对预测分数列表和真实分数列表求rmse
        """
        return sqrt(mean_squared_error(prediction, ground_truth))

    def mae(self,prediction, ground_truth):
        return mean_absolute_error(prediction, ground_truth)

    def get_rmse(self,pred,true):
        """
        对预测字典和真实字典求rmse
        """
        pred_list=[]
        true_list=[]
        for i in pred.keys():
            for j in pred[i].keys():
                if pred[i][j]!=0:
                    pred_list.append(pred[i][j])
                    true_list.append(true[i][j])
        return self.rmse(pred_list,true_list)

    def get_mae(self,pred,true):
        """
        对预测字典和真实字典求mae
        """
        pred_list=[]
        true_list=[]
        for i in pred.keys():
            for j in pred[i].keys():
                pred_list.append(pred[i][j])
                true_list.append(true[i][j])
        return self.mae(pred_list,true_list)


