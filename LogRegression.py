import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV



class LogRegression(object):
    """docstring for LogRegression."""
        
    def __init__(self,arg):
        super(LogRegression, self).__init__()
        self.arg = arg
        self.theta=None
        self.prob=[]
        self.loss=0

    
    
    def fit(self, x_train, y_train,lr,itr,alpha):
        self.cost=0
        l=len(x_train)
        self.theta = np.zeros(x_train.shape[1])
        for _ in range(itr):
            linear_model = np.dot(x_train, self.theta) 
            y_pred = self.sigmoid(linear_model)
            derivative = (1 / l) * (np.dot(x_train.T, (y_pred - y_train))+(2*alpha*self.theta))
            self.theta -= lr * derivative
        
           

    def predict(self, x_test):
        self.prob = self.sigmoid(np.dot(x_test, self.theta))
        y_pred = [1 if i > 0.5 else 0 for i in self.prob]
        return np.array(y_pred)

    def sigmoid(self, data):
        return 1 / (1 + np.exp(-data))
    
    def log_loss(self,y_test,itr):
         self.loss=0
         for i in range(len(y_test)):
            self.loss=self.loss + -y_test[i]*np.log(self.prob[i])-(1-y_test[i])*np.log(1-self.prob[i])
         
         return self.loss/len(y_test)
    
        
   
    
   
            
    
    
    
       
    
              
