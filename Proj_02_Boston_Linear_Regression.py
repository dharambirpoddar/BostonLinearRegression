import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

boston= datasets.load_boston()
X=boston.data
y=boston.target



#model learning
X_train,X_test, y_train, y_test=\
    train_test_split(X,y,test_size=0.2, random_state=10)
reg=LinearRegression()
reg.fit(X_train,y_train)

#checking
y_train_predict = reg.predict(X_train) #training 
rmse = np.sqrt(mean_squared_error(y_train,y_train_predict));
r2 = r2_score(y_train, y_train_predict)

print('Train RMSE =', rmse)
print('Train R2 score =', r2)
print("\n")

#implent on real world
y_test_predict = reg.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)
print('Test RMSE =', rmse)
print('Test R2 score =', r2)
