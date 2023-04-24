import pandas as pd
PATH = ('C:\\Workspace\\CaloryPJ\\datas\\open\\train.csv')
data = pd.read_csv(PATH)
#data.isna().sum() #Missing Values check

data.drop_duplicates(inplace=True) # Duplicates removed 
data.dropna(axis=0, inplace=True) # Missing Values removed
data.drop('ID',axis=1, inplace=True) # ID remove

#Label Encoding
data['Gender'] = data['Gender'].map({'M':0,'F':1}) # M,F label encoding
data['Weight_Status'] = data['Weight_Status'].map({'Normal Weight':0, 'Overweight':1, 'Obese':2})

# 특성(X) 및 레이블(y) define
X = data.drop('Calories_Burned', axis=1)
y = data['Calories_Burned']

#Scaling
from sklearn.preprocessing import MinMaxScaler 
mscaler = MinMaxScaler() #MinMaxScaler Class Name assigned
X= mscaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression 
model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test) #위 트레인한 값으로 X_test 예측하여 확인

from sklearn.metrics import r2_score #결정계수
r2_score(y_test, pred)

import matplotlib.pyplot as plt
plt.plot(y_test.to_numpy(),'ro') #Actual
plt.plot(pred,'go') #Prediction