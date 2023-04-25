#Data loading
import pandas as pd
PATH = ('C:\\Workspace\\CaloryPJ\\datas\\open\\train.csv')
data = pd.read_csv(PATH)

#outliers check
import plotly.express as px 
fit = px.scatter(data, x='Exercise_Duration', y='Calories_Burned')
#Delete outliers (표의 상위에 있는 2개 이상치로 추정)
rows_to_delete = data[data['Calories_Burned'].isin([295,300])]
data = data.drop(rows_to_delete.index)

#Delete duplicates,missing, unnecessary values
data.drop_duplicates(inplace=True) # Duplicates removed 
data.dropna(axis=0, inplace=True) # Missing Values removed
data.drop('ID',axis=1, inplace=True) # ID remove

#Label Encoding
data['Gender'] = data['Gender'].map({'M':0,'F':1}) 
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

#Modeling
from sklearn.linear_model import LinearRegression 
model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test) #위 트레인한 값으로 X_test 

from sklearn.metrics import r2_score #결정계수
r2_score(y_test, pred)
