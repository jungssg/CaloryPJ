# 데이터 불러오기
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

train = pd.read_csv('/Users/js/Desktop/MiniProject/CaloryPJ/data/train.csv')
test = pd.read_csv('/Users/js/Desktop/MiniProject/CaloryPJ/data/test.csv')

train_data = train.copy()
test_data = test.copy()

train_data.info()
test_data.info()

train_data.isna().sum()
test_data.isna().sum()

train_data.drop(['ID'], axis=1, inplace=True)
test_data.drop(['ID'], axis=1, inplace=True)

train_data.info()
test_data.info()

train_data.describe()

train_data['Weight_Status'].value_counts()
#몸무게 상태, 성별을 one-hot으로 변경
train_data = pd.get_dummies(train_data, columns=['Weight_Status','Gender']) 
test_data = pd.get_dummies(test_data, columns=['Weight_Status','Gender'])

train_data.head()
test_data.head()

train_data.describe()

##이상치 제거전에 poly 실행하기위해 calories burned 를 빼놓음
cal = train_data.pop('Calories_Burned')
aa_train = train_data

aa_train.info()

#정규화
from sklearn.preprocessing import MinMaxScaler
mmscaler = MinMaxScaler()
mmscaler.fit(aa_train)
aa_train = mmscaler.transform(aa_train)

#변수생성(feature수를 늘림)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()

X_poly = pd.DataFrame(poly.fit_transform(aa_train))
# X_tpoly = pd.DataFrame(poly.fit_transform(X_test))
X_poly
X_poly['Calories_Burned'] = cal
X_poly.info()

#여기서 이상치제거
def remove(df):
  columns = df.columns
  df_clean = df.copy()
  outliers_list=[]
  for column in columns:
    q1 = df_clean[column].quantile(0.25)
    q3 = df_clean[column].quantile(0.75)
    IQR = q3 - q1
    lower = q1 - (IQR * 1.5)
    upper = q3 + (IQR * 1.5)

    outliers = df_clean[(df_clean[column] < lower) | (df_clean[column] > upper)]
    outliers_list.append(outliers)
    df_clean = df_clean[(df_clean[column]>=lower)&(df_clean[column]<=upper)]
  
  return df_clean , pd.concat(outliers_list)

re_train, out_train = remove(X_poly)

re_train.info()

y = re_train.pop('Calories_Burned')
X = re_train

X
y

#모델링
from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train, y_test = train_test_split(X, y, random_state=42)

# DecisionTreeRegresoor 방법1
# from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor(random_state=42)

# model.fit(X_train1,y_train)

#xgboost사용
from xgboost import XGBRegressor

xgb_model = XGBRegressor()
xgb_model.fit(X_train1, y_train)

pred = xgb_model.predict(X_test1)

#성능검사
from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(y_test, pred)
np.sqrt(RMSE)

# test_data도 x_ploy와 같은 컬럼수를 같게하기위해 동일하게 늘려줌
test_poly = pd.DataFrame(poly.fit_transform(test_data))

test_pred = xgb_model.predict(test_poly)

#제출파일생성코드
submission = pd.read_csv('/Users/js/Desktop/MiniProject/CaloryPJ/data/sample_submission.csv')
submission['Calories_Burned'] = test_pred
submission.to_csv('./submit.csv', index=False)




