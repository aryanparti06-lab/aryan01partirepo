import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path_to_folder = r'D:\Work\Kaggle\house-prices-advanced-regression-techniques'


train_df = pd.read_csv(path_to_folder + r'\train.csv')
X_test = pd.read_csv(path_to_folder + r'\test.csv')
X_train = train_df #.iloc[:,:-1]

#drop columns
X_train.drop(['Id', 'SaleType'], inplace=True, axis=1)
X_test.drop(['Id', 'SaleType'], inplace=True, axis=1)

#PLOT DATA
#plt.scatter(X_train['GrLivArea'],X_train['SalePrice'])
#remove outlier
plt.scatter(X_train['GrLivArea'],X_train['SalePrice'], c='red') # outliers marked in red
X_train = X_train.drop(X_train[(X_train['GrLivArea'] > 4000) & (X_train['SalePrice'] < 200000)].index)
X_train.shape
plt.scatter(X_train['GrLivArea'],X_train['SalePrice'], c='blue')
#plt.hist(X_train['SalePrice'])
X_train['SalePrice'] = np.log1p(X_train['SalePrice'])
y_train = X_train['SalePrice']
X_train.drop(['SalePrice'], inplace=True, axis=1)
X_train.shape


<Figure size 432x288 with 1 Axes><img width="393" height="252" alt="image" src="https://github.com/user-attachments/assets/0ba9048c-bac4-4026-91a1-a71bb8d38a03" />


#Combine Train Test
combine_df = pd.concat([X_train, X_test])
combine_df.shape
