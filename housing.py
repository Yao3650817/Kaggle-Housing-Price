#housing['SalePrice']=numerical['SalePrice']
#housing_1['MSZoning'].describe()

list1=[]
list1.append(final_1.isna().sum())
print(list1)
df=pd.DataFrame(final_1.isna().sum())

plt2=sn.barplot(x='MSZoning', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='Street', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='Alley', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='LotShape', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='LandContour', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='Utilities', y='SalePrice', data=housing)
plt.show();plt.close()
#plt2=sn.barplot(x='LotConfig', y='SalePrice', data=housing)
#plt.show();plt.close()
#plt2=sn.barplot(x='LandSlope', y='SalePrice', data=housing)
#plt.show();plt.close()
plt2=sn.barplot(x='Neighborhood', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='Condition1', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='Condition2', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='BldgType', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='HouseStyle', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='RoofStyle', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='RoofMatl', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='Exterior1st', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='Exterior2nd', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='MasVnrType', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='ExterQual', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='ExterCond', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='Foundation', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='BsmtQual', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='BsmtCond', y='SalePrice', data=housing)
plt.show();plt.close()
#plt2=sn.barplot(x='BsmtExposure', y='SalePrice', data=housing)
#plt.show();plt.close()
#plt2=sn.barplot(x='BsmtFinType1', y='SalePrice', data=housing)
#plt.show();plt.close()
#plt2=sn.barplot(x='BsmtFinType2', y='SalePrice', data=housing)
#plt.show();plt.close()
plt2=sn.barplot(x='Heating', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='HeatingQC', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='CentralAir', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='Electrical', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='KitchenQual', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='Functional', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='FireplaceQu', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='GarageType', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='GarageFinish', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='GarageQual', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='GarageCond', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='PavedDrive', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='PoolQC', y='SalePrice', data=housing)
plt.show()
plt.close()
#plt2=sn.barplot(x='Fence', y='SalePrice', data=housing)
#plt.show();plt.close()
plt2=sn.barplot(x='MiscFeature', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='SaleType', y='SalePrice', data=housing)
plt.show();plt.close()
plt2=sn.barplot(x='SaleCondition', y='SalePrice', data=housing)
plt.show();plt.close()

sn.heatmap(corr,annot=True,annot_kws={'size':5})
plt.show()
plt.clf()
plt.cla()

boxplot=numerical.boxplot(column=['GarageYrBlt'])
plt.show()
boxplot2=numerical.boxplot(column=['LotFrontage'])
plt.show()

col_index2=[]
for col in range(0,36):
  if (0.5<corr.iloc[36,col]<1) or (-1<corr.iloc[36,col]<-0.5):
    col_index2.append(corr.columns[col])
col_index2

col_index4=[]
for col in range(0,36):
  for row in range(0,36):
    if (0.5<corr.iloc[row,col]<1) or (-1<corr.iloc[row,col]<-0.5):
      col_index4.append(corr.columns[col]+' '+corr.columns[row])
col_index4

numerical=numerical.drop(columns=col_index)

housing['Electrical'].isna().sum()
housing_1['Electrical'].isna().sum()
corr=housing.corr()

import seaborn as sn
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.feature_selection import chi2,SelectKBest
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy
import numpy as np
housing=pd.read_csv("Desktop/kaggle/housing price/train.csv")
housing_1=pd.read_csv("Desktop/kaggle/housing price/test.csv")
housing.drop(labels=['Id'],axis=1,inplace=True)


#high correlation with others
#-GarageCars,GarageYrBlt,TotalBsmtSF,GrLivArea

#final num variables to use
#OverallQual,!YearBuilt,!YearRemodAdd,1stFlrSF,FullBath,TotRmsAbvGrd,GarageArea
housing = housing[housing['Electrical'].notna()]
# or housing.dropna(subset=['Electrical'])

numerical=housing.select_dtypes(include=['int64','float64'])
housing=housing.select_dtypes(exclude=['int64','float64'])

numerical_1=housing_1.select_dtypes(include=['int64','float64'])
housing_1=housing_1.select_dtypes(exclude=['int64','float64'])

numerical['GarageYrBlt'] = numerical['GarageYrBlt'].fillna(numerical['GarageYrBlt'].median())
numerical['LotFrontage'] = numerical['LotFrontage'].fillna(numerical['LotFrontage'].median())
numerical['MasVnrArea'] = numerical['MasVnrArea'].fillna(numerical['MasVnrArea'].median())

numerical_1['GarageYrBlt'] = numerical_1['GarageYrBlt'].fillna(numerical_1['GarageYrBlt'].median())
numerical_1['LotFrontage'] = numerical_1['LotFrontage'].fillna(numerical_1['LotFrontage'].median())
numerical_1['MasVnrArea'] = numerical_1['MasVnrArea'].fillna(numerical_1['MasVnrArea'].median())
numerical_1['GarageArea'] = numerical_1['GarageArea'].fillna(numerical_1['GarageArea'].median())

housing=housing.fillna("None")
housing_1=housing_1.fillna("None")



housing.drop(['LotConfig','LandSlope'],axis=1,inplace=True)
housing_1.drop(['LotConfig','LandSlope'],axis=1,inplace=True)


housing=pd.get_dummies(housing, drop_first=True)
housing_1=pd.get_dummies(housing_1, drop_first=True)

numerical=numerical[['SalePrice','OverallQual','YearBuilt','YearRemodAdd','1stFlrSF','FullBath','TotRmsAbvGrd','GarageArea']]
numerical_1=numerical_1[['Id','OverallQual','YearBuilt','YearRemodAdd','1stFlrSF','FullBath','TotRmsAbvGrd','GarageArea']]

scipy.stats.skew(numerical, axis = 0, bias = True)
numerical['Log_SalePrice']=np.log(numerical['SalePrice'])
numerical['Log_1stFlrSF']=np.log(numerical['1stFlrSF'])
final=pd.concat([numerical,housing],axis=1)
drops=['SalePrice','1stFlrSF']
final.drop(drops,inplace=True,axis=1)

scipy.stats.skew(numerical_1, axis = 0, bias = True)
numerical_1['Log_1stFlrSF']=np.log(numerical_1['1stFlrSF'])
#numerical_1['Log_TotRmsAbvGrd']=np.log(numerical_1['TotRmsAbvGrd'])
final_1=pd.concat([numerical_1,housing_1],axis=1)
drops=['1stFlrSF']
final_1.drop(drops,inplace=True,axis=1)




#Models
x= final.loc[:, final.columns != 'Log_SalePrice']
y = final.iloc[:, 6]
x['Exterior1st_None']=0
x['Exterior2st_None']=0
x['Exterior2nd_None']=0
x['MSZoning_None']=0
x['SaleType_None']=0
x['Utilities_None']=0
x['Functional_None']=0
x['KitchenQual_None']=0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
regressor= linear_model.Lasso(max_iter=3000,alpha=0.008)
regressor.fit(x,y)
regressor.fit(x_train,y_train)

prediction=regressor.predict(x_train)
print(mean_squared_error(prediction,y_train, squared=False))

prediction1=regressor.predict(x_test)
print(mean_squared_error(prediction1,y_test, squared=False))

ID=final_1['Id']
ID=pd.DataFrame(ID)
final_1=final_1.drop('Id',axis=1)

final_1['Condition2_RRAe']=0
final_1['Condition2_RRAn']=0
final_1['Condition2_RRNn']=0
final_1['Electrical_Mix']=0
final_1['Exterior1st_ImStucc']=0
final_1['Exterior1st_Stone']=0
final_1['Exterior2nd_Other']=0
final_1['Exterior2st_None']=0
final_1['GarageQual_Fa']=0
final_1['Heating_GasA']=0
final_1['Heating_OthW']=0
final_1['HouseStyle_2.5Fin']=0
final_1['MiscFeature_TenC']=0
final_1['PoolQC_Fa']=0
final_1['RoofMatl_CompShg']=0
final_1['RoofMatl_Membran']=0
final_1['RoofMatl_Metal']=0
final_1['RoofMatl_Roll']=0
final_1['Utilities_NoSeWa']=0

final_pred = np.exp(regressor.predict(final_1))


submission = pd.DataFrame()
submission['Id']=ID['Id']
submission['SalePrice']=final_pred
#submission=submission.iloc[:, [1,0]]
submission.to_csv('submission9.csv',index=False)



