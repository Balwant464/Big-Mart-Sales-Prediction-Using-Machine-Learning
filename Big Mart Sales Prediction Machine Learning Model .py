#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df_Train = pd.read_csv(r'C:\Users\Lenovo\Downloads\archive (2)\Train.csv')
df_Test = pd.read_csv(r'C:\Users\Lenovo\Downloads\archive (2)\Test.csv')


# In[4]:


df_Train.head()


# In[5]:


df_Train.shape


# In[6]:


df_Train.isnull().sum()


# In[7]:


df_Test.isnull().sum()


# In[8]:


df_Train.info()


# In[9]:


df_Train.describe()


# In[11]:


df_Train['Item_Weight'].describe()


# In[13]:


df_Train['Item_Weight'].fillna(df_Train['Item_Weight'].mean(),inplace=True)
df_Test['Item_Weight'].fillna(df_Test['Item_Weight'].mean(),inplace=True)


# In[14]:


df_Train.isnull().sum()


# In[15]:


df_Train['Item_Weight'].describe()


# In[16]:


df_Train['Outlet_Size'].value_counts()


# In[17]:


df_Train['Outlet_Size'].mode()


# In[18]:


df_Train['Outlet_Size'].fillna(df_Train['Outlet_Size'].mode()[0],inplace=True)
df_Test['Outlet_Size'].fillna(df_Test['Outlet_Size'].mode()[0],inplace=True)


# In[19]:


df_Train.isnull().sum()


# In[20]:



df_Test.isnull().sum()


# In[21]:


df_Train.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
df_Test.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)


# In[22]:


df_Train


# In[24]:



import klib


# In[25]:


klib.cat_plot(df_Train) 


# In[29]:


klib.missingval_plot(df_Train)


# In[30]:


klib.dist_plot(df_Train) 


# In[32]:


klib.data_cleaning(df_Train)


# In[34]:


klib.clean_column_names(df_Train)


# In[35]:


df_Train.info()


# In[37]:


df_Train=klib.convert_datatypes(df_Train)
df_Train.info()


# In[38]:


klib.mv_col_handling(df_Train)


# In[39]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[41]:


df_Train['item_fat_content']= le.fit_transform(df_Train['item_fat_content'])
df_Train['item_type']= le.fit_transform(df_Train['item_type'])
df_Train['outlet_size']= le.fit_transform(df_Train['outlet_size'])
df_Train['outlet_location_type']= le.fit_transform(df_Train['outlet_location_type'])
df_Train['outlet_type']= le.fit_transform(df_Train['outlet_type'])


# In[42]:


df_Train


# In[43]:


X=df_Train.drop('item_outlet_sales',axis=1)


# In[44]:


Y=df_Train['item_outlet_sales']


# In[50]:


from sklearn.model_selection import train_test_split

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, random_state=101, test_size=0.2)


# In[51]:


X.describe()


# In[52]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()


# In[54]:


X_Train_std= sc.fit_transform(X_Train)


# In[55]:


X_Test_std= sc.transform(X_Test)


# In[56]:


X_Train_std


# In[57]:


X_Test_std


# In[58]:


Y_Train


# In[59]:


Y_Test


# In[62]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()


# In[64]:


lr.fit(X_Train_std,Y_Train)


# In[65]:


X_Test.head()


# In[67]:


Y_pred_lr=lr.predict(X_Test_std)


# In[68]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[70]:


print(r2_score(Y_Test,Y_pred_lr))
print(mean_absolute_error(Y_Test,Y_pred_lr))
print(np.sqrt(mean_squared_error(Y_Test,Y_pred_lr)))


# In[71]:


from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor(n_estimators=1000)


# In[72]:


rf.fit(X_Train_std,Y_Train)


# In[73]:


Y_pred_rf= rf.predict(X_Test_std)


# In[74]:


print(r2_score(Y_Test,Y_pred_rf))
print(mean_absolute_error(Y_Test,Y_pred_rf))
print(np.sqrt(mean_squared_error(Y_Test,Y_pred_rf)))


# In[75]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# define models and parameters
model = RandomForestRegressor()
n_estimators = [10, 100, 1000]
max_depth=range(1,31)
min_samples_leaf=np.linspace(0.1, 1.0)
max_features=["auto", "sqrt", "log2"]
min_samples_split=np.linspace(0.1, 1.0, 10)

# define grid search
grid = dict(n_estimators=n_estimators)

#cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=101)

grid_search_forest = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, 
                           scoring='r2',error_score=0,verbose=2,cv=2)

grid_search_forest.fit(X_Train_std, Y_Train)

# summarize results
print(f"Best: {grid_search_forest.best_score_:.3f} using {grid_search_forest.best_params_}")
means = grid_search_forest.cv_results_['mean_test_score']
stds = grid_search_forest.cv_results_['std_test_score']
params = grid_search_forest.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:.3f} ({stdev:.3f}) with: {param}")


# In[91]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_Test, Y_pred_rf_grid)
print("Accuracy: {:.2%}".format(accuracy))


# In[91]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_Test, Y_pred_rf_grid)
print("Accuracy: {:.2%}".format(accuracy))


# In[76]:


grid_search_forest.best_params_


# In[77]:


grid_search_forest.best_score_


# In[79]:


Y_pred_rf_grid=grid_search_forest.predict(X_Test_std)


# In[80]:


r2_score(Y_Test,Y_pred_rf_grid)


# In[81]:


import joblib


# In[85]:


joblib.dump(grid_search_forest,r'C:\Users\Lenovo\Downloads\final model\models\random_forest_grid.sav')


# In[86]:


model=joblib.load(r'C:\Users\Lenovo\Downloads\final model\models\random_forest_grid.sav')


# In[ ]:




