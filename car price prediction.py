#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
data=pd.read_csv(r"C:\Users\RUCHITHA\Desktop\car data.csv")
df=pd.DataFrame(data)
df


# In[77]:


df.columns


# In[78]:


data.isnull().sum()


# In[79]:


df.corr()


# In[80]:


df.dtypes


# In[81]:


df.select_dtypes(include="object").columns


# In[82]:


df.select_dtypes(include="float64").columns


# In[83]:


df.select_dtypes(include="int64").columns


# In[84]:


df.describe()


# In[85]:


df=df.drop(columns='Car_Name')


# In[86]:


df['years']=2022


# In[87]:


df


# In[88]:


df['years old']=df['years']-df['Year']
df


# In[89]:


df.drop(columns=["years","Year","Owner","Transmission","Seller_Type"],inplace=True)
df


# In[90]:


plt.scatter(df["years old"],df["Selling_Price"],c="black")
plt.xlabel("years old",c="red")
plt.ylabel("selling price",c="red")


# In[96]:


df=pd.get_dummies(data=df,drop_first=True)
df


# In[97]:


import matplotlib.pyplot as plt
plt.scatter(df['Present_Price'],df["Selling_Price"],c="black")
plt.xlabel('Selling_Price',c="red")
plt.ylabel('Present_Price',c="red")
plt.show()


# In[99]:


#encoding the categorical data
df=pd.get_dummies(data=df,drop_first=True)


# In[100]:


df


# In[101]:


df.shape


# In[102]:


#matrix of features
x=df.drop(columns='Selling_Price')


# In[103]:


#target variable
y=df['Selling_Price']


# In[104]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[105]:


x_train.shape


# In[106]:


x_test.shape


# In[107]:


#building a model
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(x_train,y_train)


# In[108]:


y_pred=model.predict(x_test)


# In[109]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[110]:


#predicting a single observation
df.head()


# In[111]:


a=input("present price:")
b=input("kms driven:")
c=input("years old:")
d=input("fuel type diesel:")
e=input("fuel type petrol:")
pre=[[a,b,c,d,e]]
x=model.predict(pre)
print("selling price of car:",x)


# 
