#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[5]:


df1=pd.read_csv(r"E:\Datasets\25-11-21 Dataset\Heart attack Prediction-classification dataset\heart.csv")


# In[6]:


df1


# In[7]:


df1.info()


# In[8]:


df1.describe()


# In[9]:


df1.isnull().sum() #checking for null values


# In[10]:


df1[df1.duplicated()] #checking for duplicates


# In[11]:


df1.drop_duplicates(keep='first',inplace=True)#removing Duplicates


# In[12]:


df1.describe()


# In[13]:


df1.info() # Initially 303 values,now 302 post removal of duplicate values 


# In[14]:


#checking for outliers-using boxplot
for column in df1:
    plt.figure(figsize=(15,15))
    df1.boxplot(["trtbps","chol","fbs","restecg","thalachh","exng","oldpeak","slp","caa","thall"])
    break


# In[75]:


#Pair Plot-relation between each and every variable is analysed 
plt.figure(figsize=(120,120))
sns.pairplot(df1)
plt.show()


# In[15]:


#correlation 
import seaborn as sb
fig, ax = plt.subplots(figsize=(15,15)) 
dataplot = sb.heatmap(df1.corr(), cmap="YlGnBu", annot=True)
plt.show()


# In[16]:


x=df1.iloc[:,:-1]
y=df1.iloc[:,-1]


# In[17]:


x


# In[18]:


y


# In[19]:


df1.to_excel("Heart duplicated datset python.xlsx")


# In[20]:


#data spliiting for train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)


# In[21]:


#feature Scaling 
from sklearn.preprocessing import StandardScaler


# In[22]:


print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[23]:


scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[24]:


x_train,x_test


# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[30]:


#Logistic regression
model=LogisticRegression() 
model.fit(x_train, y_train)
predicted=model.predict(x_test)
cm=confusion_matrix(y_test,predicted)
print("Confusion Matrix :\n", cm)
print()
print()
print("Confusion Matrix for Logistic regression:",accuracy_score(y_test,predicted)*100,"%")


# In[38]:


#svc
modelsvc=SVC()
modelsvc.fit(x_train,y_train)
predictedsvc=modelsvc.predict(x_test)
cmsvc=confusion_matrix(y_test,predictedsvc)
print("Confusion Matrix:\n",cmsvc)
print()
print()
print("The confusion matrix for SVC:",accuracy_score(y_test,predictedsvc)*100,"%")


# In[65]:



modeldt=tree.DecisionTreeClassifier(criterion="entropy",random_state=0)
modeldt.fit(x_train,y_train)
predicteddt=modeldt.predict(x_test)
cmdt=confusion_matrix(y_test,predicteddt)
print("confusion matrix:\n",cmdt)
print()
print()
print("The confusion matrix for Decision Tree:",accuracy_score(y_test,predicteddt)*100,"%")


# In[66]:


#KNN
modelknn=KNeighborsClassifier(n_neighbors=4,metric="euclidean",p=2)
modelknn.fit(x_train,y_train)
predictedknn=modelknn.predict(x_test)
cmknn=confusion_matrix(y_test,predictedknn)
print("The confusion Matrix:\n",cmknn)
print()
print()
print("The confusion Matrixfor KNN is",accuracy_score(y_test,predictedknn)*100,"%")


# In[68]:


#Random forest
from sklearn.ensemble import RandomForestClassifier


# In[73]:


#random Forest
modelrf=RandomForestClassifier(n_estimators=100,criterion="gini",random_state=0)
modelrf.fit(x_train,y_train)
predictedrf=modelrf.predict(x_test)
cmrf=confusion_matrix(y_test,predictedrf)
print("The confusion matrix:\n",cmrf)
print()
print()
print("The confusion matrix for Random forest is:",accuracy_score(y_test,predictedrf)*100,"%")


# # The SVC has the highest accuracy of 88%

# In[ ]:




