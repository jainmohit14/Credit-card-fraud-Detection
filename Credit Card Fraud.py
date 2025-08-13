#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv('creditcard.csv')


# In[4]:


print ("Shape of data:",df.shape)


# In[5]:


print("Data Info:",df.info)


# In[6]:


print("Data Description:",df.describe())


# In[8]:


print("Null Values in each columns:",df.isnull().sum())


# In[9]:


print("Class Distribution:",df['Class'].value_counts())
sns.countplot(x='Class',data=df)
plt.title("Distribution of Transaction(0=Not Fraud,1=Fraud)")
plt.show()


# In[10]:


df.groupby('Class')['Amount'].describe()


# In[11]:


plt.figure(figsize=(8,4))
sns.boxplot(x='Class', y='Amount', data=df)
plt.title("Transaction Amount by Class")
plt.show()


# In[13]:


plt.figure(figsize=(20,10))
corr=df.corr()
sns.heatmap(corr,cmap='coolwarm_r', annot=False)
plt.title("Correlation Matrix")
plt.show()


# In[18]:


# Step 1: Define Features (X) and Target (y)
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# In[19]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# In[20]:


y_pred = model.predict(X_test)


# In[21]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))


# In[22]:


model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)


# In[23]:


y_pred = model.predict(X_test)


# In[24]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))


# In[ ]:




