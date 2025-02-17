#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv(r"C:\Users\chinn\OneDrive\Documents\emotion.csv")


# In[2]:


df.shape


# In[3]:


df.info()


# In[4]:


df.label.value_counts()


# In[5]:


import seaborn as sns
sns.countplot(x=df.label)


# In[6]:


df.isna().sum()


# In[7]:


df['text']=df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[8]:


import nltk
nltk.download('stopwords')


# In[10]:


from nltk.corpus import stopwords
stop=stopwords.words('english')
df['text']=df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[11]:


from nltk.stem import WordNetLemmatizer
from textblob import Word
df['text']=df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df['text'].head()


# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
X=tfidf.fit_transform(df['text'])
X=X.toarray()
y=df.label.values


# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2,shuffle=True)


# In[14]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model=model.fit(X_train,y_train)
pred=model.predict(X_test)


# In[15]:


from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test,pred)*100,"%")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier()
clf_rf.fit(X_train,y_train)
rf_pred=clf_rf.predict(X_test).astype(int)


# In[ ]:


print("Accuracy:",accuracy_score(y_test,rf_pred))


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,rf_pred))
print(classification_report(y_test,rf_pred))


# In[ ]:


from sklearn.linear_model import confusion_matrix LogisticRegression
logreg=LogisticRegression(class_weight='balanced')
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

