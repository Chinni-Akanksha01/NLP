#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


data=pd.read_csv(r"C:\Users\chinn\OneDrive\Documents\fake_news.csv")
data.head()


# In[15]:


data.shape


# In[16]:


data.info()


# In[17]:


data.isna().sum()


# In[18]:


data=data.drop(['id'],axis=1)


# In[19]:


data=data.fillna('')


# In[20]:


data['content']=data['author']+''+data['title']+''+data['text']


# In[21]:


data=data.drop(['title','author','text'],axis=1)


# In[22]:


data.head()


# In[24]:


data['content']=data['content'].apply(lambda x: " ".join(x.lower() for x in  x.split()))


# In[25]:


data['content']=data['content'].str.replace('[^\w\s]','')


# In[26]:


import nltk
nltk.download('stopwords')


# In[29]:


from nltk.corpus import stopwords
stop=stopwords.words('english')
data['content']=data['content'].apply(lambda x:" ".join(x for x in x.split() if x not in stop))


# In[32]:


from nltk.stem import WordNetLemmatizer
from textblob import Word
data['content']=data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['content'].head()


# In[ ]:


X=data[['content']]
y=data['label']


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=45,stratify=y)


# In[39]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[40]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfidf_vect=TfidfVectorizer(analyzer='word',token_pattern=r'\w{1,}',max_features=5000)
tfidf_vect.fit(data['content'])
xtrain_tfidf=tfidf_vect.transform(X_train['content'])
xtest_tfidf=tfidf_vec.transform(X_test['content'])


# In[ ]:




