#!/usr/bin/env python
# coding: utf-8

# In[1]:


Text="I am learning NLP"


# In[2]:


import pandas as pd
pd.get_dummies(Text.split())


# In[3]:


text=["i love NLP and i will learn NLP in 2month"]


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
vectorizer.fit(text)
vector=vectorizer.transform(text)


# In[6]:


print(vectorizer.vocabulary_)
print(vector.toarray())


# In[7]:


print(vector)


# In[8]:


get_ipython().run_line_magic('pinfo', 'CountVectorizer')


# In[9]:


df=pd.DataFrame(data=vector.toarray(), columns=vectorizer.get_feature_names_out())
df


# In[21]:


Text=["The quick brown fox jumped over the lazy dog.","The dog.","The fox"]


# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
vectorizer.fit(Text)
print(vectorizer.vocabulary_)
print(vectorizer.idf_)


# In[ ]:




