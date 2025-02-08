#!/usr/bin/env python
# coding: utf-8

# In[29]:


from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegressionCV
import re
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[30]:


df=pd.read_csv(r"C:\Users\chinn\OneDrive\Documents\covid_fake.csv")


# In[31]:


df.head()


# In[32]:


df.shape


# In[33]:


df['label'].value_counts()


# In[34]:


df.loc[5:15]


# In[35]:


df.isna().sum()


# In[36]:


df.loc[df['label']=='Fake',['label']]='Fake'
df.loc[df['label']=='fake',['label']]='Fake'
df.loc[df['source']=='facebook',['source']]='Facebook'
df.text.fillna(df.title,inplace=True)
df.loc[5]['label']='Fake'
df.loc[15]['label']='TRUE'
df.loc[43]['label']='FAKE'
df.loc[131]['label']='TRUE'
df.loc[242]['label']='FAKE'
df.title.fillna('missing', inplace=True)  # Fixed incorrect string syntax

df.source.fillna('missing',inplace=True)
df['title_text']=df['title']+''+df['text']




# In[37]:


df.isna().sum()


# In[38]:


df['label'].value_counts()


# In[39]:


df.head()


# In[40]:


df.shape


# In[41]:


df['title_text'][3]


# In[42]:


#*-0 or more occurences
def preprocessor(text):
    text=re.sub('<[^>]*>','',text)
    text=re.sub(r'[^\w\s]','',text)
    text=re.sub(r'[\n]','',text)
    text=text.lower()
    return text
df['title_text'] =df['title_text'].apply(preprocessor)
df['title_text'][3]
    


# In[43]:


porter=PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[45]:


tfidf=TfidfVectorizer(strip_accents=None,
                     lowercase=False,
                     preprocessor=None,
                     tokenizer=tokenizer_porter,
                     use_idf=True,
                     norm='l2',
                     smooth_idf=True)
X=tfidf.fit_transform(df['title_text'])
y=df.label.values


# In[46]:


X.shape


# In[48]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.3,shuffle=False)


# In[49]:


clf=LogisticRegressionCV(cv=5,scoring='accuracy',random_state=0,n_jobs=-1,verbose=0,max_iter=300)
clf.fit(X_train,y_train)
fake_news_model=open('fake_news_model.sav','wb')
pickle.dump(clf,fake_news_model)
fake_news_model.close()


# In[51]:


filename='fake_news_model.sav'
saved_clf=pickle.load(open(filename,'rb'))
saved_clf.score(X_test,y_test)


# In[52]:


from sklearn.metrics import classification_report, accuracy_score
y_pred=clf.predict(X_test)
print("---Test Set Results---")
print(classification_report(y_test,y_pred))


# In[ ]:




