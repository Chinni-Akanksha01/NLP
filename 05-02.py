#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install emot')


# In[2]:


text1="What are you saying😂. I am the boss 😎,and why are you so 😒" 


# In[5]:


import re 
from emot.emo_unicode import UNICODE_EMOJI
from emot.emo_unicode import EMOTICONS_EMO


# In[9]:


def converting_emojis(text):
    for emot in UNICODE_EMOJI:
        text=text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","")\
                                        .replace(":","").split()))
converting_emojis(text1)


# In[ ]:


def emoji_removal(string):
    emoji_unicodes=re.compile("[]")


# In[10]:


def emoji_removal(string):
    emoji_unicodes =re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002500-\U00002BEF
                               u"\U00002702-\U000027B0
                               u"\U000024C2-\U0001F251'
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff u"\u2640-\u2642"

u"\u2600-\u2B55"

u"\u200d"
u"\u23cf"
u"\u23e9"
u"\u231a"
u"\ufe0f"
u"\u3030"
"]+",
flags=re.UNICODE)
return
emoji_unicodes.sub(r",
string)
emoji_removal(text1)


# In[1]:


import pandas as pd
dataset=pd.read_csv(r"C:\Users\chinn\OneDrive\Documents\hate_speech.csv")
dataset.head()


# In[3]:


dataset.shape


# In[5]:


dataset.label.value_counts()


# In[7]:


for iindex,tweet in enumerate(dataset["tweet"][10:5]):
    print(index+1,"-",tweet)


# In[8]:


import re
def clean_text(text):
    text=re.sub(r'[^a-zA-Z\']', ' ',text)
    text=re.sub(r'[^\x00-\x7F]+', ' ',text)
    text=text.lower()
    return text


# In[9]:


dataset['clean_text']=dataset.tweet.apply(lambda x: clean_text(x))


# In[11]:


dataset.head(10)


# In[12]:


from nltk.corpus import stopwords
len(stopwords.words('english'))


# In[13]:


stop=stopwords.words('english')


# In[14]:


def gen_freq(text):
    word_list=[]
    for tw_words in text.split():
        word_list.extend(tw_words)
    word_freq=pd.Series(word_list).value_counts()
    word_freq=word_freq.drop(stop,errors='ignore')
    return word_freq


# In[15]:


def any_neg(words):
    for word in words:
        if word in ['n','no','non','not'] or re.search(r"\wn't",word):
            return 1
        else:
            return 0


# In[16]:


def any_rte(words,rare_100):
    for word in words:
        if word in rare_100:
            return 1
        else:
            return 0


# In[17]:


def is_question(words):
    for word in words:
        if word in ['when','what','how','why','who','where']:
            return 1
        else:
            return 0


# In[ ]:


word_freq=gen_freq(datset.clean_text.str)
rare_100=word_freq[-100:]
dataset['word_count']=datset.clean_text.str.split().apply(lambda x: len(x))
dataset['any_neg']=datset.clean_text.str.split().apply(lambda x: len(x))
dataset['word_count']=datset.clean_text.str.split().apply(lambda x: len(x))
dataset['word_count']=datset.clean_text.str.split().apply(lambda x: len(x))
dataset['word_count']=datset.clean_text.str.split().apply(lambda x: len(x)

