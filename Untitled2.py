#!/usr/bin/env python
# coding: utf-8

# In[16]:


get_ipython().system('pip install pytextrank')


# In[19]:


get_ipython().system('pip install -U spacy --user')


# In[20]:


get_ipython().system('pip install --upgrade spacy==3.4.0 --user')


# In[6]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[7]:


import spacy
import pytextrank


# In[8]:


document="""Not only did it only confirm that the film would be unfunny and generic,but it also managed to give away the Entire movie;
and I'm not exaggerating - every plot point,every joke is told in the trailer."""


# In[9]:


en_nlp=spacy.load("en_core_web_sm")
en_nlp.add_pipe("textrank")
doc=en_nlp(document)


# In[10]:


tr=doc._.textrank
print(tr.elapsed_time)


# In[11]:


for combination in doc._.phrases:
    print(combination.text,combination.rank,combination.count)


# In[13]:


from bs4 import BeautifulSoup
from urllib.request import urlopen


# In[12]:


def get_only_text(url):
    page=urlopen(url)
    soup=BeautifulSoup(page)
    text='\t'.join(map(lambda p: p.text,soup.find_all('p')))
    print(text)
    return soup.title.text,text


# In[13]:


url="https://en.wikipedia.org/wiki/Natural_language_processing"
text=get_only_text(url)


# In[18]:


len(''.join(text))


# In[19]:


text[:1000]


# In[20]:


get_ipython().system('pip install sumy')


# In[21]:


get_ipython().system('pip install lxml.html.clean')


# In[1]:


from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.luhn import LuhnSummarizer


# In[5]:


import nltk
nltk.download('punkt')


# In[8]:


LANGUAGE="english"
SENTENCES_COUNT=10
url="https://en.wikipedia.org/wiki/Natural language processing"
parser=HtmlParser.from_url(url,Tokenizer(LANGUAGE))
summarizer=LsaSummarizer()
summarizer=LsaSummarizer(Stemmer(LANGUAGE))
for sentence in summarizer(parser.document,SENTENCES_COUNT):
    print(sentence)


# In[ ]:


text="""A vaccine for the coronavirus will likely be ready by early 2021 but rolling it out safely acr India,which is host to some of the front-runner vaccine clinical trials,currently has no local infrastructure in lace to go beyond immunizing babies and pregnant women,said Gagandeep kang,professor of microbiology at the vellor-based christian medical college and a member.


# In[7]:


import nltk
nltk.download('punkt_tab')


# In[10]:


from sumy.parsers.plaintext import PlaintextParser 
from sumy.nlp.tokenizers import Tokenizer


# In[11]:


parser = PlaintextParser.from_string(text, Tokenizer("english"))


# In[14]:


from sumy.summarizers.lex_rank import LexRankSummarizer 
from sumy.utils import get_stop_words
summarizer_lex = LexRankSummarizer()


# In[15]:


from sumy.summarizers.lex_rank import LexRankSummarizer 
from sumy.utils import get_stop_words
summarizer_lex = LexRankSummarizer()
summarizer_lex.stop_words = get_stop_words("english")
summary= summarizer_lex(parser.document, 5)
lex_summary=""
for sentence in summary:
    lex_summary += str(sentence)
print (lex_summary)


# In[ ]:




