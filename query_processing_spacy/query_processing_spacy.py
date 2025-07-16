#!/usr/bin/env python
# coding: utf-8

# # setup

# In[1]:


import os
import numpy as np
import string

try: import spacy
except ImportError:
  os.system('pip install -U spacy')
  os.system('python -m spacy download en_core_web_sm')
  import spacy


# In[2]:


nlp = spacy.load('en_core_web_sm')


# In[3]:


example = 'Have Apple stocks risen 40 million and what about Google stock?'


# # spacy fns
# lemmatization, entity extraction, noun phrase extraction

# In[4]:


# extract lemmas from text

def get_lemmas(text):
  doc = nlp(text)
  return [token.lemma_ for token in doc]


# In[5]:


get_lemmas(example)


# In[6]:


# extract entities from text

def get_ents(text):
  doc = nlp(text)
  return [ent.text for ent in doc.ents]


# In[7]:


get_ents(example)


# In[8]:


# extract noun phrases from text

def get_noun_phrases(text):
  doc = nlp(text)
  return [chunk.text for chunk in doc.noun_chunks]


# In[9]:


get_noun_phrases(example)


# # other fns
# cleaning text, filtering

# In[10]:


# lower cases, strips, and removes punctuation from text

def clean(text):
  return text.lower().strip().translate(str.maketrans('', '', string.punctuation))


# In[11]:


# filters a text only for tokens that are in the filter text OR have a certain part of speech

def filter_include(text, filter_txt, filter_pos):
  doc = nlp(text)
  return [token.text for token in doc if any(token.text in ft for ft in filter_txt) or token.pos_ in filter_pos]
  #                                      ^ in filter text                              ^ certain part of speech


# # query editing
# query decomposition, query extraction

# In[12]:


# query decomposition
# splits query by clauses

def query_split(query):
  doc = nlp(query)

  # coordinating and subordinate conjunctions
  conjs = ['CCONJ', 'SCONJ']

  result = []
  section = []
  for token in doc:
    if token.pos_ in conjs:
      if section: # make sure section is not empty
        result.append(' '.join(section))
        section = []
    else:
      section.append(token.text)

  if section:
    result.append(' '.join(section)) # append final part

  return result


# In[13]:


query_split(example)


# In[14]:


# query extraction
# takes lemmas and filters for entities and important parts of speech

def query_extract(query):
  lemmas = get_lemmas(query)
  ents = get_ents(query)

  important_text = ents
  important_pos = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']

  lemmas = list(dict.fromkeys(lemmas)) # eliminates redundant elements
  lemmas = filter_include(clean(str(lemmas)),
                          important_text,
                          important_pos) # filter

  return lemmas


# In[15]:


query_extract(example)


# In[16]:


# example with both fns

for query in query_split(example):
  print(' '.join(query_extract(query)))


# # master fn
# full query transformation
# 

# In[17]:


# full query transformation

def query_transform(query = None):
  '''
  does full query transformation from user prompt to searcheable retrieval query

  Args:
    query (string, optional): The query to be processed. Will prompt user if not given or None
  Returns:
    list: A list of queries mentioned in the user's prompt, each lemmatized and extracted for entities, nouns, verbs, and modifiers
  
  '''
    
  query = input("Input query: ") if query is None else query
    
  transform_query = []
  for sub_query in query_split(query):
    transform_query.append(' '.join(query_extract(sub_query)))
      
  return transform_query


# In[18]:


query_transform(example)

