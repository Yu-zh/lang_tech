
# coding: utf-8

# In[1]:

import pandas as pd
import pickle


# In[2]:

word_index = pickle.load("index.idx","rb")


# In[3]:

word_index = pickle.load(open("index.idx","rb"))


# In[16]:

word_df = pd.DataFrame(word_index)


# In[18]:

len(np.nan)


# In[9]:

word_df


# In[17]:

import numpy as np


# In[10]:

len_word = word_df.applymap(len)


# In[20]:

word_non_na = word_df.replace(np.nan,"")


# In[21]:

word_non_na


# In[22]:

len_word = word_non_na.applymap(len)


# In[23]:

len_word


# In[24]:

word_count_book = len_word.apply(sum(0))


# In[28]:

word_count_book = len_word.apply(np.sum, axis = 1)


# In[29]:

word_count_book


# In[66]:

book_count_word = (len_word!=0).sum(0)


# In[67]:

book_count_word


# In[34]:

type(book_count_word)


# In[35]:

book_count_word["övre"]


# In[36]:

book_number = len(word_count_book)


# In[37]:

book_number


# In[40]:

word_tf_df = len_word.div(word_count_book, axis=0)


# In[41]:

word_tf_df


# In[68]:

word_idf_ser = np.log10(book_number/book_count_word)


# In[69]:

word_idf_ser


# In[70]:

word_tf_idf = word_tf_df.mul(word_idf_ser, axis = 1)


# In[71]:

word_tf_idf


# In[72]:

word_tf_idf[["känna","gås","nils","et"]]


# In[59]:

word_idf_ser==np.inf


# In[60]:

(word_idf_ser==np.inf).sum()


# In[61]:

(book_count_word==0).sum()


# In[63]:

word_df["känna"]


# In[64]:

book_count_word["känna"]


# In[65]:

word_non_na["känna"]


# In[ ]:



