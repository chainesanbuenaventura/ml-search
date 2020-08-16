#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cosine

import ElasticFunctions as ef

def updateRelatedLessons(credentials, tfidfDF):
    """Function to update related lessons
    """
    ## Load TFIDF of Lessons 

    par_tf_idf = tfidfDF

    lessonsDF = ef.getLessons(credentials)
    id_par_map = lessonsDF[["_id", "paragraph"]]

    # par_tf_idf = pd.read_csv('./data/data_version_3/related_lessons/tfidf.csv').iloc[:, 1:]
    # id_par_map = pd.read_csv('./data/data_version_3/related_lessons/lessons_ids_paragraphs.csv').iloc[:, 1:]

    par_ids = par_tf_idf["id"].values
    par_tf_idf.drop("id", axis=1, inplace=True)

    # Get Related Lessons

    def get_most_similar(matrix, index, n=5):
        return matrix.loc[index].sort_values().head(n+1).index[1:].tolist()

    def get_lessons(par_ids, id_par_map):
        for par_id in par_ids:
            print(id_par_map[id_par_map['id'] == par_id]["paragraph"].values)

    distances = pairwise_distances(par_tf_idf.iloc[:,1:], metric = 'cosine', n_jobs = -1)
    distances_df = pd.DataFrame(distances)

    related_lessons = []
    for index, par_id in enumerate(par_ids):
        related_indexes = [int(i) for i in get_most_similar(distances_df, index)]
        source_lesson_id = par_id
        related_lesson_ids = [par_ids[related_index] for related_index in related_indexes]
        related_lessons.append([source_lesson_id,related_lesson_ids])

    related_lessons_df = pd.DataFrame(related_lessons, columns=['source_lesson', 'related_lessons'])

    lessonIds = lessonsDF["_id"].tolist()
    ids = related_lessons_df["source_lesson"].tolist()
    related_lessons = related_lessons_df["related_lessons"].tolist()
    newRelatedLessons = []
    for id in lessonIds:
        newRelatedLessons.append(related_lessons[ids.index(id)])
    lessonsDF["relatedLessons"] = newRelatedLessons
    ef.updateSentences(credentials, lessonsDF)


## Unused code after code refactoring

"""
# In[19]:


related_lessons_df.to_pickle("./data/data_version_3/related_lessons/related_lessons_agg.pkl")


# # Appendix

# ## Checker if Related Lessons make sense

# In[9]:


related_lessons_df.iloc[0]['related_lessons']


# In[17]:


id_par_map[id_par_map["id"] == 2]["paragraph"].values


# In[18]:


id_par_map[id_par_map["id"] == 39768]["paragraph"].values


# In[11]:


for i in range(5):
    par_id = related_lessons_df.iloc[i]['source_lesson']
    related_ids = related_lessons_df.iloc[i]['related_lessons']
    
    print("Source Lesson")
    print(par_id)
    get_lessons([par_id], id_par_map)

    print("===Related Lessons")
    print(related_ids)
    get_lessons(related_ids, id_par_map)
    print()


# In[26]:


index = 2 
par_id = par_ids[index]
related_indexes = [int(i) for i in get_most_similar(distances_df, index)]
related_lesson_ids = [par_ids[related_index] for related_index in related_indexes]

print("Source Lesson")
get_lessons([par_id], id_par_map)

print("===Related Lessons")
get_lessons(related_lesson_ids, id_par_map)


# In[98]:


related_lessons_df.shape


# In[100]:


related_lessons_df.head()


# In[ ]:


related_lessons_df.to_pickle("./data/data_version_3/related_lessons/related_lessons_agg.pkl")


# ## OLD VERSION BELOW

# In[87]:


par_tf_idf.shape


# In[7]:


par_df = pd.read_excel('./data/lessons_ids_paragarphs.xlsx')
par_df.head()


# In[67]:


par_df.shape


# In[31]:


par_tf_idf.iloc[:,1:].head()


# In[68]:


par_tf_idf.shape


# In[32]:


distances = pairwise_distances(par_tf_idf.iloc[:,1:], metric = 'cosine', n_jobs = -1)


# In[53]:


def get_most_similar(matrix, index, n=5):
    return matrix.loc[index].sort_values().head(n+1).index[1:].tolist()

distances_df = pd.DataFrame(distances)
related_lessons = []
for i in range(par_df.shape[0]):
    related_indexes = [int(i) for i in get_most_similar(distances_df, i)]
    source_lesson_id = par_df.iloc[i].id
    related_lesson_ids = par_df.iloc[related_indexes,:].id.values
    related_lessons.append([source_lesson_id,related_lesson_ids])

related_lessons_df = pd.DataFrame(related_lessons, columns=['source_lesson', 'related_lessons'])


# In[66]:


for i in range(5,10):
    source_lesson_id = related_lessons_df.iloc[i].source_lesson
    print(par_df[par_df['id'] == source_lesson_id].paragraph.values[0])
    
    for related_id in related_lessons_df.iloc[i].related_lessons:
        print(par_df[par_df['id'] == related_id].paragraph.values[0])
    print("============================================================")


# In[80]:


related_lessons_df.to_pickle("./data/related_lessons_agg.pkl")


# In[81]:


test_df = pd.read_pickle("./data/related_lessons_agg.pkl")
test_df.head()


# In[82]:


test_df['related_lessons'][0]


# ## ====== SCRATCH ======

# In[48]:


def get_top_n_related_paragraphs(ref_df, source_vec, target_vec, n=5):
    cosine_similarities = linear_kernel(source_vec, target_vec).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-(n+2):-1]
    #exclude first one since it includes the source vec sentence
    return ref_df.iloc[related_docs_indices[1:], :]


# In[36]:


par_df = pd.read_csv('./data/paragraphs_forecasted_docx.csv').iloc[:,1:]
par_df['key'] = list(range(par_df.shape[0]))
par_df.head()


# In[54]:


par_df.shape


# In[25]:


vectorizer = TfidfVectorizer()
par_vec = vectorizer.fit_transform(par_df['text'].values)


# In[51]:


related_sentences_arr = []
for i in range(par_vec.shape[0]):
    related_df = get_top_n_related_paragraphs(par_df, par_vec[i:i+1], par_vec)
    sentence_id = par_df.iloc[i,:]['key']
    related_sentences_ids = related_df['key'].values
    related_sentences_arr.append([sentence_id, related_sentences_ids])

related_sentences_df = pd.DataFrame(np.array(related_sentences_arr), columns=["lesson_id", "related_lessons_id"])


# In[55]:


par_vec.shape


# In[52]:


related_sentences_df


# In[20]:


cosine_similarities = linear_kernel(sample_par_vec[0:1], sample_par_vec).flatten()
related_docs_indices = cosine_similarities.argsort()[:-5:-1]
cosine_similarities[related_docs_indices]


# In[21]:


related_docs_indices


# In[22]:


cosine_similarities.argsort()[:-5:-1]


# In[23]:


for i in related_docs_indices:
    print(sample_par[i])

"""