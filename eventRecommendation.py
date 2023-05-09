import pandas as pd
import numpy as np
import math
from itertools import izip
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

events_features =pd.read_csv("event_short_features.csv")
user= pd.read_csv("users_edit.csv")


################# tf-idf rating ###############

df=events_features
count= len(df) -(df==0).sum()
df_word=pd.DataFrame({ 'word':count.index , 'counts':count.values})
df_word['idf']=np.log(len(df)/df_word.counts) 
idf_values =df_word.ix[:,2]
func = lambda x: np.asarray(x) * np.asarray(idf_values)
tf_idf_1=df.apply(func, axis=1)
tf_idf_1.to_csv('tf_idf.csv')

################# user clustering ###############

km = KMeans(n_clusters=10, random_state=0).fit(user)
predict=km.predict(user)
user['cluster'] = predict
user.to_csv('user_cluster.csv')    

################# item similarity ###############

def dot_product2(v1, v2):
    return lambda v1, v2: sum(map(lambda x, y: x * y, v1, v2))


def  cosine_similarity(v1, v2):
    prod = dot_product2(v1, v2)
    len1 = math.sqrt(dot_product2(v1, v1))
    len2 = math.sqrt(dot_product2(v2, v2))
    return prod / (len1 * len2)

sim=cosine_similarity(tf_idf_1)
df_1=pd.DataFrame(sim)
df_1.to_csv('similarity.csv')    




