
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

empty_matrix = pd.read_csv('..//..//data//model//empty_matrix_6H.gz')
encampments = pd.read_csv('..//..//data//sf_open//311_Cases.csv')
blocks = pd.read_csv('..//..//data//census//tl_2010_06075_tabblock10.csv')


# In[2]:


# get encampments reports
encampments = encampments[encampments['Request Type']=='Encampment Reports']
encampments = encampments[encampments['Latitude']!=0]

encampments['Opened_rnd'] = pd.to_datetime(encampments['Opened'])
encampments = encampments[encampments['Opened_rnd']>'2016-05-01']


# In[3]:


# make polygons for blocks
def make_poly(lon, lat):
    radius = Point(lon,lat).buffer(0.01)
    poly = Polygon(radius.exterior.coords)
    return poly

blocks['poly'] = blocks.apply(lambda row: make_poly(row['INTPTLON10'],row['INTPTLAT10']), axis=1)
#blocks['poly'][0].contains(Point(-122.441075,37.750066))

blocks_poly = blocks.loc[:,[

    'GEOID10',
    'poly'
]]


# In[4]:


encamp_matrix = empty_matrix.merge(blocks_poly, left_on='block_fips', right_on='GEOID10')
encamp_matrix['Opened_rnd'] = pd.to_datetime(encamp_matrix['Opened_rnd'],format='%Y-%m-%d %H:%M:%S')


# In[5]:


encamp_count = []
for index, row in encamp_matrix.iterrows():

    encamp_use = encampments[(encampments['Opened_rnd'] < row['Opened_rnd'] + datetime.timedelta(days=10))&
                            (encampments['Opened_rnd'] > row['Opened_rnd'] - datetime.timedelta(days=10))]
    adder = 0
    for index2, row2 in encamp_use.iterrows():

        if row['poly'].contains(Point(row2['Longitude'],row2['Latitude'])):
            adder += 1
    encamp_count.append(adder)


# In[ ]:


encamp_matrix['encamp_count'] = encamp_count
encamp_matrix.to_csv('..//..//data//created//encamp_feature.csv', index=False)
