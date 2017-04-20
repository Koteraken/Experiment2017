
# coding: utf-8

# # 散布図行列
# 
# <a href="DataForPractice2017.ipynb">「スポーツテストデータ」</a>を例に説明します。

# In[9]:

# 数値計算やデータフレーム操作に関するライブラリをインポートする
import numpy as np
import pandas as pd
from pandas.tools import plotting # 高度なプロットを行うツールのインポート


# In[10]:

# 「#」（シャープ）以降の文字はプログラムに影響しません。
# 図やグラフを図示するためのライブラリをインポートする。
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().magic('matplotlib inline')


# In[11]:

# URL によるリソースへのアクセスを提供するライブラリをインポートする。
# import urllib # Python 2 の場合
import urllib.request # Python 3 の場合


# In[24]:

# ウェブ上のリソースを指定する
url = 'https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/sports_dataJt.txt'
# 指定したURLからリソースをダウンロードし、名前をつける。
# urllib.urlretrieve(url, 'sports_dataJt.txt') # Python 2 の場合
urllib.request.urlretrieve(url, 'sports_dataJt.txt') # Python 3 の場合


# In[27]:

# データの読み込み
df = pd.read_csv('sports_dataJt.txt', sep='\t', index_col=0) 


# In[28]:

df


# In[19]:

df.columns[4:14]


# In[30]:

# 散布図行列（左側の４列だけ）
plotting.scatter_matrix(df.dropna(axis=1)[df.columns[0:4]], figsize=(8, 8)) 
plt.show()


# In[32]:

# 散布図行列
plotting.scatter_matrix(df.dropna(axis=1)[df.columns[:]], figsize=(20, 20)) 
plt.show()


# In[ ]:



