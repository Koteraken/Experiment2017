
# coding: utf-8

# # ヒストグラム
# 
# <a href="DataForPractice2017.ipynb">「好きなアイスクリームアンケート」</a>を例に説明します。

# In[17]:

# 「#」（シャープ）以降の文字はプログラムに影響しません。
# 図やグラフを図示するためのライブラリをインポートする。
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[20]:

# 数値計算やデータフレーム操作に関するライブラリをインポートする
import numpy as np
import pandas as pd
from pandas.tools import plotting # 高度なプロットを行うツールのインポート


# In[21]:

# URL によるリソースへのアクセスを提供するライブラリをインポートする。
# import urllib # Python 2 の場合
import urllib.request # Python 3 の場合


# In[22]:

# ウェブ上のリソースを指定する
url = 'https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/icecream_chosa.txt'
# 指定したURLからリソースをダウンロードし、名前をつける。
# urllib.urlretrieve(url, 'icecream_chosa.txt') # Python 2 の場合
urllib.request.urlretrieve(url, 'icecream_chosa.txt') # Python 3 の場合


# In[23]:

# データの読み込み
df1 = pd.read_csv('icecream_chosa.txt', sep=' ', index_col=0) 


# In[24]:

df1


# In[34]:

# ヒストグラムを作成する
df1.dropna(axis=1).hist(figsize=(20,20))


# In[36]:

# 左側４列を除外してヒストグラムを作成する
df1.dropna(axis=1).iloc[:, 4:-1].hist(figsize=(20,20))


# In[ ]:



