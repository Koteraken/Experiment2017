
# coding: utf-8

# # 折れ線グラフ
# 
# <a href="DataForPractice2017.ipynb">「ニューヨークの大気状態観測値」</a>を例に説明します。

# In[1]:

# 「#」（シャープ）以降の文字はプログラムに影響しません。
# 図やグラフを図示するためのライブラリをインポートする。
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

# 数値計算やデータフレーム操作に関するライブラリをインポートする
import numpy as np
import pandas as pd
from pandas.tools import plotting # 高度なプロットを行うツールのインポート


# In[3]:

# URL によるリソースへのアクセスを提供するライブラリをインポートする。
# import urllib # Python 2 の場合
import urllib.request # Python 3 の場合


# In[4]:

# ウェブ上のリソースを指定する
url = 'https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/airquality.txt'
# 指定したURLからリソースをダウンロードし、名前をつける。
# urllib.urlretrieve(url, 'airquality.txt') # Python 2 の場合
urllib.request.urlretrieve(url, 'airquality.txt') # Python 3 の場合


# In[5]:

# データの読み込み
df1 = pd.read_csv('airquality.txt', sep='\t', index_col=0) 


# In[6]:

df1


# In[7]:

# 折れ線グラフを作成する
df1.plot()


# In[8]:

# 左側４列だけを用いて折れ線グラフを作成する
df1.iloc[:, 0:4].plot()


# In[9]:

# 折れ線グラフの描き方（別法）
plt.figure(figsize=(8, 8))

plt.subplot(4, 1, 1)
plt.plot(df1.iloc[:, 0])
plt.xticks([])
plt.ylabel(df1.columns[0])

plt.subplot(4, 1, 2)
plt.plot(df1.iloc[:, 1])
plt.xticks([])
plt.ylabel(df1.columns[1])

plt.subplot(4, 1, 3)
plt.plot(df1.iloc[:, 2])
plt.xticks([])
plt.ylabel(df1.columns[2])

plt.subplot(4, 1, 4)
plt.plot(df1.iloc[:, 3])
plt.xticks(rotation=90)
plt.ylabel(df1.columns[3])


# In[ ]:



