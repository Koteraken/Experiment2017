
# coding: utf-8

# # ボックスプロットとバイオリンプロット
# 
# ボックスプロットとは何か？知らない人は右記参照→ http://excelshogikan.com/qc/qc03/boxplot.html
# 
# <a href="DataForPractice2017.ipynb">「あやめのデータ」</a>を例に説明します。

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
url = 'https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/iris.txt'
# 指定したURLからリソースをダウンロードし、名前をつける。
# urllib.urlretrieve(url, 'iris.txt') # Python 2 の場合
urllib.request.urlretrieve(url, 'iris.txt') # Python 3 の場合


# In[5]:

# データの読み込み
df1 = pd.read_csv('iris.txt', sep='\t', index_col=0) 


# In[6]:

df1


# In[7]:

# ボックスプロット（箱ひげ図）を作成する
df1.boxplot()


# In[8]:

# ボックスプロットの描き方（別法）
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(df1.iloc[:, 0:4].as_matrix().T.tolist())
ax.set_xticklabels(df1.columns[0:4], rotation=90)
plt.grid()
plt.show()


# In[9]:

# バイオリンプロット
fig = plt.figure()
ax = fig.add_subplot(111)
ax.violinplot(df1.iloc[:, 0:4].as_matrix().T.tolist())
ax.set_xticks([1, 2, 3, 4]) #データ範囲のどこに目盛りが入るかを指定する
ax.set_xticklabels(df1.columns[0:4], rotation=90)
plt.grid()
plt.show()


# In[ ]:



