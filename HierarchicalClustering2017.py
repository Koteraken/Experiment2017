
# coding: utf-8

# # 階層的クラスタリング
# * 本実習では教師なし学習の一種である階層的クラスタリングを行ないます。
#     * 階層的クラスタリング とは何か、知らない人は下記リンク参照↓
#         * [階層的クラスタリングとは](http://image.slidesharecdn.com/140914-intel-l-141004040412-conversion-gate02/95/-12-638.jpg) 
#         * [クラスタリング (クラスター分析)](http://www.kamishima.net/jp/clustering/)
# 
# 
# <a href="http://nbviewer.jupyter.org/github/maskot1977/ipython_notebook/blob/master/%E5%AE%9F%E7%BF%92%E7%94%A8%E3%83%86%E3%82%99%E3%83%BC%E3%82%BF2017.ipynb">「都道府県別アルコール類の消費量」</a>を例に説明します。

# In[1]:

#（シャープ）以降の文字はプログラムに影響しません。
# 図やグラフを図示するためのライブラリをインポートする。
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
url = 'https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/sake_dataJ.txt'
# 指定したURLからリソースをダウンロードし、名前をつける。
# urllib.urlretrieve(url, 'sake_dataJ.txt') # Python 2 の場合
urllib.request.urlretrieve(url, 'sake_dataJ.txt') # Python 3 の場合


# In[5]:

df = pd.read_csv('sake_dataJ.txt', sep='\t', index_col=0) # データの読み込み


# In[6]:

df


# In[7]:

dfs = df.apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)


# In[8]:

dfs


# In[11]:

# metric は色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
# method も色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
#y_labels.append("1")
from scipy.cluster.hierarchy import linkage, dendrogram
result1 = linkage(dfs.iloc[:, :], 
                  #metric = 'braycurtis', 
                  #metric = 'canberra', 
                  #metric = 'chebyshev', 
                  #metric = 'cityblock', 
                  metric = 'correlation', 
                  #metric = 'cosine', 
                  #metric = 'euclidean', 
                  #metric = 'hamming', 
                  #metric = 'jaccard', 
                  #method= 'single')
                  method = 'average')
                  #method= 'complete')
                  #method='weighted')
#dendrogram(result1, labels = list(df.iloc[:, 0:1]))
plt.figure(figsize=(8, 8))
dendrogram(result1, orientation='right', labels=list(df.index), color_threshold=0.8)
plt.title("Dedrogram")
plt.xlabel("Threshold")
plt.grid()
plt.show()


# In[15]:

# metric は色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
# method も色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
#y_labels.append("1")
from scipy.cluster.hierarchy import linkage, dendrogram
result1 = linkage(dfs.T.iloc[:, :], 
                  #metric = 'braycurtis', 
                  #metric = 'canberra', 
                  #metric = 'chebyshev', 
                  #metric = 'cityblock', 
                  metric = 'correlation', 
                  #metric = 'cosine', 
                  #metric = 'euclidean', 
                  #metric = 'hamming', 
                  #metric = 'jaccard', 
                  #method= 'single')
                  method = 'average')
                  #method= 'complete')
                  #method='weighted')
#dendrogram(result1, labels = list(df.iloc[:, 0:1]))
#plt.figure(figsize=(8, 8))
dendrogram(result1, orientation='right', labels=list(df.columns), color_threshold=0.05)
plt.title("Dedrogram")
plt.xlabel("Threshold")
plt.grid()
plt.show()


# In[ ]:



