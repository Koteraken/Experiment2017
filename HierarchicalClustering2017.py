
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


# ### 行列の正規化
# 
# 行列の正規化を行います。行列の正規化について、詳しくは<a href="UsingNumpyAndPandas.ipynb" target="_blank">Numpy と Pandas を用いた演算</a>を参照のこと。

# In[7]:

# 行列の正規化
dfs = df.apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)


# In[8]:

# 正規化を終えた行列
dfs


# ### クラスタリングについて
# クラスタリングには色んな metric と method があるので、その中から一つを選択する。
# * Metric （「距離」の定義）
#     * braycurtis
#     * canberra
#     * chebyshev
#     * cityblock
#     * correlation
#     * cosine
#     * euclidean
#     * hamming
#     * jaccard
# * Method （結合方法）
#     * single
#     * average
#     * complete
#     * weighted
# 
# 詳しくは、例えば <a href="http://qiita.com/sue_charo/items/20bad5f0bb2079257568" target="_blank">クラスタリング手法のクラスタリング</a>などを参照のこと。

# ### クラスタリングの実行
# Metricからどれかひとつ、Methodからどれかひとつ選んで実行する。エラーが起こったり、あるいは「計算はできたけど、なんかこの結果はしっくり来ないな（解釈しづらいな）」と思ったら、別の方法を試す。

# In[9]:

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


# 同じデータを用いて、別の方法を実行した結果を以下に示します。どの方法がベストかはケースバイケースで、データの性質をよく考えた上で解釈しなければいけません。

# In[10]:

# metric は色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
# method も色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
#y_labels.append("1")
from scipy.cluster.hierarchy import linkage, dendrogram
result1 = linkage(dfs.iloc[:, :], 
                  #metric = 'braycurtis', 
                  #metric = 'canberra', 
                  #metric = 'chebyshev', 
                  #metric = 'cityblock', 
                  #metric = 'correlation', 
                  #metric = 'cosine', 
                  metric = 'euclidean', 
                  #metric = 'hamming', 
                  #metric = 'jaccard', 
                  #method= 'single')
                  #method = 'average')
                  #method= 'complete')
                  method='weighted')
#dendrogram(result1, labels = list(df.iloc[:, 0:1]))
plt.figure(figsize=(8, 8))
dendrogram(result1, orientation='right', labels=list(df.index), color_threshold=0.8)
plt.title("Dedrogram")
plt.xlabel("Threshold")
plt.grid()
plt.show()


# ### 行と列を入れ替えてクラスタリング
# 行列の転置を行えば、行と列を入れ替えたクラスタリングが行えます。行列の転置について、詳しくは<a href="UsingNumpyAndPandas.ipynb" target="_blank">Numpy と Pandas を用いた演算</a>を参照のこと。

# In[11]:

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


# In[12]:

# metric は色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
# method も色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
#y_labels.append("1")
from scipy.cluster.hierarchy import linkage, dendrogram
result1 = linkage(dfs.T.iloc[:, :], 
                  #metric = 'braycurtis', 
                  #metric = 'canberra', 
                  #metric = 'chebyshev', 
                  #metric = 'cityblock', 
                  #metric = 'correlation', 
                  #metric = 'cosine', 
                  metric = 'euclidean', 
                  #metric = 'hamming', 
                  #metric = 'jaccard', 
                  #method= 'single')
                  #method = 'average')
                  #method= 'complete')
                  method='weighted')
#dendrogram(result1, labels = list(df.iloc[:, 0:1]))
#plt.figure(figsize=(8, 8))
dendrogram(result1, orientation='right', labels=list(df.columns), color_threshold=0.05)
plt.title("Dedrogram")
plt.xlabel("Threshold")
plt.grid()
plt.show()


# 上記の例では、方法を変えても同じような形のデンドログラムが生じたので、都道府県ごとの消費量を元にした「酒類間の類似関係」はクラスタリング方法の違いによらず一定の解釈ができそうである。

# In[ ]:



