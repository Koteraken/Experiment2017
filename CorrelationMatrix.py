
# coding: utf-8

# # 相関行列
# 
# <a href="DataForPractice2017.ipynb">「新国民生活指標データ」</a>を例に説明します。

# In[38]:

# 数値計算やデータフレーム操作に関するライブラリをインポートする
import numpy as np
import pandas as pd
from pandas.tools import plotting # 高度なプロットを行うツールのインポート


# In[39]:

# 「#」（シャープ）以降の文字はプログラムに影響しません。
# 図やグラフを図示するためのライブラリをインポートする。
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().magic('matplotlib inline')


# In[40]:

# URL によるリソースへのアクセスを提供するライブラリをインポートする。
# import urllib # Python 2 の場合
import urllib.request # Python 3 の場合


# In[41]:

url = 'https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/PLIlive_dataJ.txt'
# 指定したURLからリソースをダウンロードし、名前をつける。
# urllib.urlretrieve(url, 'PLIlive_dataJ.txt') # Python 2 の場合
urllib.request.urlretrieve(url, 'PLIlive_dataJ.txt') # Python 3 の場合


# In[42]:

# データの読み込み
df = pd.read_csv('PLIlive_dataJ.txt', sep='\t', index_col=0) 


# In[43]:

df


# In[44]:

df.T


# In[45]:

# 相関行列
pd.DataFrame(np.corrcoef(df.dropna().iloc[:, :].as_matrix().tolist()), 
             columns=df.index, index=df.index)


# In[59]:

corrcoef = np.corrcoef(df.dropna().iloc[:, :].as_matrix().tolist())
fig = plt.figure(figsize=(12, 10))
plt.imshow(corrcoef, interpolation='nearest', cmap=plt.cm.coolwarm)
plt.colorbar()
tick_marks = np.arange(len(corrcoef))
plt.xticks(tick_marks, df.index, rotation=90)
plt.yticks(tick_marks, df.index)
plt.tight_layout()


# In[51]:

# 相関行列
pd.DataFrame(np.corrcoef(df.dropna().iloc[:, :].T.as_matrix().tolist()), 
             columns=df.columns, index=df.columns)


# In[60]:

corrcoef = np.corrcoef(df.dropna().iloc[:, :].T.as_matrix().tolist())
plt.figure(figsize=(12, 10))
plt.imshow(corrcoef, interpolation='nearest', cmap=plt.cm.coolwarm)
plt.colorbar()
tick_marks = np.arange(len(corrcoef))
plt.xticks(tick_marks, df.columns, rotation=90)
plt.yticks(tick_marks, df.columns)
plt.tight_layout()


# In[ ]:



