
# coding: utf-8

# # 散布図
# 
# <a href="http://nbviewer.jupyter.org/github/maskot1977/ipython_notebook/blob/master/%E5%AE%9F%E7%BF%92%E7%94%A8%E3%83%86%E3%82%99%E3%83%BC%E3%82%BF2017.ipynb">「合州国の州別暴力犯罪率」</a>を例に説明します。

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

df2 = pd.read_csv('USArrests.txt', sep='\t', index_col=0) # データの読み込み


# In[5]:

df2


# In[6]:

# 散布図を描く
df2.plot(kind='scatter', x=df2.columns[2], y=df2.columns[0], grid=True)


# In[7]:

# 丸のサイズと色に意味をもたせた散布図を描く
df2.plot(kind='scatter', x=df2.columns[2], y=df2.columns[0], grid=True, c=df2.iloc[:, 3], cmap='coolwarm', 
         s=df2.iloc[:, 1], alpha=0.5)


# In[8]:

# 丸のサイズと色に意味をもたせた散布図を描く（別法）

names, name_label = df2.dropna().index, "States"
x_axis, x_label = df2.dropna()["UrbanPop"], "UrbanPop"
y_axis, y_label = df2.dropna()["Murder"], "Murder"
sizes, size_label = df2.dropna()["Assault"], "Assault" 
colors, color_label = df2.dropna()["Rape"], "Rape"

plt.figure(figsize=(15, 10))
for x, y, name in zip(x_axis, y_axis, names):
    plt.text(x, y, name, alpha=0.8, size=12)
plt.scatter(x_axis, y_axis, s=sizes, c=colors, alpha=0.5, cmap='coolwarm')
plt.colorbar(alpha=0.8)
plt.title("x=%s, y=%s, size=%s, color=%s" % (x_label, y_label, size_label, color_label))
plt.xlabel(x_label, size=12)
plt.ylabel(y_label, size=12)
plt.grid()
plt.show()


# In[ ]:



