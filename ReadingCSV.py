
# coding: utf-8

# # タブ区切りデータ、コンマ区切りデータ等の読み込み

# In[1]:

# 数値計算やデータフレーム操作に関するライブラリをインポートする
import numpy as np
import pandas as pd


# In[2]:

# URL によるリソースへのアクセスを提供するライブラリをインポートする。
# import urllib # Python 2 の場合
import urllib.request # Python 3 の場合


# __「ニューヨークの大気状態観測値」__のデータを読み込んでみましょう。
# （<a href="DataForPractice2017.ipynb">詳細</a>）

# In[3]:

# ウェブ上のリソースを指定する
url = 'https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/airquality.txt'
# 指定したURLからリソースをダウンロードし、名前をつける。
# urllib.urlretrieve(url, 'airquality.txt') # Python 2 の場合
urllib.request.urlretrieve(url, 'airquality.txt') # Python 3 の場合


# In[4]:

# データの読み込み
df1 = pd.read_csv('airquality.txt', sep='\t', index_col=0) 


# In[5]:

# 読み込んだデータの確認
df1


# うまく読み込めました。

# 次に、__「好きなアイスクリームアンケート」__のデータを読み込んでみましょう。
# （<a href="DataForPractice2017.ipynb">詳細</a>）

# In[6]:

# ウェブ上のリソースを指定する
url = 'https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/icecream_chosa.txt'
# 指定したURLからリソースをダウンロードし、名前をつける。
# urllib.urlretrieve(url, 'icecream_chosa.txt') # Python 2 の場合
urllib.request.urlretrieve(url, 'icecream_chosa.txt') # Python 3 の場合


# In[7]:

# データの読み込み
df2 = pd.read_csv('icecream_chosa.txt', sep='\t', index_col=0) 


# In[8]:

df2


# 上の方法では、「好きなアイスクリームアンケート」のデータがうまく読み込めていません。原因は、実際のデータ区切り文字が「 」（空白）なのに、データの読み込み時に「sep='\t'」（データ区切り文字はタブ）と指定したからです。では改めて、データ区切り文字に空白を指定して読み込んでみましょう。

# In[9]:

# データの読み込み
df2 = pd.read_csv('icecream_chosa.txt', sep='\s+', index_col=0) 


# In[10]:

df2


# うまく読み込めました。
# 
# 次に、__「ワインの品質」__のデータを読み込んでみましょう。
# （<a href="DataForPractice2017.ipynb">詳細</a>）

# In[11]:

# ウェブ上のリソースを指定する
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
# 指定したURLからリソースをダウンロードし、名前をつける。
# urllib.urlretrieve(url, 'winequality-red.csv') # Python 2 の場合
urllib.request.urlretrieve(url, 'winequality-red.csv') # Python 3 の場合


# In[12]:

# データの読み込み
df3 = pd.read_csv('winequality-red.csv', sep='\t', index_col=0) 


# In[13]:

df3


# 上の方法では、「ワインの品質」のデータがうまく読み込めていません。原因は、実際のデータ区切り文字が「;」（セミコロン）なのに、データの読み込み時に「sep='\t'」（データ区切り文字はタブ）と指定したからです。では改めて、データ区切り文字にセミコロンを指定して読み込んでみましょう。

# In[14]:

# データの読み込み
df3 = pd.read_csv('winequality-red.csv', sep=';', index_col=0) 


# In[15]:

df3


# うまく読み込めたように見えるかもしれませんが、不十分です。第１列目（いちばん左）のデータが、インデックス番号として取り扱われています。このデータにはインデックス番号が指定されていませんので、次のようにして読み込みましょう。

# In[16]:

# データの読み込み
df3 = pd.read_csv('winequality-red.csv', sep=';') 


# In[17]:

df3


# うまく読み込めました。
# 
# 次は、__「あわびのデータ」__のデータを読み込んでみましょう。
# （<a href="DataForPractice2017.ipynb">詳細</a>）

# In[18]:

# ウェブ上のリソースを指定する
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
# 指定したURLからリソースをダウンロードし、名前をつける。
# urllib.urlretrieve(url, 'abalone.data') # Python 2 の場合
urllib.request.urlretrieve(url, 'abalone.data') # Python 3 の場合


# In[19]:

# データの読み込み
df4 = pd.read_csv('abalone.data', sep='\t', index_col=0) 


# In[20]:

df4


# 上の方法では、「あわびのデータ」がうまく読み込めていません。原因は、実際のデータ区切り文字が「,」（コンマ）なのに、データの読み込み時に「sep='\t'」（データ区切り文字はタブ）と指定したからです。では改めて、データ区切り文字にコンマを指定して読み込んでみましょう。

# In[21]:

# データの読み込み
df4 = pd.read_csv('abalone.data', sep=',', index_col=0) 


# In[22]:

df4


# うまく読み込めたように見えるかもしれませんが、不十分です。第１列目（いちばん左）のデータが、インデックス番号として取り扱われています。このデータにはインデックス番号が指定されていませんので、次のようにして読み込みましょう。

# In[23]:

# データの読み込み
df4 = pd.read_csv('abalone.data', sep=',') 


# In[24]:

df4


# うまく読み込めたように見えるかもしれませんが、不十分です。第１行目（いちばん上）のデータが、ヘッダ行として取り扱われています。このデータにはヘッダ行が指定されていませんので、次のようにして読み込みましょう。

# In[25]:

# データの読み込み
df4 = pd.read_csv('abalone.data', sep=',', header=None) 


# In[26]:

df4


# これで、うまくデータを読み込めました。

# In[ ]:



