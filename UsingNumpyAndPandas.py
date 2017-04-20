
# coding: utf-8

# # Numpy と Pandas を用いた演算

# In[1]:

# 数値計算やデータフレーム操作に関するライブラリをインポートする
import numpy as np
import pandas as pd


# ## まずは Pandas の基本操作から

# In[2]:

df = pd.DataFrame([[1,4,7,10,13,16],[2,5,8,11,14,17],[3,6,9,12,15,18],[21,24,27,20,23,26]],
                   index = ['i1','i2','i3', 'i4'],
                   columns = list("abcdef"))


# In[3]:

df


# In[4]:

# インデックス名を指定した行の取り出し。
df.ix['i1']


# In[5]:

# インデックス番号を指定した行の取り出し
df.ix[1]


# In[6]:

# インデックス番号を指定した行の取り出し
df.ix[1:] # （ゼロスタートで）１行目以降を取り出す


# <font color="#ff0000">日常的な感覚では「一番最初」を示すのに「１」を使いますが、Pythonを含む多くのプログラミング言語では「一番最初」を示すのに「０」（ゼロ）を使います。<b>[1:]</b>で、「（ゼロスタートで）１行目以降全て」＝「（日常的な感覚で）２行目以降全て」を指していることに注意してください。</font>

# In[7]:

# インデックス番号を指定した行の取り出し
df.ix[:1] # （ゼロスタートで）１行目より手前を取り出す


# <font color="#ff0000"><b>[:1]</b>で、「（ゼロスタートで）１行目より手前全て」を示します。「（ゼロスタートで）１行目」は含まれないことに注意してください。</font>

# In[8]:

# インデックス番号を指定した行の取り出し
df.ix[1:3] # （ゼロスタートで）１行目から、３行目の手前までを取り出す


# <font color="#ff0000"><b>[1:3]</b>で、「（ゼロスタートで）１行目以降〜３行目手前まで」を示します。「（ゼロスタートで）３行目」は含まれないことに注意してください。</font>

# In[9]:

# 一つ目のパラメータで行を、二つ目のパラメータで列を指定して取り出す
df.ix['i3','b']


# In[10]:

# : は全指定の意味
df.ix[:, 'a']


# In[11]:

# 複数の指定も可能。飛び飛びの指定も可能。
# 番号での指定も名前での指定も可能。
df.ix[[1,3], ['b','d']]


# In[12]:

# 列に関する操作は[カラム名]で渡す。
df['a']


# In[13]:

# arrayとして取得する。
df['a'].values


# In[14]:

# さらにindex名を指定することで値として取得できる。
df['a']['i3']


# In[15]:

# DataFrameをtableとみなして、位置指定から値を明示的に取る方法。
df.iloc[2,3]


# In[16]:

# これも同じ
df.ix[2,3]


# In[17]:

# 特定の列だけ取り出す
df.iloc[2]


# In[18]:

# 複数の列を取り出す
df.iloc[:, 2:4]


# In[19]:

# 改めて、データの中身を確認
df


# In[20]:

# 最後の列だけ除外する（「最後の」は「-1」で指定できます）。
df.iloc[:, :-1]


# In[21]:

# 指定した列だけ除外する（この例では、２列目だけ除外しています）。
df.iloc[:, [0,1,3,4,5]]


# ## Numpy で生成した乱数を Pandas で使う

# In[22]:

df1 = pd.DataFrame(np.random.randint(10, size=(4,5)))


# In[23]:

df1


# In[24]:

# 条件を満たすものだけを抽出
df1[df1>1]


# In[25]:

# # 条件をみたすものに-1を代入
df1[df1>5] = -1


# In[26]:

df1


# ## 欠損値を含むデータを取り扱う

# In[27]:

# 欠損値 (NaN) を含むランダムデータを作成する
df2 = pd.DataFrame(np.random.randint(10, size=(8,7)))
df2 = df2[df2>0]


# In[28]:

df2


# In[29]:

# NaNを含む行を削除
df2.dropna()


# In[30]:

# NaNを含む列を削除
df2.dropna(axis=1)


# In[31]:

# NaNを埋める
df2.fillna(-1)


# In[32]:

#直前の値で埋める
df2.fillna(method='pad')


# In[33]:

#直後の値で埋める
df2.fillna(method='bfill')


# In[34]:

# misssing valueの前後の線形の値で埋める
df2.apply(pd.Series.interpolate)


# ## 重複のあるデータを取り扱う

# In[35]:

df3 = pd.DataFrame(np.random.randint(2, size=(10,4)))


# In[36]:

df3


# In[37]:

# 重複を調べる
df3.duplicated()


# In[38]:

# 重複を調査するcolumnを指定することも可能
df3.duplicated(0)


# In[39]:

# 重複を除去する
df3.drop_duplicates()


# In[40]:

# 指定した列の重複を除去する
df3.drop_duplicates(0)


# In[41]:

# 指定した列の重複を除去し最後のを残す
df3.drop_duplicates(0, take_last=True)


# ## 行列演算を行う

# In[42]:

A = pd.DataFrame(np.random.randint(10, size=(2,2)))
B = pd.DataFrame(np.random.randint(10, size=(2,2)))


# In[43]:

A


# In[44]:

B


# In[45]:

# 行列の転置
A.T


# In[46]:

# 行列の転置
B.T


# In[47]:

# 行列の要素ごとの和
A + B


# In[48]:

# 行列の要素ごとの積（「行列の積」ではない）
A * B


# In[49]:

# 行列の積をとりたい場合は DataFrame.dot。ただし、行列の積をとるためには元データの columns と 引数の index のラベルが一致している必要がある
A.dot(B)


# In[50]:

B.dot(A)


# ## 簡単な統計量

# In[51]:

df4 = pd.DataFrame(np.random.randint(10, size=(5,10)))


# In[52]:

df4


# In[53]:

# 基本統計量の表示
df4.describe()


# In[54]:

# 列の合計値
df4.sum()


# In[55]:

# 列の平均値
df4.mean()


# In[56]:

# 列の不偏分散
df4.var()


# In[57]:

# 列の標本分散
df4.var(ddof=False)


# In[58]:

# 列の不偏標準偏差
df4.std()


# In[59]:

# 列の標本標準偏差
df4.std(ddof=False)


# In[60]:

# 行の合計値
df4.sum(axis = 1)


# In[61]:

# 行の平均値
df4.mean(axis = 1)


# ## 行列の正規化（標準化）
# 
# 正規化 (normalize) とは、異なる基準のデータを一定の基準にしたがって変形し利用しやすくすることです。

# In[62]:

df4


# In[63]:

# 一般的には平均 0 、分散 (及び標準偏差) が 1 になるように値を変換することを指します。
df4.apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)


# In[64]:

# 最大値を1、最小値を0にするような正規化もできます。
df4.apply(lambda x: (x-x.min())/(x.max() - x.min()), axis=0).fillna(0)


# In[65]:

# 合計値が１になるような正規化もできます。
df4.apply(lambda x: x/x.sum(), axis=0).fillna(0)


# ## 相関行列
# 
# 相関行列とは、各要素間の相関係数を並べたものであり、その性質から必ず対称行列である。

# In[66]:

# まずランダムな行列を作ってみる
df5 = pd.DataFrame(np.random.rand(5, 10))


# In[67]:

df5


# In[68]:

# 行間の相関行列
pd.DataFrame(np.corrcoef(df5.dropna().as_matrix().tolist()))


# In[69]:

# 列間の相関行列
pd.DataFrame(np.corrcoef(df5.dropna().T.as_matrix().tolist()))


# In[ ]:



