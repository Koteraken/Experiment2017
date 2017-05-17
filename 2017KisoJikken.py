
# coding: utf-8

# # 2017年基礎実験
# <h2><a href="https://tinyurl.com/2017KisoJikken" target="_blank">https://tinyurl.com/2017KisoJikken</a></h2>
# 
# ## Jupyter Notebook を使った大規模データの可視化
# 
# 本実験では、Jupyter Notebook を使って大規模データの可視化を行います。本実験では、生物学実験で得られたものではないデータも多く取り扱いますが、数値データを可視化し俯瞰することは解析の基礎であり、どのようなデータであれ非常に大切なことです。
# 
# ### Jupyter Notebook (IPython Notebook) とは
# * Python という名のプログラミング言語が使えるプログラミング環境。計算コードと計算結果を同じ場所に時系列で保存できるので、実験系における実験ノートのように、いつどんな処理を行って何を得たのか記録して再現するのに便利。
# * <a href="https://raw.githubusercontent.com/maskot1977/-/master/%E6%BC%94%E7%BF%92%E5%AE%A4.txt" target="_blank">当学の演習室での使い方</a>
# * <a href="http://www.task-notes.com/entry/20151129/1448794509" target="_blank">個人PCでのインストールと始め方</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/-/master/%E5%B0%8F%E5%AF%BA%E7%A0%942017.pptx.png" target="_blank">小寺研究室</a> では、MacOSX上で右記のようにセットアップしています。 <a href="https://sites.google.com/site/masaakikotera/8-python/8-1-huan-jing-gou-zhu" target="_blank">環境設定</a>

# ## 本実習で用いるデータ
# （<a href="DataForPractice2017.ipynb" target="_blank">詳細</a>）
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/airquality.txt" target="_blank">「ニューヨークの大気状態観測値」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/USArrests.txt" target="_blank">「合州国の州別暴力犯罪率」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/sports_dataJt.txt" target="_blank">「スポーツテストデータ」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/icecream_chosa.txt" target="_blank">「好きなアイスクリームアンケート」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/PLIlive_dataJ.txt" target="_blank">「新国民生活指標データ」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/sake_dataJ.txt" target="_blank">「都道府県別アルコール類の消費量」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/sbnote_dataJt.txt" target="_blank">「スイス銀行紙幣データ」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/iris.txt" target="_blank">「あやめのデータ」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/pima-indians-diabetes.txt" target="_blank">「ピマ・インディアンの糖尿病診断」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/parkinsons.data" target="_blank">「パーキンソン病診断データ」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/ecoli.data" target="_blank">「大腸菌タンパク質の細胞内局在」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/yeast.data" target="_blank">「酵母タンパク質の細胞内局在」</a>

# ## 本実習で収集するデータ
# 本実習では、データ収集から体験してもらうために、匿名アンケートを実施します。あまり深く考えず、短時間で気軽にパパッと回答してください。
# * <a href="https://docs.google.com/forms/d/e/1FAIpQLSf4yvNCXQRRuhwSxB5KyBDjL6RLbcPyilSKpI6tiUy7Vad2Pg/viewform" target="_blank">学生実験用アンケート</a>

# ## 本実習スタート
# 
# 私の担当回では、実験の班分けは関係ありません。個人で課題を解いて、個人で提出してください。友達同士で話し合ったり助け合ったりすることは自由です。ただし、レポートは個人個人のものですので、それぞれ自分のオリジナリティを出すこと。
# 
# __課題について__ ：大学受験までの問題では、そのほとんどが、答えがどこかに書いてあって、先生は答えを知っていて、その答えを書けば正解でした。ですが社会にあるほとんどの問題には、答えがなかったり、誰も答えを知らなかったりします。そういう問題について分析して、自分の答えを出す能力が求められます。本実習でも、誰も知らない答えを自らで導き出すつもりで解いてください。

# ### 本日の目標（本日の講義時間終了まで）
# * __Step1__ から __Step6__ まで、とりあえず見よう見まねで計算を実行する（コピペでOK）。
# * 先に進める人は、どんどん先に進んでください。課題をさっさと解いてしまってもOKです。
# * 私の説明を聞きながら進めていってもOKです。
# * 分からないところや、思った通りにプログラムが動作しないことがあったら手を挙げて、私かTAに助けを求めてください。
# * 最後に課題がありますので、１週間以内に解いてレポートを提出してください（本日中に提出してもOKです）。
# 
# ### レポートの提出方法について（７日後の１３時２０分まで）
# 
# 下記の課題のレポートを、指定されたメールアドレスまで送信してください。
# * __締切__：７日後の１３時２０分。提出締切厳守とします（遅れた場合、受け取らないことがあります）。
# * __提出方法__：Microsoft Word、Microsoft PowerPoint、あるいは .ipynbファイルのいずれかの形式のファイルでレポートを作成し、指定されたメールアドレスまでメールで送信してください。メールタイトルは「2017基礎実験」とし、メール本文と添付ファイルの両方に、学籍番号と氏名を明記すること。

# ### Step 1. まずは、ウォーミングアップから
# まずは次のリンクをクリックして、Pythonで簡単な計算をしてみましょう。
# 
# * <a href="PythonWarmup.ipynb" target="_blank">まずは、ウォーミングアップから</a>

# ### Step 2. ライブラリのインポート、乱数、そしてMatplotlibによる描画
# Pythonでは、皆がよく使う関数などをライブラリとしてまとめてあり、それをインポートして使います。乱数のライブラリの使い方や、ヒストグラムなどのグラフの描画の仕方を体験してみましょう。
# 
# * <a href="ImportingLibrariesAndMatplotlib.ipynb" target="_blank">ライブラリのインポート、乱数、そしてMatploblibによる描画</a>

# ### Step 3. Numpy と Pandas を用いた演算
# 行列データの取り扱いについて、簡単に学んでみましょう。
# 
# * <a href="UsingNumpyAndPandas.ipynb" target="_blank">NumpyとPandasを用いた演算</a>

# ### Step 4. タブ区切りデータ、コンマ区切りデータ等の読み込み
# インターネット上にあるテキストファイル（タブ区切りデータ、コンマ区切りデータ等）をダウンロードし、Pandasに読み込む方法を会得しましょう。
# 
# * <a href="ReadingCSV.ipynb" target="_blank">タブ区切りデータ、コンマ区切りデータ等の読み込み</a>
# 
# 次に、先ほどの学生実験用アンケート結果もダウンロードして Pandas に読み込んでみましょう。
# 
# * <a href="https://docs.google.com/spreadsheets/d/1u50kS8Ztmgjjs--S1AM4753quL3QEPRR7xIGjCxcSCw/edit#gid=941271295" target="_blank">学生実験用アンケート結果</a>
# 
#     * 左上の「File」→「Download as」→「Comma-separated values (.csv, current sheet)」で保存

# ### Step 5. 読み込んだデータの可視化
# 大規模データを可視化する方法はいくつかありますが、本実習では以下の方法を学んでください。

# <ul>
# <li><a href="LineGraph.ipynb" target="_blank">折れ線グラフ</a> ... 「ニューヨークの大気状態観測値」を例に
# <li><a href="Histgrams.ipynb" target="_blank">ヒストグラム</a> ... 「好きなアイスクリームアンケート」を例に
# <li><a href="Boxplots.ipynb" target="_blank">ボックスプロットとバイオリンプロット</a> ... 「あやめのデータ」を例に
# <li><a href="ScatterPlot.ipynb" target="_blank">散布図</a> ... 「合州国の州別暴力犯罪率」を例に
# <li><a href="ScatterMatrix.ipynb" target="_blank">散布図行列</a> ... 「スポーツテストデータ」を例に
# <li><a href="CorrelationMatrix.ipynb" target="_blank">相関行列</a> ... 「新国民生活指標データ」を例に
# </ul>
# 
# 次に、先ほどの学生実験用アンケート結果も可視化してみてください。

# ### Step 6. 機械学習（教師なし）

# 教師なし学習（Unsupervised Learning）とは、機械学習の手法の一つで、データの背後に存在する本質的な構造を抽出するために用いられます。本実習ではこれを少しだけ体験してみましょう。
# <ul>
# <li><a href="PCA2017.ipynb" target="_blank">主成分分析</a> ... 「都道府県別アルコール類の消費量」を例に
# <li><a href="HierarchicalClustering2017.ipynb" target="_blank">階層的クラスタリング</a> ... 「都道府県別アルコール類の消費量」を例に
# </ul>

# 「教師あり学習」（Supervised Learning）というものもありますが、それはまた次の機会に、、、

# # 課題
# 
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/airquality.txt" target="_blank">「ニューヨークの大気状態観測値」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/USArrests.txt" target="_blank">「合州国の州別暴力犯罪率」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/sports_dataJt.txt" target="_blank">「スポーツテストデータ」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/icecream_chosa.txt" target="_blank">「好きなアイスクリームアンケート」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/PLIlive_dataJ.txt" target="_blank">「新国民生活指標データ」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/sake_dataJ.txt" target="_blank">「都道府県別アルコール類の消費量」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/sbnote_dataJt.txt" target="_blank">「スイス銀行紙幣データ」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/iris.txt" target="_blank">「あやめのデータ」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/pima-indians-diabetes.txt" target="_blank">「ピマ・インディアンの糖尿病診断」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/parkinsons.data" target="_blank">「パーキンソン病診断データ」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/ecoli.data" target="_blank">「大腸菌タンパク質の細胞内局在」</a>
# * <a href="https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/yeast.data" target="_blank">「酵母タンパク質の細胞内局在」</a>
# * 学生実験用アンケート結果
# 
# 以上のデータ（<a href="DataForPractice2017.ipynb" target="_blank">詳細</a>）の中から３つ以上を選び、それぞれデータの可視化と機械学習を用いながら解析し、結果を考察してください。
# 
# __締切__ ：７日後の１３時２０分。提出締切厳守とします（遅れた場合、受け取らないことがあります）。
# 
# __提出方法__ ：Microsoft Word、Microsoft PowerPoint、あるいは .ipynbファイルのいずれかの形式のファイルでレポートを作成し、指定されたメールアドレスまでメールで送信してください。メールタイトルは「2017基礎実験」とし、メール本文と添付ファイルの両方に、学籍番号と氏名を明記すること。
# 
# また、この実験に関して感想などがありましたら書いてくださると嬉しいです。今後の講義や実験などの改善につながるかもしれません。

# In[ ]:



