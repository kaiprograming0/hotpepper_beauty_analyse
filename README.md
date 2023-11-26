# ホットペッパービューティーの口コミデータ解析
口コミのデータ解析をすることで　　<br>
・サービスの評価と評判　<br>
・顧客のニーズと要望　<br>
・店舗ごとの強み　<br>
を理解することができる。

## 今回は円グラフや積立棒グラフで可視化した後に、<br>自然言語処理を用いて、wordcloudやtsne,デンドログラムなどを用いて可視化をしてみた。
今回用いたライブラリは以下である。　<br>
```
import pandas as pd
import matplotlib.pyplot as plt
import MeCab
import tqdm
import warnings
from wordcloud import WordCloud
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
```

#### データの読み込み
```
df = pd.rea_csv('ファイル名.csv')
```
#### 円グラフの出し方
```
category_counts = df['円グラフを作りたい列'].value_counts()
plt.pie(category_counts, startangle=90)
```

#### tfidfでテキスト情報を数値化
```
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['テキスト列'])
```

 #### エルボー法
 ```
distortions = []
num = 50

for i in range(1, num+1):
    model = KMeans(n_clusters=i, init='k-means++',
                   n_init=10, max_iter=300, random_state=50)
    
    model.fit(X)
    
    distortions.append(model.inertia_)
    
plt.figure(figsize=(10, 10))
plt.plot(range(1, num+1), distortions, marker='o')
plt.title('エルボー法')
plt.xlabel('クラスター数')
plt.show()
```
#### 文章を名詞、動詞、形容詞に分けるコード
```
m = MeCab.Tagger('-Owakati')
meishi_list = []
cdoushi_list = []
keiyoushi_list = []
for i in range(len(df)):
    if df['label'][i] == cluster_num:
        node = m.parseToNode(df['口コミ'][i])
        while node:
        if node.feature.split(',')[0] == '名詞':
            meishi_list.append(node.surface)
        elif node.feature.split(',')[0] == '動詞':
            doushi_list.append(node.feature.split(',')[7])
        elif node.feature.split(',')[0] == '形容詞':
            keiyoushi_list.append(node.feature.split(',')[7])
        node = node.next
```

