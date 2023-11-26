# ホットペッパービューティーの口コミデータ解析
口コミのデータ解析をすることで　　<br>
・サービスの評価と評判　<br>
・顧客のニーズと要望　<br>
・店舗ごとの強み　<br>
を理解することができる。

## 今回は円グラフや棒グラフで可視化した後に、<br>自然言語処理を用いて、wordcloudやtsne,デンドログラムなどを用いて可視化をしてみた。
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
