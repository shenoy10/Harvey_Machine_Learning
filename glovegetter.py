from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
import gensim.downloader as api

model = None
first = True

model = api.load("glove.twitter.27B.50d");
path = "data/word2vec.kv"

model.wv.save(path)

wv = model.wv
print (wv.similar_by_word('computer'))
print (wv.similarity('woman', 'man'))