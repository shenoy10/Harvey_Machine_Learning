from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
import gensim.downloader as api
from gensim.models import KeyedVectors


wv = KeyedVectors.load("data/word2vec.kv", mmap='r')

print (wv.similar_by_word('computer'))
print (wv.similarity('woman', 'man'))