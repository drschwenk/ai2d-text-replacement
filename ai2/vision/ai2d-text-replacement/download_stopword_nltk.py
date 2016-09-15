from nltk.corpus import stopwords
import pickle

cachedStopWords = stopwords.words("english")

stopword_fn = 'cachedStopWords.p'
pickle.dump(cachedStopWords, open(stopword_fn, 'wb'))