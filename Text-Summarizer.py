import nltk
nltk.download('stopwords')
import numpy as np
document = input("Enter the para")
print('The length of the file is:', end=' ')
print(len(document))
from string import punctuation
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize,sent_tokenize
from collections import defaultdict
from nltk.corpus import stopwords
doc = ''.join(c for c in document if not c.isdigit())
sent_token = sent_tokenize(doc)
doc = ''.join(c for c in doc if c not in punctuation).lower()
doc =' '.join([word for word in doc.split() if word not in (stopwords.words("english"))])
word_token = word_tokenize(doc)
print(doc)

def score_tokens(filterd_words, sentence_tokens):
    word_freq = FreqDist(filterd_words)
    ranking = defaultdict(int)
    for i, sentence in enumerate(sentence_tokens):
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                ranking[i] += word_freq[word]
                
    return ranking
sentence_ranks = score_tokens(word_token, sent_token)
print(sentence_ranks)
from heapq import nlargest
def summarize(ranks, sentences, length):
    indexes=nlargest(1,ranks, key=ranks.get)
    final_sentences = [sentences[j] for j in sorted(indexes)]
    return ' '.join(final_sentences)
summary = summarize(sentence_ranks, sent_token,length=1)
print(summary)
