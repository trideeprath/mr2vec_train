#from gensim.models import Word2Vec
from nltk.corpus import brown
from tests.my_utils import visualize_vectors
#from gensim.models.word2vec import LineSentence
#from gensim.models.word2vec import BrownRelCorpus
from mr2vec_train.gensim_package.models import Word2Vec
from mr2vec_train.gensim_package.models.word2vec import LineSentence
import datetime


def w2v_model_accuracy(self):
    print("Finding Accuracy")
    accuracy = self.accuracy("tests/questions-words.txt")
    sum_corr = len(accuracy[-1]['correct'])
    sum_incorr = len(accuracy[-1]['incorrect'])
    total = sum_corr + sum_incorr
    percent = lambda a: a / total * 100
    print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, percent(sum_corr),
                                                                             percent(sum_incorr)))



#sentences = LineSentence('corpus/brown_sen_rel.txt', limit=200000)
print(datetime.datetime.now())
sentences = LineSentence('corpus/wiki_rel_text9.txt')
model = Word2Vec(sentences, size=100, window=5, min_count=10, negative=10, negative_rel=15, workers=4, iter=1, batch_words=1000)
model.save_word2vec_format("trained_models/w2vectors"+"_s100_w5_neg1520" +".txt")
print("Training Finished")
#visualize_vectors(model)
w2v_model_accuracy(model)
print(datetime.datetime.now())
