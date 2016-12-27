#from gensim.models import Word2Vec
from nltk.corpus import brown
from tests.my_utils import visualize_vectors
#from gensim.models.word2vec import LineSentence
#from gensim.models.word2vec import BrownRelCorpus
from mr2vec_train.gensim_package.models import Word2Vec
from mr2vec_train.gensim_package.models.word2vec import LineSentence
from mr2vec_train.gensim_package.models.word2vec import BrownRelCorpus


def w2v_model_accuracy(self):
    print("Finding Accuracy")
    accuracy = self.accuracy("tests/questions-words.txt")
    sum_corr = len(accuracy[-1]['correct'])
    sum_incorr = len(accuracy[-1]['incorrect'])
    total = sum_corr + sum_incorr
    percent = lambda a: a / total * 100
    print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, percent(sum_corr),
                                                                             percent(sum_incorr)))



sentences = BrownRelCorpus
sent = []

sentences = LineSentence('corpus/brown_sen_rel.txt', limit=200000)
model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4, iter=3, batch_words=1000)
model.save_word2vec_format("trained_models/w2vectors.txt")
print("Training Finished")
visualize_vectors(model)
w2v_model_accuracy(model)




