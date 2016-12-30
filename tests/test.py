from mr2vec_train.gensim_package.models import Word2Vec
from tests.my_utils import visualize_vectors
from pprint import pprint

def w2v_model_accuracy(self, section_wise = False):
    print("Finding Accuracy")
    accuracy = self.accuracy("tests/questions-words.txt")
    pprint(accuracy)
    if section_wise:
        for acc in accuracy:
            section = acc['section']
            correct = len(acc['correct'])
            incorrect = len(acc['incorrect'])
            tot = correct + incorrect
            print(section, correct * 100 / tot)

    sum_corr = len(accuracy[-1]['correct'])
    sum_incorr = len(accuracy[-1]['incorrect'])
    total = sum_corr + sum_incorr
    percent = lambda a: a / total * 100
    print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, percent(sum_corr),
                                                                             percent(sum_incorr)))



model = Word2Vec.load_word2vec_format("trained_models/w2vectors_s100_w5_neg1520.txt")
w2v_model_accuracy(model)

#print(model.most_similar('Turing'))

#print(model.most_similar(positive=['nsubj','Batman']))
#visualize_vectors(model,words=['king', 'queen', 'food', 'delicious', 'the', 'a', 'an'])

