import csv
from gensim_package.models import Word2Vec
import numpy as np
from tests.my_utils import visualize_vectors
from my_utils import get_all_names_params

def word_sim_accuracy(model, compare=True, c_model = None, print_result=False):
    m_sim_array = []
    c_sim_array = []
    human_array = []
    if c_model is None:
        c_model = Word2Vec.load_word2vec_format('/home/ASUAD/trath/Downloads/GoogleNews-vectors-negative300.bin', binary=True) # C binary format
        print("loaded")
    with open('tests/combined.csv', 'r') as f:
        next(f, None)
        reader = csv.reader(f)
        c = 0
        for row in reader:
            try:
                word1 = row[0]
                word2 = row[1]
                human = float(row[2])/10
                #print(word1, word2, human)
                c +=1
                m_sim =  model.similarity(word1,word2)
                m_sim_array.append(m_sim)
                c_sim = c_model.similarity(word1, word2)
                c_sim_array.append(c_sim)
                human_array.append(human)
                if print_result:
                    print(c, word1, word2, model.similarity(word1,word2), c_model.similarity(word1, word2), human)
            except Exception as e:
                print(str(e))
                '''
                if word1 not in model and print_result:
                    print(word1)
                else:
                    if print_result:
                        print(word2)
                '''
    m_rms = rmse(np.array(m_sim), np.array(human))
    c_rms = rmse(np.array(c_sim), np.array(human))
    print(m_rms, c_rms)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())



#model = Word2Vec.load_word2vec_format("trained_models/w2vectors_s300_w5_neg1520_m1.txt")
#word_sim_accuracy(model)
#visualize_vectors(model)


sizes = [300, 400]
windows = [5, 8]
negs = [15]
neg_rels = [15, 20, 25]
methods = ["m1"]
all_names, params = get_all_names_params(sizes, windows, negs, neg_rels, methods)

c = 0
print("total : " , len(all_names))
c_model = Word2Vec.load_word2vec_format(
    "/home/ASUAD/trath/Downloads/deps.words_w2v", binary=False)
for name, param in zip(all_names, params):
    print("=====================================================")
    f_name = "trained_models/w2vectors"+name +".txt"
    print(c, f_name, param)
    model = Word2Vec.load_word2vec_format(f_name)
    #w2v_model_accuracy(model)
    word_sim_accuracy(model, c_model=c_model)
    c=+1

