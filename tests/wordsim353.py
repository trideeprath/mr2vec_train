import csv
from mr2vec_train.gensim_package.models import Word2Vec
import numpy as np


def word_sim_accuracy(model, compare = True, compare_model = None):
    m_sim_array = []
    c_sim_array = []
    human_array = []
    if compare_model is None:
        c_model = Word2Vec.load_word2vec_format('/home/trideep/Downloads/GoogleNews-vectors-negative300.bin', binary=True) # C binary format
        print("loaded")
    with open('tests/combined.csv', 'r') as f:
        next(f, None)
        reader = csv.reader(f)
        c = 0
        for row in reader:
            word1 = row[0]
            word2 = row[1]
            human = float(row[2])/10
            #print(word1, word2, human)
            c +=1
            if word1 in model and word2 in model:
                m_sim =  model.similarity(word1,word2)
                m_sim_array.append(m_sim)
                c_sim = c_model.similarity(word1, word2)
                c_sim_array.append(c_sim)
                human_array.append(human)
                print(c, word1, word2, model.similarity(word1,word2), c_model.similarity(word1, word2), human)
            else:
                if word1 not in model:
                    print(word1)
                else:
                    print(word2)

    m_rms = rmse(np.array(m_sim), np.array(human))
    c_rms = rmse(np.array(c_sim), np.array(human))
    print(m_rms, c_rms)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())








