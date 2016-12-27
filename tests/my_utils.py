import numpy as np
import matplotlib.pyplot as plt


def visualize_vectors(model, words=None):
    if words is None:
        visualizeWords = ["the", "a", "an", "good", "great", "well", "worth",  "bad", "dumb", "king","queen"]
    else:
        visualizeWords= words
    visualizeVecs = np.array([model[word] for word in visualizeWords])
    covariance = 1.0 / len(visualizeWords) * visualizeVecs.T.dot(visualizeVecs)
    U, S, V = np.linalg.svd(covariance)
    coord = visualizeVecs.dot(U[:, 0:2])

    for i in range(len(visualizeWords)):
        plt.text(coord[i, 0], coord[i, 1], visualizeWords[i],
                 bbox=dict(facecolor='green', alpha=0.1))

    plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

    plt.savefig('q3_word_vectors.png')
    plt.show()

