import warnings
import numpy as np

from collections import Counter


def k_nearest_neighbors(data, predict, k=3):
    """A KNN implementation."""
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups ({}, {})!'.format(len(data), k))

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    # calculate the best k distances
    votes = [i[1] for i in sorted(distances)[:k]]

    # return the class with best number of votes
    return Counter(votes).most_common(1)[0][0]


def softmax(x):
    """A softmax implementation."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
