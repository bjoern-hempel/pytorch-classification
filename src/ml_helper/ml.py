import warnings
import numpy as np

from collections import Counter


def k_nearest_neighbors(data, predict, k=3):
    """A KNN implementation.

    Usage:

    dataset = {
        'k': [[1, 2, 3, 5], [1, 2, 3, 4], [2, 3, 4, 5], [3, 1, 5, 6]],
        'r': [[6, 5, 3, 6], [7, 7, 4, 7], [8, 6, 5, 8]]
    }
    
    new_features = [1, 2, 3, 4]

    result = ml_helper.k_nearest_neighbors(dataset, new_features, 5)

    print(result)

    """
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
