#!/usr/bin/env python

"""
Simulates a binary classification network.

Uses a normally distributed random number generation via function "np.random.normal".

Formula:

Probability density function (PDF): p(x) = 1/(√(2·π·σ²))·e^(-((x - μ)²/(2·σ²)))


Some configs and results (for 50 classes):

μ = 0.3333, σ = 0.0750  ⟶  0.9868  ⟶  0.9774
μ = 0.3533, σ = 0.0750  ⟶  0.9748  ⟶  0.9320
μ = 0.3833, σ = 0.0750  ⟶  0.9392  ⟶  0.7910
μ = 0.4083, σ = 0.0750  ⟶  0.8863  ⟶  0.5774
μ = 0.4333, σ = 0.0750  ⟶  0.8132  ⟶  0.3408

Usage:

user$ src/simulate-binary-data.py

"""

__author__ = "Björn Hempel"
__copyright__ = "Copyright 2019, An ixnode project"
__credits__ = ["Björn Hempel"]
__license__ = "MIT"
__version__ = "1.0.0 (2019-01-27)"
__maintainer__ = "Björn Hempel"
__email__ = "bjoern@hempel.li"
__status__ = "Production"

import numpy as np
import pprint

# Configure the pretty printer
pp = pprint.PrettyPrinter(indent=4)

# some configs
classes = 50
correct_class = 0
test_iterations = 10000

# some other configs
probability_settings = {
    # default (acc: 0.9868)
    'default': {
        'usage': [],
        'setting': {
            'mu': 1 / 3 - 0.01,
            'sigma': 0.075
        }
    },
    # use outlier for 2 classes (acc: 0.9748)
    'outlier-simple': {
        'usage': [2, 3, 4, 5],
        'setting': {
            'mu': 1 / 3 + 0.02,
            'sigma': 0.075
        }
    },
    # use outlier for 2 classes (acc: 0.8132)
    'outlier-big': {
        'usage': [10, 11],
        'setting': {
            'mu': 1 / 3 + 0.1,
            'sigma': 0.075
        }
    }
}


def softmax(x):
    """A softmax implementation."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_settings(probability_settings, id):

    setting = probability_settings['default']['setting']

    for name in probability_settings:

        if name == 'default':
            continue

        if id in probability_settings[name]['usage']:
            return probability_settings[name]['setting']

    return setting


def get_prediction(probability_setting, opposite):
    """Returns a normally distributed random number."""

    mu = probability_setting['mu']

    if opposite:
        mu = 1 - mu

    return np.random.normal(mu, probability_setting['sigma'])


def get_prediction_vector(probability_settings, classes, correct_class):
    """Returns a prediction vector with one correct predicted class (randomly normal distributed)"""
    vector = []

    for class_id in range(classes):

        setting = get_settings(probability_settings, class_id)

        if class_id == correct_class:
            prediction = get_prediction(setting, True)
        else:
            prediction = get_prediction(setting, False)

        vector.append(prediction)

    return softmax(vector)


def get_accuracy_overall(probability_settings, iterations, classes, correct_class):
    """Returns the accuracy of the predictions for given iterations (all classes)."""
    predicted = 0
    predicted_correct = 0

    for i in range(iterations):
        prediction_vector = get_prediction_vector(probability_settings, classes, correct_class)
        predicted_class = np.argmax(prediction_vector)
        predicted += 1

        if predicted_class == correct_class:
            predicted_correct += 1

    return predicted_correct / predicted


def get_accuracy_class(probability_setting, iterations):
    """Returns the accuracy for given iterations and one class."""
    counter_all = 0
    counter_true = 0

    for i in range(iterations):
        prediction = get_prediction(probability_setting, True)

        counter_all += 1

        if prediction > 0.5:
            counter_true += 1

    return counter_true / counter_all


# calculate the accuracy
accuracy_overall = get_accuracy_overall(probability_settings, test_iterations, classes, correct_class)

# print probability settings
for name in probability_settings:
    accuracy_class = get_accuracy_class(probability_settings[name]['setting'], test_iterations)
    count = len(probability_settings[name]['usage'])

    if count > 0:
        print('{} ({} classes):'.format(name, count))
    else:
        print('{}:'.format(name))

    print('μ   = {:.4f}'.format(probability_settings[name]['setting']['mu']))
    print('σ   = {:.4f}'.format(probability_settings[name]['setting']['sigma']))
    print('σ²  = {:.4f}'.format(probability_settings[name]['setting']['sigma']**2))
    print('acc = {:.4f}'.format(accuracy_class))
    print()

# print predicted overview
print(
    'Correct prediction overall ({} classes / {} iterations): {:.2f}%'.format(
        classes,
        test_iterations,
        accuracy_overall * 100
    )
)
