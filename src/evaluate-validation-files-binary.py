#!/usr/bin/env python

"""
Get the overall accuracy from the validation csv files from binary train.


Usage:

user$ src/evaluate-validation-files-binary.py

TODO: furthermore
- point 1
- point 2
"""

__author__ = "Björn Hempel"
__copyright__ = "Copyright 2019, An ixnode project"
__credits__ = ["Björn Hempel"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Björn Hempel"
__email__ = "bjoern@hempel.li"
__status__ = "Production"


import pprint
import csv
import os
import math
import pickle
import file_helper


# Configure the pretty printer
pp = pprint.PrettyPrinter(indent=4)


data_path = 'data/processed/food/binary/unbalanced/90_10/elements/all/csv'

save_files = False


def get_best_class_name(classes):

    max_value = -math.inf
    max_class_name = None

    for class_name in classes:
        if classes[class_name]['true'] > max_value:
            max_value = classes[class_name]['true']
            max_class_name = class_name

    return [max_class_name, max_value]


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


if save_files:
    validated_files = file_helper.search_files(data_path, 'validated_*.csv')

    files = {}
    counter_model = {}
    counter_model_correct = {}
    counter_model_own = {}
    counter_model_own_correct = {}
    counter_model_other = {}
    counter_model_other_correct = {}

    counter_model_overall = 0
    counter_model_correct_overall = 0
    counter_model_own_overall = 0
    counter_model_own_correct_overall = 0
    counter_model_other_overall = 0
    counter_model_other_correct_overall = 0

    path_wrong_own = {}
    path_wrong_other = {}
    values = {}

    for validated_file in validated_files:
        class_name = file_helper.get_class_name_from_config_file(validated_file)

        counter_model[class_name] = 0
        counter_model_correct[class_name] = 0
        counter_model_own[class_name] = 0
        counter_model_own_correct[class_name] = 0
        counter_model_other[class_name] = 0
        counter_model_other_correct[class_name] = 0
        path_wrong_own[class_name] = []
        path_wrong_other[class_name] = []
        values[class_name] = {}

        counter_line = 0

        with open(validated_file, newline='') as csvfile:
            settings = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)

            for row in settings:
                counter_line += 1

                # ignore first line
                if counter_line <= 1:
                    continue

                path = os.readlink(row[2]).split('/food/')[1]
                splitted_path = path.split('/')

                if not splitted_path[0] in files:
                    files[splitted_path[0]] = {}

                if not splitted_path[1] in files[splitted_path[0]]:
                    files[splitted_path[0]][splitted_path[1]] = {}

                # count the examined elements
                counter_model[class_name] += 1
                counter_model_overall += 1

                if splitted_path[0] == class_name:
                    counter_model_own[class_name] += 1
                    counter_model_own_overall += 1
                else:
                    counter_model_other[class_name] += 1
                    counter_model_other_overall += 1

                # 0 - true, 1 - false
                predicted = [
                    float(row[8]) if row[5] == 'false' else float(row[6]),
                    float(row[6]) if row[5] == 'false' else float(row[8])
                ]

                # use softmax
                if False:
                    predicted = tools_helper.softmax(predicted)

                values[class_name][path] = [predicted[0], predicted[1]]

                correct = False

                if splitted_path[0] == class_name:
                    # class "true" expected
                    correct = predicted[0] > predicted[1]
                else:
                    # class "false" expected
                    correct = predicted[1] > predicted[0]

                # count correct predicted class
                if correct:
                    counter_model_correct[class_name] += 1
                    counter_model_correct_overall += 1

                    if splitted_path[0] == class_name:
                        counter_model_own_correct[class_name] += 1
                        counter_model_own_correct_overall += 1
                    else:
                        counter_model_other_correct[class_name] += 1
                        counter_model_other_correct_overall += 1
                else:
                    if splitted_path[0] == class_name:
                        path_wrong_own[class_name].append(path)
                    else:
                        path_wrong_other[class_name].append(path)

                files[splitted_path[0]][splitted_path[1]][class_name] = {
                    'true': predicted[0],
                    'false': predicted[1],
                    'corr': correct
                }

    # save read setting files
    save_obj(files, 'files')

    # evaluate all models (overview)
    print('Evaluate model (overview):')
    print('---------------')

    for cn in counter_model:
        print('{: <25} - {:6.2f}% ({:5d}/{:5d})    own: {:6.2f}% ({:4d}/{:4d        })    other: {:6.2f}% ({:5d}/{:5d})'.format(
           cn,
           counter_model_correct[cn] / counter_model[cn] * 100,
           counter_model_correct[cn], counter_model[cn],
           counter_model_own_correct[cn] / counter_model_own[cn] * 100,
           counter_model_own_correct[cn], counter_model_own[cn],
           counter_model_other_correct[cn] / counter_model_other[cn] * 100,
           counter_model_other_correct[cn], counter_model_other[cn]
        ))
    print('----------')
    print('{: <25} - {:6.2f}% ({:5d}/{:5d})    own: {:6.2f}% ({:3d}/{:3d})    other: {:6.2f}% ({:5d}/{:5d})'.format(
        'overall',
        counter_model_correct_overall / counter_model_overall * 100,
        counter_model_correct_overall, counter_model_overall,
        counter_model_own_correct_overall / counter_model_own_overall * 100,
        counter_model_own_correct_overall, counter_model_own_overall,
        counter_model_other_correct_overall / counter_model_other_overall * 100,
        counter_model_other_correct_overall, counter_model_other_overall
    ))
    print('')
    print('')

    # evaluate all models (detail)
    print('Evaluate model (detail):')
    print('---------------')

    for cn in counter_model:
        print('{: <25} - {:6.2f}% ({:4d}/{:4d})    own: {:6.2f}% ({:3d}/{:3d})    other: {:6.2f}% ({:4d}/{:4d})'.format(
           cn,
           counter_model_correct[cn] / counter_model[cn] * 100,
           counter_model_correct[cn], counter_model[cn],
           counter_model_own_correct[cn] / counter_model_own[cn] * 100,
           counter_model_own_correct[cn], counter_model_own[cn],
           counter_model_other_correct[cn] / counter_model_other[cn] * 100,
           counter_model_other_correct[cn], counter_model_other[cn]
        ))
        for path in path_wrong_own[cn]:
            print('- False predicted (true = {:6.2f}, false = {:6.2f}): {}'.format(
                values[cn][path][0],
                values[cn][path][1],
                path
            ))
        for path in path_wrong_other[cn]:
            print('- True  predicted (true = {:6.2f}, false = {:6.2f}): {}'.format(
                values[cn][path][0],
                values[cn][path][1],
                path
            ))
        print('')
    print('')

else:
    files = load_obj('files')


counter = 0
counter_correct = 0

counter_class = {}
counter_class_correct = {}

length_class = {}
for class_name in files:
    for file_name in files[class_name]:
        for class_name_2 in files[class_name][file_name]:
            if class_name_2 not in length_class:
                length_class[class_name_2] = 0

            length = files[class_name][file_name][class_name_2]['true'] - files[class_name][file_name][class_name_2]['false']

            length = length if length > 0 else length * -1

            length_class[class_name_2] = length if length > length_class[class_name_2] else length_class[class_name_2]


if True:
    for class_name in files:
        for file_name in files[class_name]:
            for class_name_2 in files[class_name][file_name]:
                if class_name_2 not in length_class:
                    length_class[class_name_2] = 0

                files[class_name][file_name][class_name_2]['true'] /= (length_class[class_name_2])
                files[class_name][file_name][class_name_2]['false'] /= (length_class[class_name_2])


for class_name in files:

    #if class_name != 'sloppy_joe':
    #    continue

    counter_class[class_name] = 0
    counter_class_correct[class_name] = 0

    for file_name in files[class_name]:
        counter += 1
        counter_class[class_name] += 1

        (class_name_predicted, value_predicted) = get_best_class_name(files[class_name][file_name])

        #class_name_predicted = class_name

        predicted_correct = False
        if class_name == class_name_predicted:
            predicted_correct = True

        #if files[class_name][file_name][class_name]['corr']:
        #    predicted_correct = True

        if predicted_correct:
            counter_correct += 1
            counter_class_correct[class_name] += 1

            print('{}/{}; real = "{}"; pred = "{}"'.format(class_name, file_name, class_name, class_name_predicted))
            if False:
                for cn in files[class_name][file_name]:
                    print('    {: <25}: true = {:6.2f}, false = {:6.2f}, correct = {}'.format(
                        cn,
                        files[class_name][file_name][cn]['true'],
                        files[class_name][file_name][cn]['false'],
                        files[class_name][file_name][cn]['corr']
                    ))
        else:
            print('{}/{}; real = "{}"; pred = "{}"'.format(class_name, file_name, class_name, class_name_predicted))
            if True:
                for cn in files[class_name][file_name]:
                    print('    {: <25}: true = {:6.2f}, false = {:6.2f}, correct = {}'.format(
                        cn,
                        files[class_name][file_name][cn]['true'],
                        files[class_name][file_name][cn]['false'],
                        files[class_name][file_name][cn]['corr']
                    ))

print('')

acc = counter_correct / counter
print('Acc "overall": {:6.2f}% ({:4d} / {:4d})'.format(acc * 100, counter_correct, counter))

print('')

for cn in counter_class:
    acc = counter_class_correct[cn] / counter_class[cn]
    print('Acc {: <25}: {:6.2f}% ({:2d} / {:2d})'.format(cn, acc * 100, counter_class_correct[cn], counter_class[cn]))
