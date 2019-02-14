#!/usr/bin/env python

"""
Evaluate all binary class models with KNN.


Usage:

user$ src/evaluate-validation-files-binary-knn.py

TODO: furthermore
- point 1
- point 2
"""

__author__ = "Björn Hempel"
__copyright__ = "Copyright 2019, An ixnode project"
__credits__ = ["Björn Hempel"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Björn Hempel"
__email__ = "bjoern@hempel.li"
__status__ = "Production"


# import some libraries
import pprint
import ml_helper
import file_helper

# Configure the pretty printer
pp = pprint.PrettyPrinter(indent=4)


data_path = 'data/processed/food/binary/unbalanced/90_10/elements/all/csv'
trained_files = file_helper.search_files(data_path, 'validated_trained_*.csv')
validated_files = file_helper.search_files(data_path, 'validated_lr*.csv')
class_names = file_helper.get_class_names_from_files(trained_files, True)

# some booleans
save_trained_data = False
save_validated_data = False
save_trained_data_values = False
save_validated_data_values = False

# get trained and validated data
data_train = file_helper.get_binary_data(trained_files, save_trained_data, 'data_train')
data_val = file_helper.get_binary_data(validated_files, save_validated_data, 'data_val')

# get trained and validated data values
data_train_values = file_helper.get_binary_data_values(
    data_train,
    save_trained_data_values,
    'data_train_values',
    class_names
)
data_val_values = file_helper.get_binary_data_values(
    data_val,
    save_validated_data_values,
    'data_val_values',
    class_names
)

counter_all = 0
counter_correct = 0

for class_name in data_val_values:
    for data in data_val_values[class_name]:
        predicted_class_name = ml_helper.k_nearest_neighbors(data_train_values, data, 51)

        counter_all += 1

        if class_name == predicted_class_name:
            counter_correct += 1

        print('{}) Predicted current: {:.2f}%'.format(counter_all, counter_correct / counter_all * 100))

print('Predicted overall: {:.2f}%'.format(counter_correct / counter_all * 100))
