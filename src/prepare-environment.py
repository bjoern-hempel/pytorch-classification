#!/usr/bin/env python

import os
import pprint
import sys

# pretty printer settings
pp = pprint.PrettyPrinter(indent=4)

# the label
label = 'all'

# the used ratio
ratio = '90_10'

# specify the classifier path
classifier_path = 'food/binary/unbalanced'

# source path
source = 'data/prepared/{}/{}/{}'.format(classifier_path, ratio, label)

# target path
target = 'data/processed/{}/{}/elements/{}'.format(classifier_path, ratio, label)

# specifies whether different models are to be generated
multi_model = True

# force prepare environment (even if already exists)
force = True

# folders within the environment
folders = ['charts', 'csv', 'log', 'models']


def collect_all_classes(path):
    """Collects all class names from the given folder. The class names must be folders in the given folder."""
    classes = []

    # collect all classes
    for name in os.listdir(path):
        classes.append(name)

    return classes


# check if target environment already exists
if os.path.exists(target) and not force:
    sys.exit('Environment folder "{}" already exists. Cancel.'.format(target))

# collect all class names
class_names = collect_all_classes(source)

# create environment folder
if not os.path.exists(target):
    print('Create environment folder "{}"'.format(target))
    os.makedirs(target)

# create folders
for folder in folders:
    folder_path = os.path.join(target, folder)

    if not os.path.exists(folder_path):
        print('Create folder "{}"'.format(folder_path))
        os.makedirs(folder_path)

if multi_model:
    folder_path = os.path.join(target, 'data')

    # create data folder
    if not os.path.exists(folder_path):
        print('Create data folder {}'.format(folder_path))
        os.makedirs(folder_path)

    # create data symlinks
    for class_name in class_names:
        symlink_source = '../../../../../../../../../' + os.path.join(source, class_name)
        symlink_target = os.path.join(folder_path, class_name)

        if not os.path.exists(symlink_target):
            print('Create symlink {}'.format(symlink_target))
            os.symlink(symlink_source, symlink_target)

        # create class_model folders
        for folder in folders:
            type_path = os.path.join(target, folder, class_name)

            if not os.path.exists(type_path):
                print('Create folder {}'.format(type_path))
                os.makedirs(type_path)


