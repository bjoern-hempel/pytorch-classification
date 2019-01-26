#!/usr/bin/env python

import os
import pprint
import sys

# pretty printer settings
pp = pprint.PrettyPrinter(indent=4)

# source path: {} => train/val
source = 'data/prepared/food/unbalanced/90_10/all/{}'

# {} => train/val; {} => class_name
target = 'data/prepared/food/binary/unbalanced/90_10/all/{}/{}'


def collect_all_classes(path):
    """Collects all class names from the given folder. The class names must be folders in the given folder."""
    classes = []

    # collect all classes
    for name in os.listdir(path):
        classes.append(name)

    return classes


def prepare_data_folders(classes, path):
    for class_name in classes:
        for x in ['train', 'val']:

            # create true directory
            dir_true = os.path.join(path.format(x, class_name), 'true')
            if not os.path.exists(dir_true):
                print('Create "{}"'.format(dir_true))
                os.makedirs(dir_true)
            else:
                print('Target folder "{}" exists.'.format(dir_true))

            # create false directory
            dir_false = os.path.join(path.format(x, class_name), 'false')
            if not os.path.exists(dir_false):
                print('Create "{}"'.format(dir_false))
                os.makedirs(dir_false)
            else:
                print('Target folder "{}" exists.'.format(dir_false))


def copy_symlink(path_source, path_target):
    if os.path.exists(path_target):
        if os.path.islink(path_target):
            os.unlink(path_target)
            print('Delete symlink "{}".'.format(path_target))
        else:
            sys.exit('The given path must be empty or a link!')

    if os.path.islink(path_source):
        # TODO: Build path_linkto from path_source and path_target
        path_linkto = '../../' + os.readlink(path_source)

        # Create the symlink
        print('Create symlink "{}".'.format(path_target))
        os.symlink(path_linkto, path_target)
    else:
        sys.exit('Unsupported: {}'.format(path_source))


# gets all class names
class_names = collect_all_classes(source.format('train'))

# create needed directories
prepare_data_folders(class_names, target)


for class_name_1 in class_names:
    for x in ['train', 'val']:
        write_dir_true = os.path.join(target.format(x, class_name_1), 'true')
        write_dir_false = os.path.join(target.format(x, class_name_1), 'false')

        for class_name_2 in class_names:

            # target and source destination for symlinks
            read_dir = os.path.join(source.format(x), class_name_2)
            write_dir = write_dir_true if class_name_1 == class_name_2 else write_dir_false

            # some loggings
            print('Source: "{}"'.format(read_dir))
            print('Target: "{}"'.format(write_dir))

            for name_source in os.listdir(read_dir):
                path_source = os.path.join(read_dir, name_source)
                path_target = os.path.join(write_dir, name_source)

                copy_symlink(path_source, path_target)
