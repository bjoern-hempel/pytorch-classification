import os
import glob
import platform
import re
import pickle
import csv


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime


def create_folder_for_file(file, mode=0o775):
    path = os.path.dirname(file)
    os.makedirs(path, mode, True)


def search_files(path, name):
    files = []
    for file in glob.iglob('{}/**/{}'.format(path, name), recursive=True):
        files.append(file)

    return sorted(files)


def get_class_name_from_config_file(path):
    """Extracts the class name from given path."""
    paths = path.split('/csv/')
    splitted_paths = paths[1].split('/')
    class_name = splitted_paths[0]

    return class_name

def analyse_file_and_get_config(file):

    # build file template
    file_template = re.sub(
        r"/(csv|models)/",
        "/{}/",
        re.sub(
            r"/(settings|summary_full|summary|validated|checkpoint|model_best)_",
            "/{}_",
            re.sub(
                r"\.(csv)$",
                ".{}",
                file
            )
        )
    )

    # settings of all files to create
    settings = {
        'csv.settings': ['csv', 'settings', 'csv'],
        'csv.summary_full': ['csv', 'summary_full', 'csv'],
        'csv.summary': ['csv', 'summary', 'csv'],
        'csv.validated': ['csv', 'validated', 'csv'],

        'model.checkpoint': ['models', 'checkpoint', 'pth'],
        'model.model_best': ['models', 'model_best', 'pth'],

        'pdf.confusion_matrix_val': ['charts', 'confusion_matrix_val', 'pdf'],
        'png.confusion_matrix_val': ['charts', 'confusion_matrix_val', 'png'],

        'pdf.mds': ['charts', 'mds', 'pdf'],
        'png.mds': ['charts', 'mds', 'png']
    }

    config = {
        'files': {},
        'property_path': os.path.dirname(file.split('/processed/')[1]).replace('/elements/', '/')
    }

    for key, values in settings.items():
        file_path = file_template.format(values[0], values[1], values[2])

        config['files'][key] = {
            'path': file_path,
            'exists': os.path.isfile(file_path)
        }

    return config


def get_class_names_from_files(files, only_keys=False):

    class_names = {}
    for file in files:
        class_name = get_class_name_from_config_file(file)
        class_names[class_name] = len(class_names)

    if only_keys:
        return class_names.keys()

    return class_names


def get_binary_data(files, save, name):

    if save:
        data = {}

        for file in files:
            class_name = get_class_name_from_config_file(file)
            print(class_name)

            with open(file, newline='') as csv_file:
                counter_line = 0

                settings = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)

                for row in settings:
                    counter_line += 1

                    # ignore first line
                    if counter_line <= 1:
                        continue

                    path = os.readlink(row[2]).split('/food/')[1]
                    splitted_path = path.split('/')

                    # 0 - true, 1 - false
                    predicted = [
                        float(row[8]) if row[5] == 'false' else float(row[6]),
                        float(row[6]) if row[5] == 'false' else float(row[8])
                    ]

                    if splitted_path[0] not in data:
                        data[splitted_path[0]] = {}

                    if splitted_path[1] not in data[splitted_path[0]]:
                        data[splitted_path[0]][splitted_path[1]] = {}

                    # add true value to data
                    data[splitted_path[0]][splitted_path[1]][class_name] = predicted[0]

        save_obj(data, name)
    else:
        data = load_obj(name)

    return data


def get_binary_data_values(data_values, save, name, class_names):
    if save:
        data = {}

        for class_name in data_values:
            data[class_name] = []
            for file_name in data_values[class_name]:
                values = []
                for class_name_2 in class_names:
                    values.append(data_values[class_name][file_name][class_name_2])
                data[class_name].append(values)

        save_obj(data, name)
    else:
        data = load_obj(name)

    return data


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
