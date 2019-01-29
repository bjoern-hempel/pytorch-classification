import os
import csv
import glob
import pprint

from datetime import datetime
from datetime import timezone

from __helper import *
from __file import *
from __args import *

# pretty printer
pp = pprint.PrettyPrinter(indent=4)


def convert_data(data):
    if isinstance(data, str):
        if data.isdigit():
            return int(data)

        if isFloat(data):
            return float(data)

        if data == 'False':
            return False

        if data == 'True':
            return True

    return data


def get_data(path_to_csv):
    data = {}
    counter = 0
    with open(path_to_csv, newline='') as csvfile:
        counter += 1

        if counter <= 1:
            next(csvfile)

        settings = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        for row in settings:
            data[row[1]] = row[2]

    dirname = os.path.dirname(path_to_csv)
    folders = dirname.split('/csv/')

    data['process_path'] = folders[0]
    data['property_path'] = folders[1]

    properties = folders[1].split('/')

    basename = os.path.basename(path_to_csv)

    data['csv_path_settings'] = os.path.join(dirname, basename)
    data['csv_path_summary_full'] = data['csv_path_settings'].replace('settings_', 'summary_full_')
    data['csv_path_summary'] = data['csv_path_settings'].replace('settings_', 'summary_')
    data['model_path'] = os.path.join(folders[0], 'models', folders[1], basename).replace('settings_', 'model_best_').replace('.csv', '.pth')
    data['log_version'] = '1.0'

    data['model_size'] = 0
    if os.path.isfile(data['model_path']):
        stat_info = os.stat(data['model_path'])
        data['model_size'] = stat_info.st_size

    # convert all data
    for index in data:
        data[index] = convert_data(data[index])

    time_taken = 0
    max_train_accuracy = 0
    max_val_accuracy = 0
    max_train_accuracy_5 = 0
    max_val_accuracy_5 = 0
    number_trained = 0
    best_epoch = 0
    counter = 0

    with open(data['csv_path_summary'], newline='') as csvfile:
        counter += 0

        if counter <= 1:
            next(csvfile)

        summary = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        for row in summary:
            for i in range(len(row)):
                row[i] = convert_data(row[i])

            time_taken += row[1]

            if row[5] == 'train':
                if row[7] > max_train_accuracy:
                    max_train_accuracy = row[7]

            if row[5] == 'val':
                number_trained += 1
                if row[7] > max_val_accuracy:
                    max_val_accuracy = row[7]
                    best_epoch = number_trained

            if len(row) >= 12:
                data['log_version'] = '1.1'

                if row[5] == 'train':
                    max_train_accuracy_5 = row[9] if row[9] > max_train_accuracy_5 else max_train_accuracy_5

                if row[5] == 'val':
                    max_val_accuracy_5 = row[9] if row[9] > max_val_accuracy_5 else max_val_accuracy_5

    data['csv_path_validated'] = get_validated_path_from_model(data['model_path'], True)
    data['time_taken'] = time_taken
    data['max_train_accuracy'] = max_train_accuracy
    data['max_val_accuracy'] = max_val_accuracy
    data['max_train_accuracy_5'] = max_train_accuracy_5
    data['max_val_accuracy_5'] = max_val_accuracy_5
    data['number_trained'] = number_trained
    data['best_epoch'] = best_epoch
    data['label'] = os.path.basename(data['process_path'])
    data['class_name'] = properties[0] if len(properties) >= 4 else None
    data['multi_model'] = True if len(properties) >= 4 else False
    data['main_class'] = data['process_path'].split('/')[2]
    data['time_start'] = datetime.utcfromtimestamp(
        int(creation_date(data['csv_path_settings']))
    ).replace(tzinfo=timezone.utc).astimezone(tz=None).strftime('%y-%m-%d %H:%M')

    return data


def get_folders(path, includes=[], excludes=[]):

    assert(len(includes) > 0)

    folders = []

    # search within given paths for folders
    for name in os.listdir(path):
        full_path = os.path.join(path, name)

        if not os.path.isdir(full_path):
            continue

        if name in excludes:
            continue

        if name in includes:
            folders.append(full_path)
            continue

        for folder in get_folders(full_path, includes, excludes):
            folders.append(folder)

    return folders


def get_datas_sorted_by(args, sortedBy='max_val_accuracy'):

    # collect all configs
    setting_files = []

    # get all csv folders within given path
    csv_folders = get_folders(args.path, ['csv'], ['data'])

    # find all setting files
    for csv_folder in csv_folders:
        for file in glob.glob('{}/**/{}'.format(csv_folder, 'settings*.csv'), recursive=True):
            setting_files.append(file)

    # get filters
    filters = get_filters(args)

    # collect all datas
    datas = []
    for setting_file in setting_files:
        data = get_data(setting_file)

        filter_ok = True

        for filter_arr in filters:
            for name in filter_arr:
                if name in data:
                    if str(data[name]) != str(filter_arr[name]):
                        filter_ok = False

        if not filter_ok:
            continue

        datas.append(data)

    # sort datas
    datas.sort(key=lambda x: x[sortedBy], reverse=True)
    return datas


def get_data_grouped_by_point_of_interest(datas, fields, args):
    data_grouped = {}

    if args.point_of_interest is None:
        data_grouped['all'] = []

        for index_data in range(len(datas)):
            data_grouped['all'].append(datas[index_data])
    else:
        fields.remove(args.point_of_interest)

        for index_data in range(len(datas)):
            key = ''
            data = datas[index_data]
            for field in fields:
                key += '_' if key else ''
                key += str(data[field])

            if not key in data_grouped:
                data_grouped[key] = []

            data_grouped[key].append(datas[index_data])

    return data_grouped
