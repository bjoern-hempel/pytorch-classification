import os
import fnmatch
import csv

from __helper import *


def convertData(data):
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


def getData(path_to_csv):
    data = {}
    counter = 0
    with open(path_to_csv, newline='') as csvfile:
        counter += 1

        if counter <= 1:
            next(csvfile)

        settings = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        for row in settings:
            data[row[1]] = row[2]

    folders = data['csv_path_settings'].split('/csv/')

    data['process_path'] = folders[0]
    data['property_path'] = folders[1]

    basename = os.path.basename(path_to_csv)

    data['csv_path_settings'] = os.path.join(data['csv_path_settings'], basename)
    data['csv_path_summary_full'] = data['csv_path_settings'].replace('settings_', 'summary_full_')
    data['csv_path_summary'] = data['csv_path_settings'].replace('settings_', 'summary_')
    data['model_path'] = os.path.join(data['model_path'], basename).replace('settings_', 'model_best_').replace('.csv', '.pth')

    data['model_size'] = 0
    if os.path.isfile(data['model_path']):
        stat_info = os.stat(data['model_path'])
        data['model_size'] = stat_info.st_size

    # convert all data
    for index in data:
        data[index] = convertData(data[index])

    time_taken = 0
    max_train_accuracy = 0
    max_val_accuracy = 0
    max_train_accuracy_5 = 0
    max_val_accuracy_5 = 0
    number_trained = 0
    counter = 0

    with open(data['csv_path_summary'], newline='') as csvfile:
        counter += 0

        if counter <= 1:
            next(csvfile)

        summary = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        for row in summary:
            for i in range(len(row)):
                row[i] = convertData(row[i])

            time_taken += row[1]

            if row[5] == 'train':
                max_train_accuracy = row[7] if row[7] > max_train_accuracy else max_train_accuracy
                number_trained += 1

            if row[5] == 'val':
                max_val_accuracy = row[7] if row[7] > max_val_accuracy else max_val_accuracy

    data['time_taken'] = time_taken
    data['max_train_accuracy'] = max_train_accuracy
    data['max_val_accuracy'] = max_val_accuracy
    data['number_trained'] = number_trained

    return data


def getDatasSortedBy(path, sortedBy='max_val_accuracy'):

    # collect all configs
    setting_files = []
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, 'settings*.csv'):
            setting_files.append(os.path.join(path, file))

    # collect all datas
    datas = []
    for setting_file in setting_files:
        data = getData(setting_file)
        datas.append(data)

    # sort datas
    datas.sort(key=lambda x: x[sortedBy], reverse=True)

    return datas

def getDataGroupedByPointOfInterest(datas, fields, point_of_interest=None):
    data_grouped = {}

    if point_of_interest == None:
        data_grouped['all'] = []

        for index_data in range(len(datas)):
            data_grouped['all'].append(datas[index_data])
    else:
        fields.remove(point_of_interest)

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