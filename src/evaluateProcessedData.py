import csv
import fnmatch
import os
import pprint
import copy
import math
import sys

# path in which to search for settings
path = 'data/processed/food/unbalanced/90_10/elements/all/csv/resnet18/224x224/gpu1060'

# fields that can be grouped by
fields = ['arch', 'epochs', 'batch_size', 'lr', 'weight_decay', 'momentum', 'linear_layer', 'workers']

# point of interest: None (show all together) or one element of fields array
point_of_interest = None if len(sys.argv) < 2 else sys.argv[1] #'batch_size'

# pretty printer
pp = pprint.PrettyPrinter(indent=4)

# check point of interest
if point_of_interest is not None:
    if not point_of_interest in fields:
        print('Unknown point of interest: {}'.format(point_of_interest))
        print('Allowed fields: "{}"'.format('", "'.join(fields)))
        exit()


def isFloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


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

    # convert all data
    for index in data:
        data[index] = convertData(data[index])

    time_taken = 0
    max_train_accuracy = 0
    max_val_accuracy = 0
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


def get_len_str():
    len_str = 2 + 8 * 3 + 2
    len_str += 6 + 13 + 2 + 2 + 2 + 8 + 5 + 7 + 60

    return len_str


def printHeader(data=None, fields=None):

    print('')

    if point_of_interest is not None:
        if data is not None and fields is not None:
            caption_str = ''
            for field in fields:
                caption_str += '; ' if caption_str else ''
                caption_str += '{}: {}'.format(field, str(data[field]))
            print('{}'.format(caption_str))


    print('-' * get_len_str())
    print(
        '| {acc:<6} | {arch:<13} | {epochs:<2} | {trained:<2} | {batch_size:<2} | {timeFormated:<8} | {time:<5} | {device:<7} | {settings_name:<60} |'.format(
            acc='acc',
            arch='arch',
            epochs='ep',
            trained='tr',
            batch_size='bs',
            timeFormated='hh:mm:ss',
            device='device',
            time='sec',
            settings_name='settings file name'
        )
    )
    print('-' * get_len_str())


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

def getDataGroupedByPointOfInterest(datas, point_of_interest=None):
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

def printData(data, counter):
    if counter == 0:
        printHeader(data, fields)

    timeHours = math.floor(data['time_taken'] / 3600)
    timeMinutes = math.floor((data['time_taken'] - timeHours * 3600) / 60)
    timeSeconds = math.floor(data['time_taken'] - timeHours * 3600 - timeMinutes * 60)

    timeFormated = '{:02d}:{:02d}:{:02d}'.format(timeHours, timeMinutes, timeSeconds)

    print(
        '| {acc:5.2f}% | {arch:>13} | {epochs:2d} | {trained:2d} | {batch_size:2d} | {timeFormated} | {time:5.0f} | {device:<7} | {settings_name:<60} |'.format(
            acc=data['max_val_accuracy'],
            arch=data['arch'],
            epochs=data['epochs'],
            trained=data['number_trained'],
            batch_size=data['batch_size'],
            timeFormated=timeFormated,
            time=data['time_taken'],
            device='gtx1060',
            settings_name=os.path.basename(data['csv_path_settings'])
        )
    )


def printDatas(datas):
    counter = 0
    for data in datas:
        printData(data, counter)

        counter += 1
    print('-' * get_len_str())


def printDatasGrouped(datas_grouped):
    for key, datas in datas_grouped.items():
        printDatas(datas)


# get datas and group them
datas_grouped = getDataGroupedByPointOfInterest(getDatasSortedBy(path), point_of_interest)

# print datas
printDatasGrouped(datas_grouped)
print('')
