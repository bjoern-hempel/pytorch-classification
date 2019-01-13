import math
import os


def get_columns():
    """Returns all columns and the width of them"""

    # print(sum([v[0] for v in columns.values()]))
    return {
        'arch': [13, '<13', '>13', None],
        'acc': [6, '<6', '5.2f', '{}%'],
        'acc5': [6, '<6', '5.2f', '{}%'],
        'main_class': [5, '<5', '<5', None],
        'label': [5, '<5', '<5', None],
        'epochs': [2, '<2', '2d', None],
        'trained_epochs': [2, '<2', '2d', None],
        'best_epoch': [2, '<2', '2d', None],
        'batch_size': [2, '<2', '2d', None],
        'date': [14, '<14', '<14', None],
        'timeFormated': [8, '<8', None, None],
        'model_size': [10, '<10', '7.2f', '{} MB'],
        'log_version': [4, '<4', '>3', 'v{}'],
        'device': [7, '<7', '<7', None],
        'settings_name': [30, '<30', '<30', None]
    }


def get_len_str(with_margin=True):

    space_left = 2
    space_right = 2

    space_width = 3

    columns = get_columns()

    len_str = (len(columns) - 1) * space_width + sum(columns.values())

    if with_margin:
        len_str += space_left + space_right

    return len_str


def get_line():
    """"""
    line = ''

    columns = get_columns()

    for field_name in columns:
        line += '' if line == '' else '-+-'
        line += '-' * columns[field_name][0]

    return '+-' + line + '-+'


def get_format_string_header():

    format_string = ''

    columns = get_columns()

    for field_name in columns:
        format_string += '' if format_string == '' else ' | '
        format_string += '{{{}:{}}}'.format(field_name, columns[field_name][1])

    return '| {} |'.format(format_string)


def get_format_string_row():

    format_string = ''

    columns = get_columns()

    for field_name in columns:
        format_string += '' if format_string == '' else ' | '

        if columns[field_name][2] is not None:
            format_string_field = '{{{}:{}}}'.format(field_name, columns[field_name][2])
        else:
            format_string_field = '{{{}}}'.format(field_name)

        if columns[field_name][3]:
            format_string_field = columns[field_name][3].format(format_string_field)

        format_string += format_string_field

    return '| {} |'.format(format_string)


def print_header(fields=None, point_of_interest=None, data=None):

    print('')

    if point_of_interest is not None:
        if data is not None and fields is not None:
            caption_str = ''
            for field in fields:
                caption_str += '; ' if caption_str else ''
                caption_str += '{}: {}'.format(field, str(data[field]))
            print('{}'.format(caption_str))

    print(get_line())
    print(
        get_format_string_header().format(
            arch='model',
            acc='acc 1',
            acc5='acc 5',
            main_class='mc',
            label='label',
            epochs='ep',
            trained_epochs='te',
            best_epoch='be',
            batch_size='bs',
            date='start',
            timeFormated='duration',
            model_size='size',
            log_version='vers',
            device='device',
            time='sec',
            settings_name='settings file name'
        )
    )
    print(get_line())


def print_data(fields, point_of_interest, data, counter):
    if counter == 0:
        print_header(fields, point_of_interest, data)

    time_hours = math.floor(data['time_taken'] / 3600)
    time_minutes = math.floor((data['time_taken'] - time_hours * 3600) / 60)
    time_seconds = math.floor(data['time_taken'] - time_hours * 3600 - time_minutes * 60)

    time_formated = '{:02d}:{:02d}:{:02d}'.format(time_hours, time_minutes, time_seconds)

    settings_name = os.path.basename(data['csv_path_settings'])
    if len(settings_name):
        settings_name = '...' + os.path.basename(data['csv_path_settings'])[-27:]

    print(
        get_format_string_row().format(
            arch=data['arch'],
            acc=data['max_val_accuracy'],
            acc5=data['max_val_accuracy_5'],
            main_class=data['main_class'],
            label=data['label'],
            epochs=data['epochs'],
            trained_epochs=data['number_trained'],
            best_epoch=data['best_epoch'],
            batch_size=data['batch_size'],
            date=data['time_start'],
            timeFormated=time_formated,
            model_size=data['model_size'] / 1024 / 1024,
            log_version=data['log_version'],
            device='gtx1060',
            settings_name=settings_name
        )
    )


def print_datas(fields, point_of_interest, datas):
    counter = 0
    for data in datas:

        if counter != 0 and counter % 5 == 0:
            print(get_line())

        print_data(fields, point_of_interest, data, counter)

        counter += 1
    print(get_line())


def print_legend():
    print('')
    print('* model:    The name of the used deep neural network')
    print('* acc 1:    Accuracy of predicted best class')
    print('* acc 5:    Accuracy of predicted best five classes')
    print('* mc:       The main class (the name of the training set)')
    print('* label:    The label (trained / validated data set) under which the training was carried out')
    print('* ep:       The number of epochs to be trained')
    print('* te:       The number of epochs already trained')
    print('* be:       The epoch with the best accuracy')
    print('* bs:       The batch size under which the training was performed')
    print('* start:    The start time at which the training was started')
    print('* duration: The time needed for the training')
    print('* size:     The required memory for all parameters of the network')
    print('* vers:     The version of the logging file')
    print('* device:   The device on which the training was performed')


def print_datas_grouped(fields, point_of_interest, datas_grouped):
    for key, datas in datas_grouped.items():
        print_datas(fields, point_of_interest, datas)
    print_legend()
    print('')