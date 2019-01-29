import math
import os


def get_columns():
    """Returns all columns and the width of them"""

    label_width = 5
    class_name_width = 19

    return {
        'arch': [13, '<13', '>13', None],
        'acc': [6, '<6', '5.2f', '{}%'],
        'acc5': [6, '<6', '5.2f', '{}%'],
        'main_class': [5, '<5', '<5', None],
        'class_name': [class_name_width, '<{}'.format(class_name_width), '<{}'.format(class_name_width), None],
        'label': [label_width, '<{}'.format(label_width), '<{}'.format(label_width), None],
        'epochs': [2, '<2', '2d', None],
        'trained_epochs': [2, '<2', '2d', None],
        'best_epoch': [2, '<2', '2d', None],
        'batch_size': [2, '<2', '2d', None],
        'date': [14, '<14', '<14', None],
        'time_formated': [8, '<8', None, None],
        'model_size': [10, '<10', '7.2f', '{} MB'],
        'log_version': [4, '<4', '>3', 'v{}'],
        'device': [7, '<7', '<7', None],
        'validated_file_available': [1, '<1', '<1', None],
        'multi_model': [1, '<1', '<1', None],
        'settings_name': [22, '<22', '<22', None],
        'settings_name_full': [100, '<100', '<100', None]
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


def get_line(args):
    """"""
    line = ''

    columns = get_columns()

    for field_name in args.fields:
        line += '' if line == '' else '-+-'
        line += '-' * columns[field_name][0]

    return '+-' + line + '-+'


def get_format_string_header(args):

    format_string = ''

    columns = get_columns()

    for field_name in args.fields:
        format_string += '' if format_string == '' else ' | '
        format_string += '{{{}:{}}}'.format(field_name, columns[field_name][1])

    return '| {} |'.format(format_string)


def get_format_string_row(args):

    format_string = ''

    columns = get_columns()

    for field_name in args.fields:
        format_string += '' if format_string == '' else ' | '

        if columns[field_name][2] is not None:
            format_string_field = '{{{}:{}}}'.format(field_name, columns[field_name][2])
        else:
            format_string_field = '{{{}}}'.format(field_name)

        if columns[field_name][3]:
            format_string_field = columns[field_name][3].format(format_string_field)

        format_string += format_string_field

    return '| {} |'.format(format_string)


def print_header(fields, args, data=None):

    print('')

    if args.point_of_interest is not None:
        if data is not None and fields is not None:
            caption_str = ''
            for field in fields:
                caption_str += '; ' if caption_str else ''
                caption_str += '{}: {}'.format(field, str(data[field]))
            print('{}'.format(caption_str))

    print(get_line(args))
    print(
        get_format_string_header(args).format(
            arch='model',
            acc='acc 1',
            acc5='acc 5',
            main_class='mc',
            class_name='class',
            label='label',
            epochs='ep',
            trained_epochs='te',
            best_epoch='be',
            batch_size='bs',
            date='start',
            time_formated='duration',
            model_size='size',
            log_version='vers',
            device='device',
            validated_file_available='v',
            multi_model='m',
            settings_name='settings file name',
            settings_name_full='settings file name'
        )
    )
    print(get_line(args))


def print_data(fields, args, data, counter):
    if counter == 0:
        print_header(fields, args, data)

    time_hours = math.floor(data['time_taken'] / 3600)
    time_minutes = math.floor((data['time_taken'] - time_hours * 3600) / 60)
    time_seconds = math.floor(data['time_taken'] - time_hours * 3600 - time_minutes * 60)

    time_formated = '{:02d}:{:02d}:{:02d}'.format(time_hours, time_minutes, time_seconds)

    columns = get_columns()

    settings_name = os.path.basename(data['csv_path_settings'])
    if len(settings_name) > columns['settings_name'][0]:
        settings_name = '...' + settings_name[-columns['settings_name'][0]+3:]

    settings_name_full = data['csv_path_settings'].split('/csv/')[1]
    if len(settings_name_full) > columns['settings_name_full'][0]:
        settings_name_full = '...' + settings_name_full[-columns['settings_name_full'][0]+3:]

    print(
        get_format_string_row(args).format(
            arch=data['arch'],
            acc=data['max_val_accuracy'],
            acc5=data['max_val_accuracy_5'],
            main_class=data['main_class'],
            class_name=data['class_name'],
            label=data['label'],
            epochs=data['epochs'],
            trained_epochs=data['number_trained'],
            best_epoch=data['best_epoch'],
            batch_size=data['batch_size'],
            date=data['time_start'],
            time_formated=time_formated,
            model_size=data['model_size'] / 1024 / 1024,
            log_version=data['log_version'],
            device='gtx1060',
            validated_file_available='-' if data['csv_path_validated'] is None else 'x',
            multi_model='-' if data['multi_model'] is None else 'x',
            settings_name=settings_name,
            settings_name_full=settings_name_full
        )
    )


def print_datas(fields, args, datas):
    counter = 0
    for data in datas:

        if counter != 0 and args.devider is not None and counter % args.devider == 0:
            print(get_line(args))

        print_data(fields, args, data, counter)

        counter += 1
    print(get_line(args))


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
    print('* v:        Shows whether the validated file has already been generated')
    print('* m:        Shows whether this is a multi validated model')


def print_datas_grouped(fields, args, datas_grouped):
    for key, datas in datas_grouped.items():
        print_datas(fields, args, datas)

    if args.show_legend:
        print_legend()

    print('')