import math
import os



def get_len_str():
    len_str = 2 + 10 * 3 + 2
    len_str += 6 + 6 + 13 + 2 + 2 + 2 + 8 + 5 + 10 + 7 + 60

    return len_str


def print_header(fields=None, point_of_interest=None, data=None):

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
        '| {acc:<6} | {acc5:<6} | {arch:<13} | {epochs:<2} | {trained:<2} | {batch_size:<2} | {timeFormated:<8} | {time:<5} | {model_size:<10} | {device:<7} | {settings_name:<60} |'.format(
            acc='acc',
            acc5='acc5',
            arch='arch',
            epochs='ep',
            trained='tr',
            batch_size='bs',
            timeFormated='hh:mm:ss',
            model_size='model size',
            device='device',
            time='sec',
            settings_name='settings file name'
        )
    )
    print('-' * get_len_str())


def print_data(fields, point_of_interest, data, counter):
    if counter == 0:
        print_header(fields, point_of_interest, data)

    time_hours = math.floor(data['time_taken'] / 3600)
    time_minutes = math.floor((data['time_taken'] - time_hours * 3600) / 60)
    time_seconds = math.floor(data['time_taken'] - time_hours * 3600 - time_minutes * 60)

    time_formated = '{:02d}:{:02d}:{:02d}'.format(time_hours, time_minutes, time_seconds)

    print(
        '| {acc:5.2f}% | {acc5:5.2f}% | {arch:>13} | {epochs:2d} | {trained:2d} | {batch_size:2d} | {timeFormated} | {time:5.0f} | {model_size:7.2f} MB | {device:<7} | {settings_name:<60} |'.format(
            acc=data['max_val_accuracy'],
            acc5=0,
            arch=data['arch'],
            epochs=data['epochs'],
            trained=data['number_trained'],
            batch_size=data['batch_size'],
            timeFormated=time_formated,
            time=data['time_taken'],
            model_size=data['model_size'] / 1024 / 1024,
            device='gtx1060',
            settings_name=os.path.basename(data['csv_path_settings'])
        )
    )


def print_datas(fields, point_of_interest, datas):
    counter = 0
    for data in datas:
        print_data(fields, point_of_interest, data, counter)

        counter += 1
    print('-' * get_len_str())


def print_datas_grouped(fields, point_of_interest, datas_grouped):
    for key, datas in datas_grouped.items():
        print_datas(fields, point_of_interest, datas)
    print('')