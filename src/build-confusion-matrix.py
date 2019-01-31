#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import csv
import pprint

from matplotlib import colors
from math import ceil
from file_helper.__file import *
from __classes import *

# pretty printer
pp = pprint.PrettyPrinter(indent=4)

# what to do?
showPlot = False
savePDF = True
savePNG = True

# train data elements
category = 'food'
dispersionType = 'unbalanced'
dispersion = '90_10'
label = 'all'

# technique data elements
model = 'resnet152'
featureSize = '224x224'
validationSet = 'val'

# other variables
machine='gpu1060'

# build the root path
rootPath = 'data/processed/{}/{}/{}/elements/{}'.format(
    category,
    dispersionType,
    dispersion,
    label
)

# get all validated files
validated_files = search_files(os.path.join(rootPath, 'csv'), 'validated_*.csv')

# translate dict: class name -> real name
translateClass = get_dict_translate_class()

# calculate the hex value
def getHex(r, g, b, f=0):
    # calculate the color factor
    r = ceil(r + ((255 - r) * f))
    g = ceil(g + ((255 - g) * f))
    b = ceil(b + ((255 - b) * f))

    # normalize r(ed)
    r = r if r <= 255 else 255
    r = r if r >= 0 else 0

    # normalize g(reen)
    g = g if g <= 255 else 255
    g = g if g >= 0 else 0

    # normalize b(lue)
    b = b if b <= 255 else 255
    b = b if b >= 0 else 0

    return '#' + format(r, 'x').zfill(2) + format(g, 'x').zfill(2) + format(b, 'x').zfill(2)


def get_id_of_given_label(label):
    return list(translateClass.keys()).index(label)


def get_normalized_data(path_csv, numberClasses):
    data = np.zeros((numberClasses, numberClasses))

    print('Load csv path: {}'.format(path_csv))
    with open(path_csv, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        counter = 0
        countClasses = np.zeros(numberClasses).astype(int)

        count_classes_all = 0
        correct_predicted = 0

        # collect the numbers of train and validation
        for row in plots:
            counter += 1

            # ignore header
            if counter == 1:
                continue

            realId = get_id_of_given_label(row[4])
            predId = get_id_of_given_label(row[5])

            countClasses[realId] += 1
            data[realId][predId] += 1

            count_classes_all += 1
            if realId == predId:
                correct_predicted += 1


        # replace 0 > nan
        for i in range(numberClasses):
            for j in range(numberClasses):
                if data[i, j] == 0.0:
                    data[i, j] = np.nan
                else:
                    data[i, j] = data[i, j] / countClasses[i]

    if count_classes_all != 0:
        accuracy = 100 * correct_predicted / count_classes_all
    else:
        accuracy = 0

    return data, countClasses, accuracy


def get_class_names_with_number(class_names, count_classes, data):
    class_names_return = []

    for i in range(len(class_names)):
        if np.isnan(data[i][i]):
            percent = 0
        else:
            percent = 100 * float(data[i][i])

        class_names_return.append('{} ({}; {:.1f}%)'.format(class_names[i], str(count_classes[i]), percent))

    return class_names_return


def build_confusion_matrix(config, translate_class):
    # set output size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 18
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

    # calculate data
    (data, count_classes, accuracy) = get_normalized_data(config['files']['csv.validated']['path'], len(translate_class))
    class_names = list(translate_class.values())
    class_names_with_number = get_class_names_with_number(class_names, count_classes, data)
    add_color = 0
    cmap = colors.ListedColormap([
        getHex(0 + add_color, 128 + add_color, 0 + add_color, 0.8),
        getHex(0 + add_color, 128 + add_color, 0 + add_color, 0.6),
        getHex(0 + add_color, 128 + add_color, 0 + add_color, 0.4),
        getHex(0 + add_color, 128 + add_color, 0 + add_color, 0.2),
        getHex(0 + add_color, 128 + add_color, 0 + add_color, 0)
    ])

    fig, ax = plt.subplots()
    im = ax.imshow(
        data,
        interpolation='nearest',
        cmap=cmap,
        vmin=0,
        vmax=1
    )
    im.set_zorder(1)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names_with_number)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names_with_number)

    # set grid, so we can analyse the chart easily (set this grid below the text)
    ax.grid(linestyle='-', linewidth=0.5, alpha=0.25, color=getHex(0, 0, 0))
    ax.set_axisbelow(True)

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    # Loop over data dimensions and create text annotations.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(
                j,
                i,
                "{:.0f}".format(data[i, j] * 100),
                fontsize=8,
                ha="center",
                va="center",
                color="w"
            )
            text.set_zorder(2)

    title = 'Confusion matrix "{}" ({}) [{:.2f}%] '.format(
        validationSet,
        config['property_path'],
        accuracy
    )

    ax.set_title(title)
    fig.tight_layout()

    return plt, ax


for validated_file in validated_files:
    config = analyse_file_and_get_config(validated_file)

    # cancel if validated csv file does not exist
    if not config['files']['csv.validated']['exists']:
        print('Validated CSV file "{}" was not found.'.format(config['files']['csv.validated']['path']))
        continue

    # build the chart
    (plt, ax) = build_confusion_matrix(config, translateClass)

    # save the diagram (pdf)
    if savePDF:
        path_pdf = config['files']['pdf.confusion_matrix_val']['path']
        create_folder_for_file(path_pdf)
        print('Save document to {}'.format(path_pdf))
        plt.savefig(path_pdf)

    # save the diagram (png)
    if savePNG:
        path_png = config['files']['png.confusion_matrix_val']['path']
        create_folder_for_file(path_png)
        print('Save document to {}'.format(path_png))
        plt.savefig(path_png)

    if showPlot:
        plt.show()
        exit()
