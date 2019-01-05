import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import os
import csv

from matplotlib import cm, colors
from random import randint
from math import ceil, floor
from __file import *

# what to do?
showPlot = False
savePDF = True
savePNG = True

# train data elements
categorie = 'food'
dispersionType = 'unbalanced'
dispersion = '90_10'

# technique data elements
model = 'resnet152'
featureSize = '224x224'
validationSet = 'val'

# other variables
machine='gpu1060'

# build the
calculatedPath = '/media/bjoern/Daten/Development/classification/data/processed/{}/{}/{}/elements'.format(
    categorie,
    dispersionType,
    dispersion
)

# build csv, pdf and png path
pathCSV = '{}' + '/csv/{}/{}/{}/validated.csv'.format(
    model,
    featureSize,
    machine
)
pathPDF = '{}' + '/charts/{}/{}/{}'.format(
    model,
    featureSize,
    machine
) + '/confusion-matrix.{}.pdf'
pathPNG = '{}' + '/charts/{}/{}/{}'.format(
    model,
    featureSize,
    machine
) + '/confusion-matrix.{}.png'

# set output size
figSize = plt.rcParams["figure.figsize"]
figSize[0] = 18
figSize[1] = 12
plt.rcParams["figure.figsize"] = figSize


# translate dict: class name -> real name
translateClass = {
    'baked_beans': 'Baked Beans',
    'baked_salmon': 'Baked Salmon',
    'beef_stew': 'Beef Stew',
    'beef_stroganoff': 'Beef Stroganoff',
    'brownies': 'Brownies',
    'bundt_cake': 'Bundt Cake',
    'burger': 'Burger',
    'burrito': 'Burrito',
    'buttermilk_biscuits': 'Buttermilk Biscuits',
    'caesar_salad': 'Caesar Salad',
    'calzone': 'Calzone',
    'cheesecake': 'Cheesecake',
    'chicken_piccata': 'Chicken Piccata',
    'chicken_wings': 'Chicken Wings',
    'cinnamon_roll': 'Cinnamon Roll',
    'cobb_salad': 'Cobb Salad',
    'coleslaw': 'Coleslaw',
    'creamed_spinach': 'Creamed Spinach',
    'donut': 'Donut',
    'empanada': 'Empanada',
    'french_fries': 'French Fries',
    'frittata': 'Frittata',
    'granola_bar': 'Granola Bar',
    'grilled_cheese_sandwich': 'Grilled Cheese Sandwich',
    'guacamole': 'Guacamole',
    'ice_cream': 'Ice Cream',
    'kebabs': 'Kebabs',
    'key_lime_pie': 'Key Lime Pie',
    'lasagne': 'Lasagne',
    'macaroni_and_cheese': 'Macaroni and Cheese',
    'margarita': 'Margarita',
    'martini': 'Martini',
    'mashed_potatoes': 'Mashed Potatoes',
    'meatballs': 'Meatballs',
    'meatloaf': 'Meatloaf',
    'muffin': 'Muffin',
    'nachos': 'Nachos',
    'omelet': 'Omelet',
    'pancakes': 'Pancakes',
    'pizza': 'Pizza',
    'popcorn': 'Popcorn',
    'quesadilla': 'Quesadilla',
    'salad': 'Salad',
    'sloppy_joe': 'Sloppy Joe',
    'smoothie': 'Smoothie',
    'soup': 'Soup',
    'spaghetti': 'Spaghetti',
    'stuffed_pepper': 'Stuffed Pepper',
    'waffles': 'Waffles',
    'corn_dog': 'Corn Dog'
}


# calulate the hex value
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


def getIdOfGivenLabel(label):
    return list(translateClass.keys()).index(label)


def getNormalizedData(calculatedPath, label, numberClasses):
    data = np.zeros((numberClasses, numberClasses))

    with open(os.path.join(calculatedPath, pathCSV.format(label)), 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        counter = 0
        countClasses = np.zeros(numberClasses).astype(int)

        # collect the numbers of train and validation
        for row in plots:
            counter += 1

            # ignore header
            if counter == 1:
                continue

            realId = getIdOfGivenLabel(row[4])
            predId = getIdOfGivenLabel(row[5])

            countClasses[realId] += 1
            data[realId][predId] += 1

        # replace 0 > nan
        for i in range(50):
            for j in range(50):
                if data[i, j] == 0.0:
                    data[i, j] = np.nan
                else:
                    data[i, j] = data[i, j] / countClasses[i]

    return (data, countClasses)


def getClassNamesWithNumber(classNames, countClasses, data):
    classNamesReturn = []

    for i in range(len(classNames)):
        if np.isnan(data[i][i]):
            percent = 0
        else:
            percent = 100 * float(data[i][i])

        classNamesReturn.append('{} ({}; {:.1f}%)'.format(classNames[i], str(countClasses[i]), percent))

    return classNamesReturn


def getLabels(calculatePath):
    labels = []

    for label in os.listdir(calculatePath):
        if os.path.isdir(os.path.join(calculatePath, label)):
            labels.append(label)

    return labels


# get all labels from calculated path
labels = getLabels(calculatedPath)

# build confusion matrix from all labels
for label in labels:

    # check if csv is available
    if not os.path.isfile(os.path.join(calculatedPath, pathCSV.format(label))):
        continue

    (data, countClasses) = getNormalizedData(calculatedPath, label, len(translateClass))
    classNames = list(translateClass.values())
    classNamesWithNumber = getClassNamesWithNumber(classNames, countClasses, data)
    addColor = 0
    cmap = colors.ListedColormap([
        getHex(0 + addColor, 128 + addColor, 0 + addColor, 0.8),
        getHex(0 + addColor, 128 + addColor, 0 + addColor, 0.6),
        getHex(0 + addColor, 128 + addColor, 0 + addColor, 0.4),
        getHex(0 + addColor, 128 + addColor, 0 + addColor, 0.2),
        getHex(0 + addColor, 128 + addColor, 0 + addColor, 0)
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
    ax.set_xticks(np.arange(len(classNames)))
    ax.set_yticks(np.arange(len(classNamesWithNumber)))

    ax.set_xticklabels(classNames)
    ax.set_yticklabels(classNamesWithNumber)

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    # Loop over data dimensions and create text annotations.
    for i in range(len(classNames)):
        for j in range(len(classNames)):
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

    title = 'Confusion matrix "{}-{}" ({}/{}/{}/{}/{}) [%] '.format(
        label,
        validationSet,
        categorie,
        dispersionType,
        dispersion,
        model,
        featureSize
    )

    ax.set_title(title)
    fig.tight_layout()

    # save the diagram
    if savePDF:
        outputPDFPath = os.path.join(calculatedPath, pathPDF.format(label, validationSet))
        createFolderForFile(outputPDFPath)
        print('Save document to {}'.format(outputPDFPath))
        plt.savefig(outputPDFPath)
    if savePNG:
        outputPNGPath = os.path.join(calculatedPath, pathPNG.format(label, validationSet))
        createFolderForFile(outputPNGPath)
        print('Save document to {}'.format(outputPNGPath))
        plt.savefig(outputPNGPath)

    ax.grid(linestyle='-', linewidth=0.5, color=getHex(200, 200, 200))

    if showPlot:
        plt.show()