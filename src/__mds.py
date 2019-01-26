import csv
import math
import matplotlib.pyplot as plt

from numpy import *
from numpy.linalg import *
from numpy.random import *


def print_points(points, title = None):
    var_names = ['x', 'y', 'z']

    if title is not None:
        print('{}:'.format(title))

    for i in range(len(points)):
        output = ''
        for j in range(len(points[i])):
            output += '\t' if output else ''

            var_name = var_names[j] if len(var_names) >= len(points[i]) else 'x{}'.format(j)

            output += '{:<2} = {:6.1f}'.format(var_name, points[i][j])

        print(output)
    print('')


def get_distances(points):
    distances = zeros((50, 50))
    for (i, point_i) in enumerate(points):
        for (j, point_j) in enumerate(points):
            distances[i, j] = norm(point_i - point_j)

    return distances


def multidimensional_scaling(d, dimensions=2):
    E = (-0.5 * d**2)

    # Use mat to get column and row means to act as column and row means.
    Er = mat(mean(E, 1))
    Es = mat(mean(E, 0))

    # From Principles of Multivariate Analysis: A User's Perspective (page 107).
    F = array(E - transpose(Er) - Es + mean(E))

    [U, S, V] = svd(F)

    Y = U * sqrt(S)

    return Y[:, 0:dimensions]


def normalize_points(points):

    x = -1
    w = 2

    y = -1
    h = 2

    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf

    for point in points:
        x_min = point[0] if point[0] < x_min else x_min
        x_max = point[0] if point[0] > x_max else x_max
        y_min = point[1] if point[1] < y_min else y_min
        y_max = point[1] if point[1] > y_max else y_max

    width = (x_max - x_min) / w
    height = (y_max - y_min) / h

    for point in points:
        point[0] = (point[0] - x_min) / width + x
        point[1] = (point[1] - y_min) / height + y

    return points


def norm(vec):
    return sqrt(sum(vec**2))


def get_points_from_csv(csv_path, dict_translate_class):
    data = {}
    P = []

    # initialize the class counter
    count = {'all': 0}
    for class_name in dict_translate_class:
        count[class_name] = 0

    # read csv
    counter = 0
    print('Read csv: {}'.format(csv_path))
    with open(csv_path, newline='') as csvfile:
        counter += 1

        if counter <= 1:
            next(csvfile)

        settings = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        for row in settings:
            counter += 1

            real_class = row[4]

            if not real_class in data:
                data[real_class] = {}
                for class_name in dict_translate_class:
                    data[real_class][class_name] = 0

            for i in range(50):
                pred_class = row[2 * i + 5]
                pred_val = row[2 * i + 6]

                data[real_class][pred_class] += float(pred_val)

            count[real_class] += 1
            count['all'] += 1

    # build points
    for class_name in data:
        number = []
        for class_name_2 in data:
            value = data[class_name][class_name_2] / count[class_name]
            number.append(value)
        P.append(array(number))

    return P


def build_mds(points_2d, config, accuracy, translate_class, markers):
    """Builds the multidimensional scaling."""
    # decrease the font size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 18
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

    fig, ax = plt.subplots()

    keys = translate_class.keys()

    for i in range(len(points_2d)):
        point = points_2d[i]

        marker_index = i % len(markers['signs'])
        marker = markers['signs'][marker_index]

        color_index = math.floor(i / len(markers['signs']))
        color = markers['colors'][color_index]

        label = translate_class[list(keys)[i]]

        ax.scatter(*point, color=color, marker=marker, alpha=0.5, s=100, edgecolors='none', label=label)

    ax.legend(loc=9, bbox_to_anchor=(0.5, 0.18), ncol=5)
    ax.grid(True)

    title = 'Cluster analysis (multidimensional scaling) "{}; {}%; 90 epochs"'.format(
        config['property_path'],
        accuracy * 100
    )

    ax.set_title(title)

    axes = plt.gca()
    axes.set_xlim([-1.1, 1.1])
    axes.set_ylim([-1.5, 1.1])

    return plt, ax