import csv

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