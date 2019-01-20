#!/usr/bin/env python

"""
Reduces the dimensionality of given points.


Usage:

user$ src/dimensionality-reduction-2d.py


TODO: add some more fields to consider (point of interest fields)
- ...
"""

__author__ = "Björn Hempel"
__copyright__ = "Copyright 2019, An ixnode project"
__credits__ = ["Björn Hempel"]
__license__ = "MIT"
__version__ = "1.0.0 (2019-01-19)"
__maintainer__ = "Björn Hempel"
__email__ = "bjoern@hempel.li"
__status__ = "Production"


import pprint
import math
import sys
import matplotlib.pyplot as plt


# Configure the pretty printer
pp = pprint.PrettyPrinter(indent=4)


# PointCalculationError class
class PointCalculationError(Exception):
    pass


def get_interbreeding_indexes(index_count, including_same = False, cross = False, deep = 0) -> 'list':
    """Returns an array combining each with each element.

    Parameters
    ----------
    index_count : int
        The number of elements to interbreed.
    including_same : bool
        Combinations are possible.
    cross : bool
        The order of the returned indices is significant.

    Returns
    -------
    list
        Returns a list (array) of index combinations.

    """
    indexes = []
    correct = 1 if not including_same else 0
    for i in range(index_count - correct):
        index_from = deep
        index_to = i + deep + correct

        indexes.append([
            index_from,
            index_to
        ])

        if cross and index_from != index_to:
            indexes.append([
                index_to,
                index_from
            ])

    if index_count - correct > 1:
        indexes += get_interbreeding_indexes(index_count - 1, including_same, cross, deep + 1)

    return indexes


def calculate_distance(P1, P2):
    """Calculates the distance of given points (1D to infinity-D)."""
    if len(P1) != len(P2):
        raise ValueError('Different dimension of given points.')

    square_sum = 0
    for i in range(len(P1)):
        square_sum += (P1[i] - P2[i])**2

    return square_sum**(1 / 2)


def solve_quadratic_equation(a, b, c, direction='right'):
    """Solves the quadratic equation of given a, b and c factor.

    The calculation side can be specified using the "direction" variable (right or left).
    """
    s = b ** 2 - 4 * a * c

    # Ignore very small negative numbers (to avoid the following negative root square error)
    if -1e-10 < s < 0:
        s = 0

    # negative root square error
    if s < 0:
        raise ValueError('Try to calculate the square root of 2 of negative number ({})!'.format(s))

    if direction == 'right':
        return (-b + s**(1 / 2)) / (2 * a)
    else:
        return (-b - s ** (1 / 2)) / (2 * a)


def get_point_from_lengths_x_first(xA, yA, xB, yB, l2, l3, direction='right'):
    """Calculate x first (yA and yB must not be the same)"""

    A = ((l2**2 - l3**2) + (xB**2 - xA**2) + (yB**2 - yA**2)) / (2 * (yB - yA))
    B = (xA - xB) / (yB - yA)

    a = B**2 + 1
    b = 2 * A * B - 2 * xA - 2 * yA * B
    c = A**2 + xA**2 + yA**2 - l2**2 - 2 * yA * A

    xC = solve_quadratic_equation(a, b, c, direction)
    yC = A + B * xC

    return [xC, yC]


def get_point_from_lengths_y_first(xA, yA, xB, yB, l2, l3, direction='right'):
    """Calculate y first (xA and xB must not be the same)"""

    A = ((l2**2 - l3**2) + (xB**2 - xA**2) + (yB**2 - yA**2)) / (2 * (xB - xA))
    B = (yA - yB) / (xB - xA)

    a = B**2 + 1
    b = 2 * A * B - 2 * yA - 2 * xA * B
    c = A**2 + xA**2 + yA**2 - l2**2 - 2 * xA * A

    yC = solve_quadratic_equation(a, b, c, direction)
    xC = A + B * yC

    return [xC, yC]


def get_point_from_lengths(point_A, point_B, length_2, length_3, direction='right'):
    """Calculate the point from 2 points and 2 lengths with given direction (right or left)."""
    xA = point_A[0]
    yA = point_A[1]
    xB = point_B[0]
    yB = point_B[1]

    # the same point was given
    if xA == xB and yA == yB:
        if length_2 == 0 and length_3 == 0:
            return point_A
        else:
            raise ValueError('Two same points are given, but l2 and l3 are greater than 0!')

    # calculate the distance of P_A and P_B
    length_1 = calculate_distance(point_A, point_B)

    # One edge length must not be longer than the other two.
    assert(length_1 <= length_2 + length_3), 'The length of l2 and l3 must be greater than l1!'
    assert(length_2 <= length_1 + length_3), 'The length of l1 and l3 must be greater than l2!'
    assert(length_3 <= length_1 + length_2), 'The length of l1 and l2 must be greater than l3!'

    dist_y = abs(yA - yB)
    dist_x = abs(xA - xB)

    # the point is not in the same y postion
    if dist_y > dist_x:
        return get_point_from_lengths_x_first(xA, yA, xB, yB, length_2, length_3, direction)

    # the point is not in the same x postion
    elif dist_x > dist_y:
        return get_point_from_lengths_y_first(xA, yA, xB, yB, length_2, length_3, direction)

    # same distance
    return get_point_from_lengths_x_first(xA, yA, xB, yB, length_2, length_3, direction)


def get_points_from_lengths(point_A, point_B, length_1, length_2):
    """Corresponds to function get_point_from_lengths, but calculates both directions left and right and
    returns the result as an array."""
    return [
        get_point_from_lengths(point_A, point_B, length_1, length_2, 'right'),
        get_point_from_lengths(point_A, point_B, length_1, length_2, 'left')
    ]


def get_best_distances(distances, points_base, points_compare):
    """Calculate the best distance for points_base[0] and points_base[1] to the two given points points_compare."""
    indexes = get_interbreeding_indexes(2, including_same=True, cross=True)

    for i in range(len(indexes)):
        distance = calculate_distance(
            points_base[indexes[i][0]],
            points_compare[indexes[i][1]]
        )

        if distance < distances[indexes[i][0]]:
            distances[indexes[i][0]] = distance

    return distances


def get_point(previously_calculated_points, length_array, current_point, cancel_distance = None):
    """Calculate the best "current" point with the condition of all given length_array."""
    data = []

    # prepare data array
    for i in range(current_point):
        data.append({
            'point': previously_calculated_points[i],
            'length': length_array[i][current_point]
        })

    # the number of given data must be at least the number of two
    if len(data) < 2:
        raise ValueError('Unsupported number of data!')

    # if we do have the number of two of given data elemens, we can still choose one direction (left or right)
    if len(data) == 2:
        try:
            point = get_point_from_lengths(
                data[0]['point'],
                data[1]['point'],
                data[0]['length'],
                data[1]['length'],
                'right'
            )
        except AssertionError as error:
            raise PointCalculationError()

        return point

    # len of data is higher than 2 -> we do have to compare the direction of calculated point
    if len(data) >= 3:
        points = []
        distances = [math.inf, math.inf]

        indexes = get_interbreeding_indexes(len(data))

        for i in range(len(indexes)):
            # first save base points (left and right); one of these both is the right one
            if len(points) == 0:
                try:
                    points = get_points_from_lengths(
                        data[indexes[i][0]]['point'],
                        data[indexes[i][1]]['point'],
                        data[indexes[i][0]]['length'],
                        data[indexes[i][1]]['length']
                    )
                except AssertionError:
                    continue
                continue

            try:
                compare_points = get_points_from_lengths(
                    data[indexes[i][0]]['point'],
                    data[indexes[i][1]]['point'],
                    data[indexes[i][0]]['length'],
                    data[indexes[i][1]]['length']
                )
            except AssertionError:
                continue

            distances = get_best_distances(
                distances,
                points,
                compare_points
            )

            # cancel, if the distance to compared point is lower than cancel_distance
            if cancel_distance is not None:
                for j in range(2):
                    if distances[j] < cancel_distance:
                        return points[j]

        # get the point with the lowest distance (this point is on the right side)
        if distances[0] < distances[1]:
            return points[0]
        else:
            return points[1]

    return None


def calculate_length_array(P):
    """Calculate the length array of given points."""
    length_array = []

    # prepare some data
    for i in range(len(P)):
        length_array.append([])
        for j in range(len(P)):
            length_array[i].append(calculate_distance(P[i], P[j]))

    return length_array


def calculate_points_from_lengths(length_array):
    """Calculates all points from given length_array."""
    points_calculated = [[]] * len(length_array)

    for current_point in range(len(length_array)):
        if current_point == 0:
            points_calculated[current_point] = [0, 0]
            continue

        if current_point == 1:
            points_calculated[current_point] = [0, length_array[0][1]]
            continue

        try:
            point = get_point(points_calculated, length_array, current_point, 1e-10)
        except PointCalculationError:
            sys.exit('It was not possible to calculate the current point ({}).'.format(current_point))

        points_calculated[current_point] = point

    return points_calculated


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



P = [
    [1, 1],
    [1, 3],
    [3, 2],
    [4, 3],
    [4, 6],
    [6, 2]
]

P = [
    [4,   6, 1, -10,  0],
    [1,   3, 1,  10, 10],
    [3,   2, 1, -10,  2],
    [1,   1, 1, -10,  3],
    [4,   3, 0, -10,  0],
    [6,   2, 0, -10,  0],
    [10, 10, 0,  10,  0]
]

P = [
    [1, 1],
    [2, 2],
    [3, 3]
]

print_points(P, 'Given points')

# calculate the length array from given points above
lengths = calculate_length_array(P)

# and now try to calculate the points from given points
P_calculated = calculate_points_from_lengths(lengths)

# print the calculated points
print_points(P_calculated, 'Calculated points')

plt.scatter(*zip(*P_calculated))
plt.show()
