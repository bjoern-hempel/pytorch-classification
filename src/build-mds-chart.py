#!/usr/bin/env python

"""
Build a 2D chart from given multidimensional input.


Usage:

user$ src/build-mds-chart.py


TODO: add some more fields to consider (point of interest fields)
- ...
"""

__author__ = "Björn Hempel"
__copyright__ = "Copyright 2019, An ixnode project"
__credits__ = ["Björn Hempel"]
__license__ = "MIT"
__version__ = "1.0.0 (2019-01-24)"
__maintainer__ = "Björn Hempel"
__email__ = "bjoern@hempel.li"
__status__ = "Production"


import matplotlib.pyplot as plt
import pprint

from __classes import *
from __mds import *


# Configure the pretty printer
pp = pprint.PrettyPrinter(indent=4)

# some markers and colors to mark the points on map
markers = ['o', 'v', '^', '<', '>', 'X', 's', 'p', 'P', '*']
colors = ['#ff0000', '#00ff00', '#0000ff', '#808000', '#008080', '#800080']

# some other variables
build_pdf = False
show_chart = True

# path to csv
csv_path = 'data/processed/food/unbalanced/90_10/elements/all/csv/densenet201/224x224/gpu1060/validated_lr0.001_m0.9_bs8_w4_wd0.0001_p.20190112_011853.20190120_212304.csv'

# get dict from all classes
dict_translate_class = get_dict_translate_class()

# get all class points (multidimensional)
P = get_points_from_csv(csv_path, dict_translate_class)

# print the given points
print_points(P, 'Given points')

# get all class points (2-dimensional)
P_2D = multidimensional_scaling(get_distances(P), 2)

# print the calculated points
print_points(P_2D, 'Multidimensional scaled points')

# set output size
if False:
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 18
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

fig, ax = plt.subplots()

keys = dict_translate_class.keys()

for i in range(len(P_2D)):
    point = P_2D[i]

    marker_index = i % len(markers)
    marker = markers[marker_index]

    color_index = math.floor(i / len(markers))
    color = colors[color_index]

    label = dict_translate_class[list(keys)[i]]

    ax.scatter(*point, color=color, marker=marker, alpha=0.5, s=100, edgecolors='none', label=label)

ax.legend(loc=9, bbox_to_anchor=(0.5, 0.18), ncol=5)
ax.grid(True)

title = 'Cluster analysis (multidimensional scaling) "densenet201; 86.27%; 90 epochs"'

ax.set_title(title)
axes = plt.gca()

if build_pdf:
    path_pdf = 'cluster-analysis.pdf'
    print('Save document to {}'.format(path_pdf))
    plt.savefig(path_pdf)

if show_chart:
    plt.show()
