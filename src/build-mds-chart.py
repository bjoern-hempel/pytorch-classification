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


import pprint

from __classes import *
from __mds import *
from file_helper.__file import *


# Configure the pretty printer
pp = pprint.PrettyPrinter(indent=4)

# some markers and colors to mark the points on map
markers = {
    'signs': ['o', 'v', '^', '<', '>', 'X', 's', 'p', 'P', '*'],
    'colors': ['#ff0000', '#00ff00', '#0000ff', '#808000', '#008080', '#800080']
}

# some other variables
build_pdf = True
build_png = True
show_chart = False
increase_output = True
show_points = False

# path to csv
csv_path = 'data/processed/food/unbalanced/90_10/elements/all/csv/densenet201/224x224/gpu1060/validated_lr0.001_m0.9_bs8_w4_wd0.0001_p.20190112_011853.20190120_212304.csv'
accuracy = 0.8627

# get dict from all classes
dict_translate_class = get_dict_translate_class()

# get config
config = analyse_file_and_get_config(csv_path)

# get all class points (multidimensional)
P = get_points_from_csv(csv_path, dict_translate_class)

# print the given points
if show_points:
    print_points(P, 'Given points')

# get all class points (2-dimensional)
P_2D = normalize_points(multidimensional_scaling(get_distances(P), 2))

# print the calculated points
if show_points:
    print_points(P_2D, 'Multidimensional scaled points')

# build the chart
(plt, ax) = build_mds(P_2D, config, accuracy, dict_translate_class, markers)

# build the pdf from chart
if build_pdf:
    path_pdf = config['files']['pdf.mds']['path']
    print('Save document to {}'.format(path_pdf))
    plt.savefig(path_pdf)

# build the png from chart
if build_png:
    path_png = config['files']['png.mds']['path']
    print('Save document to {}'.format(path_png))
    plt.savefig(path_png)

if show_chart:
    plt.show()
