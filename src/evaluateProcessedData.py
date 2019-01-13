#!/usr/bin/env python

"""
Processed data evaluator. Evaluates all processed data produced by vendor/pytorch-examples/imagenet/main.py
(bin/train).


Usage:

user$ src/evaluateProcessedData.py data/processed/food/unbalanced/90_10/elements/all/csv


TODO: add some more fields to consider (point of interest fields)
- auto model and csv finder (independent from given settings file path)
- label (number of learned elements: all, 500, etc.)
- device

TODO: furthermore
- net accuracy (information gain)
- filter argument
- sort argument (currently accuracy downwards)
- better argument extraction (with labels; independent from argument order)
- db to improve the performance reading the evaluate processed data
"""

__author__ = "Björn Hempel"
__copyright__ = "Copyright 2019, An ixnode project"
__credits__ = ["Björn Hempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Björn Hempel"
__email__ = "bjoern@hempel.li"
__status__ = "Production"


import pprint
import sys
import argparse

from __print import *
from __data import *
from __args import *

# create the argument parser
parser = argparse.ArgumentParser(description='ImageNet Training Evaluation')

# add all arguments to parser
parser.add_argument('path',
    help='Path to find the setting csv files', metavar='PATH')
parser.add_argument('-p', '--point-of-interest', default=None, type=str, metavar='POINT_OF_INTEREST',
    help='Group the output by point of interest (default: None)')
parser.add_argument('-o', '--order-by', default='acc', type=str, metavar='ORDER_BY',
    help='Group the output by point of interest (default: acc)')
parser.add_argument('-s', '--show-legend', action='store_true',
    help='Show legend if given')
parser.add_argument('-d', '--devider', default=5, type=int, metavar='DEVIDER',
    help='The number after which the output is to be optically separated by a separator line. (default: 5)')

# parse all arguments
args = parser.parse_args()

# fields that can be grouped by
fields = ['arch', 'epochs', 'batch_size', 'lr', 'weight_decay', 'momentum', 'linear_layer', 'workers']

# pretty printer
pp = pprint.PrettyPrinter(indent=4)

# check point of interest
check_point_of_interest(fields, args)

# get datas and group them
datas_grouped = get_data_grouped_by_point_of_interest(get_datas_sorted_by(args.path), fields, args)

# print datas
print_datas_grouped(fields, args, datas_grouped)
