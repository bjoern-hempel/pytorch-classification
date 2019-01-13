#!/usr/bin/env python

"""
Processed data evaluator. Evaluates all processed data produced by vendor/pytorch-examples/imagenet/main.py
(bin/train).

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

from __print import *
from __data import *
from __args import *

# path in which to search for settings
path = 'data/processed/food/unbalanced/90_10/elements/all/csv'

# fields that can be grouped by
fields = ['arch', 'epochs', 'batch_size', 'lr', 'weight_decay', 'momentum', 'linear_layer', 'workers']

# point of interest: None (show all together) or one element of fields array
point_of_interest = None if len(sys.argv) < 2 else sys.argv[1]

# order field
order_by = 'acc'

# pretty printer
pp = pprint.PrettyPrinter(indent=4)

# check point of interest
check_point_of_interest(fields, point_of_interest)

# get datas and group them
datas_grouped = get_data_grouped_by_point_of_interest(get_datas_sorted_by(path), fields, point_of_interest)

# print datas
print_datas_grouped(fields, point_of_interest, datas_grouped)
