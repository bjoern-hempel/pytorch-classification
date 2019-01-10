import csv
import fnmatch
import os
import pprint
import copy
import math
import sys

from __print import *
from __data import *
from __args import *

# path in which to search for settings
path = 'data/processed/food/unbalanced/90_10/elements/all/csv/resnet18/224x224/gpu1060'

# fields that can be grouped by
fields = ['arch', 'epochs', 'batch_size', 'lr', 'weight_decay', 'momentum', 'linear_layer', 'workers']

# point of interest: None (show all together) or one element of fields array
point_of_interest = None if len(sys.argv) < 2 else sys.argv[1] #'batch_size'

# pretty printer
pp = pprint.PrettyPrinter(indent=4)

# check point of interest
check_point_of_interest(fields, point_of_interest)

# get datas and group them
datas_grouped = getDataGroupedByPointOfInterest(getDatasSortedBy(path), point_of_interest)

# print datas
print_datas_grouped(fields, point_of_interest, datas_grouped)
