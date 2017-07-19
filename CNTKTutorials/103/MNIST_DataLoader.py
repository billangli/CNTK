"""
MNIST_DataLoader.py
CNTK Tutorial 103
Bill Li
Jul. 18th, 2017
"""

# Importing modules to be used later
from __future__ import print_function

import gzip
import struct

try:
    from urllib.request import urlretrieve
except:
    from urllib import urlretrieve

### Data Download

# Functions to load MNIST images
"""loadData reads image data and formats into a 28x28 long array"""


def loadData(src, cimg):
    print('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read a magic number
