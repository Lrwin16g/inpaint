# coding: UTF-8

import os, sys

if len(sys.argv) != 2:
    print 'Usage: ' + sys.argv[0] + ' <filepath>'
    sys.exit()

path = sys.argv[1]

for root, dirs, files in os.walk(path):
    for file in files:
        print root + os.sep + file
