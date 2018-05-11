import sys
import os
import re


files = os.listdir('./')
for file in files:
    if '.wav' not in file:
        continue

    newfilename = int(file[0:3]) + 1
    newfilename = ('%04d' % newfilename) + '.wav'
    os.rename(file, newfilename)