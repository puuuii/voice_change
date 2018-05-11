import sys
import os
import re


def rename_wav():
    files = os.listdir('./')
    cnt = 1
    for file in files:
        if '.wav' not in file:
            continue

        newfilename = ('%08d' % cnt) + '.wav'
        os.rename(file, newfilename)

        cnt += 1


def rename_lab():
    files = os.listdir('./')
    cnt = 1
    for file in files:
        if '.lab' not in file:
            continue

        newfilename = ('%08d' % cnt) + '.lab'
        os.rename(file, newfilename, )

        cnt += 1


def remove():
    files = os.listdir('./')
    remove_list = []
    for file in files:
        if '.lab' not in file:
            continue

        data = open(file, "r")
        content = data.read()

        if content == "":
            number_str = file[:8]
            remove_wav = number_str + ".wav"
            remove_lab = number_str + ".lab"
            remove_list.append(remove_wav)
            remove_list.append(remove_lab)

        data.close()

    for file in remove_list:
        os.remove(file)


rename_lab()
rename_wav()
remove()
rename_lab()
rename_wav()
