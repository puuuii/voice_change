import time
import pandas as pd
import numpy as np
import numba
from pysptk import *
from scipy.io import wavfile as io
from functools import lru_cache


def main():
    path_datadir = './data/data/'
    cnt = 1
    arr_all = pd.DataFrame()

    # wavとlabのペアを作り結合していく
    while True:
        print(cnt)
        arr_lab = pd.DataFrame()
        arr_wav = pd.DataFrame()

        # wavデータ作成
        path_wav = path_datadir + ('%08d' % cnt) + '.wav'
        sr, data_wav = io.read(path_wav)
        # データが空ならdf作成終了
        if len(data_wav) == 0:
            break
        arr_wav = pd.concat([arr_wav, pd.DataFrame(data_wav)])

        # labデータ作成
        path_lab = path_datadir + ('%08d' % cnt) + '.lab'
        with open(path_lab, "r") as f:
            content_list = [line.split() for line in f.readlines()]
            # labファイル1行ずつのdfを作成し結合していく
            for content in content_list:
                arr_lab = pd.concat([arr_lab, make_arr_from_per_line(content)], ignore_index=True)

        # wavデータとlabデータの結合
        pair_arr = pd.concat([arr_wav, arr_lab], axis=1, join='inner')

        # 最終データ作成
        arr_all = pd.concat([arr_all, pair_arr])

        cnt += 1
        if cnt == 101:
            print(arr_all.shape)
            break


def make_arr_from_per_line(content):
    """
    1行のlabデータを1塊のdfにして返す
    :param content: 1行のlabデータ
    :return:        1塊のdf
    """
    start_time = float(content[0])
    end_time = float(content[1])
    n_frame = int(16000 * round((end_time - start_time), 4))
    phoneme = content[2]
    ret_arr = pd.DataFrame([phoneme for i in range(n_frame)])

    return ret_arr


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed = time.time() - start
    print("elapsed_time:{0}".format(elapsed) + "[sec]")