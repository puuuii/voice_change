import pandas as pd
import pickle
import pyworld as pw
import numpy as np
from scipy.io import wavfile as io
from sklearn.preprocessing import StandardScaler


FS = 16000
DIR_DATA = './data/data/'
DIR_PICKLES = './pickles/'

def main():
    cnt = 1
    list_all = []

    # wavとlabのペアを作り結合していく
    while True:
        print('process file:', cnt)
        arr_lab = pd.DataFrame()
        arr_wav = pd.DataFrame()

        # wavデータ作成
        path_wav = DIR_DATA + ('%08d' % cnt) + '.wav'
        try:
            sr, data_wav = io.read(path_wav)
        except FileNotFoundError:
            print('finish')
            break
        arr_wav = pd.concat([arr_wav, pd.DataFrame(data_wav)])

        # labデータ作成
        path_lab = DIR_DATA + ('%08d' % cnt) + '.lab'
        with open(path_lab, "r") as f:
            content_list = [line.split() for line in f.readlines()]
            # labファイル1行ずつのdfを作成し結合していく
            for content in content_list:
                arr_lab = pd.concat([arr_lab, make_arr_from_per_line(content)], ignore_index=True)

        # wavデータとlabデータの結合
        pair_arr = pd.concat([arr_wav, arr_lab], axis=1, join='inner')
        list_all.append(pair_arr)

        cnt += 1

    # 全データをdf化
    arr_all = pd.concat(list_all, ignore_index=True)

    # 'silE'と'silB'を'sil'に変換
    arr_all = arr_all.replace({'silB': 'sil', 'silE': 'sil'})

    # データの水増し
    arr_all = inflate(arr_all)

    # pickle化
    make_pickles(arr_all)


def make_arr_from_per_line(content):
    """
    1行のlabデータを1塊のdfにして返す
    :param content: 1行のlabデータ
    :return:        1塊のdf
    """

    start_time = float(content[0])
    end_time = float(content[1])
    n_frame = int(FS * round((end_time - start_time), 4))
    phoneme = content[2]
    ret_arr = pd.DataFrame([phoneme for i in range(n_frame)])

    return ret_arr


def inflate(arr_all):
    """
    データの水増し
    """

    data = np.array(arr_all.iloc[:, 0]).astype(np.float)  # WORLDはfloat前提のコードになっているのでfloat型にしておく

    _f0, t = pw.dio(data, FS)                   # 基本周波数の抽出
    f0 = pw.stonemask(data, _f0, t, FS)         # 基本周波数の修正
    sp = pw.cheaptrick(data, f0, t, FS)         # スペクトル包絡の抽出
    ap = pw.d4c(data, f0, t, FS, threshold=0.3) # 非周期性指標の抽出

    new_wav = pw.synthesize(f0 / 2, sp, ap, FS)
    new_wav = np.asarray(new_wav, dtype='int16')

    # 水増し
    pair_arr = pd.concat([pd.Series(new_wav),  arr_all.iloc[:, 1]], axis=1, join='inner')
    ret_arr = pd.concat([arr_all, pair_arr], ignore_index=True)

    return ret_arr


def make_pickles(arr_all):
    """
    各種pickle化
    ------------
    :param arr_all:
    """

    wav_data = np.array(arr_all.iloc[:, 0]).astype(np.float)
    phoneme_data = arr_all.iloc[:, 1]

    _f0, t = pw.dio(wav_data, FS, frame_period=3)   # 基本周波数の抽出
    f0 = pw.stonemask(wav_data, _f0, t, FS)         # 基本周波数の修正
    sp = pw.cheaptrick(wav_data, f0, t, FS)         # スペクトル包絡の抽出
    ap = pw.d4c(wav_data, f0, t, FS, threshold=0.3) # 非周期性指標の抽出

    # データの標準化
    stdsc = StandardScaler()
    f0 = stdsc.fit_transform(f0.reshape(-1, 1)).reshape(1, -1)[0]
    sp = stdsc.fit_transform(sp)
    ap = stdsc.fit_transform(ap)

    # 音素データを音声データ数に合わせて抽出
    phoneme_data = pd.concat([phoneme_data, pd.Series('sil')], ignore_index=True)
    phoneme_data = np.array([phoneme_data.iloc[int((time * FS))] for time in t])

    # 基本周波数
    with open(DIR_PICKLES + 'f0.pkl', mode='wb') as f:
        pickle.dump(f0, f)

    # スペクトル包絡
    with open(DIR_PICKLES + 'sp.pkl', mode='wb') as f:
        pickle.dump(sp, f, protocol=4)

    # 非周期成分
    with open(DIR_PICKLES + 'ap.pkl', mode='wb') as f:
        pickle.dump(ap, f, protocol=4)

    # 音素
    with open(DIR_PICKLES + 'phoneme.pkl', mode='wb') as f:
        pickle.dump(phoneme_data, f)


if __name__ == '__main__':
    main()
