import pandas as pd
import pickle
import pyworld as pw
import numpy as np
from scipy.io import wavfile as io
from sklearn.preprocessing import StandardScaler
from numpy import round
from matplotlib import pylab


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

    # 各音声パラメータ取得
    f0, sp, ap, t = get_sound_params(wav_data, 0.1, 750, 128, 0.3)
    del wav_data

    # 音素データ長を音声データ長に揃える
    phoneme_data = list(phoneme_data)
    phoneme_data = np.array([phoneme_data[int(np.round(time*FS))-1] for time in t])
    del t

    # データの標準化
    stdsc = StandardScaler()
    f0 = stdsc.fit_transform(f0.reshape((-1, 1))).astype(np.float32).reshape((-1))
    sp = np.array([stdsc.fit_transform(x.reshape((-1, 1))) for x in sp], dtype=np.float32).reshape((-1, 65))
    ap = np.array([stdsc.fit_transform(x.reshape((-1, 1))) for x in ap], dtype=np.float32).reshape((-1, 65))

    # データを800フレーム単位に分割
    # n_split_frame = 800
    # f0, sp, ap, phoneme_data = split_n_frame(n_split_frame, f0, sp, ap, phoneme_data)

    print(f0.shape, sp.shape, ap.shape, phoneme_data.shape)

    # 音素
    with open(DIR_PICKLES + 'phoneme.pkl', mode='wb') as f:
        pickle.dump(phoneme_data, f)
    del phoneme_data

    # 基本周波数
    with open(DIR_PICKLES + 'f0.pkl', mode='wb') as f:
        pickle.dump(f0, f)
    del f0

    # スペクトル包絡
    with open(DIR_PICKLES + 'sp.pkl', mode='wb') as f:
        pickle.dump(sp, f, protocol=4)
    del sp

    # 非周期成分
    with open(DIR_PICKLES + 'ap.pkl', mode='wb') as f:
        pickle.dump(ap, f, protocol=4)
    del ap


def get_sound_params(wav_data, frame_period, f0_floor, fft_size, threshold):
    """
    各音声パラメータ取得
    """

    f0, t = pw.dio(wav_data, FS, frame_period=frame_period)
    f0 = pw.stonemask(wav_data, f0, t, FS)
    sp = pw.cheaptrick(wav_data, f0, t, FS, f0_floor=f0_floor)
    ap = pw.d4c(wav_data, f0, t, FS, fft_size=fft_size, threshold=threshold)

    return f0, sp, ap, t


def split_n_frame(n, f0, sp, ap, phoneme_data):
    """
    データをnフレーム単位に分割
    --------------------------
    :return:
    """

    n_max_roop = int(len(f0) / n)
    f0 = np.array([f0[i * n:(i + 1) * n] for i in range(n_max_roop - 1)], dtype=np.float32)
    sp = np.array([sp[i * n:(i + 1) * n] for i in range(n_max_roop - 1)], dtype=np.float32).reshape((-1, n, 65))
    ap = np.array([ap[i * n:(i + 1) * n] for i in range(n_max_roop - 1)], dtype=np.float32).reshape((-1, n, 65))
    phoneme_data = np.array([phoneme_data[i * n:(i + 1) * n] for i in range(n_max_roop - 1)])

    return f0, sp, ap, phoneme_data


if __name__ == '__main__':
    main()
