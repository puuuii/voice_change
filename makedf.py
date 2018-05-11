import pandas as pd
import pickle
from scipy.io import wavfile as io


def main():
    path_datadir = './data/data/'
    cnt = 1
    list_all = []

    # wavとlabのペアを作り結合していく
    while True:
        print('process file:', cnt)
        arr_lab = pd.DataFrame()
        arr_wav = pd.DataFrame()

        # wavデータ作成
        path_wav = path_datadir + ('%08d' % cnt) + '.wav'
        try:
            sr, data_wav = io.read(path_wav)
        except FileNotFoundError:
            print('finish')
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
        list_all.append(pair_arr)

        cnt += 1

    # 全データをdf化
    arr_all = pd.concat(list_all, ignore_index=True)

    # 'silE'と'silB'を'sil'に変換
    arr_all = arr_all.replace({'silB': 'sil', 'silE': 'sil'})

    # pickle化
    with open('df.pkl', mode='wb') as f:
        pickle.dump(arr_all, f)


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
    main()
