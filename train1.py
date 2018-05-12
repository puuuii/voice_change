import pickle
import numpy as np
import pandas as pd
import pyworld as pw
import math


DIR_PICKLES = './pickles/'
FS = 16000


def main():
    # データのロード
    f0, sp, phoneme = load_data()
    exit()

    # 各学習に必要な変数作成
    x, y = make_variables(f0, sp, phoneme)

    # モデル作成

    # 学習実行

    # 予測実行


def load_data():
    """
    pklからデータロード
    """

    # 基本周波数
    with open(DIR_PICKLES + 'f0.pkl', 'rb') as f:
        f0 = pickle.load(f)

    # スペクトル包絡
    with open(DIR_PICKLES + 'sp.pkl', 'rb') as f:
        sp = pickle.load(f)

    # 音素
    with open(DIR_PICKLES + 'phoneme.pkl', 'rb') as f:
        phoneme = pickle.load(f)

    # 時間的位置
    with open(DIR_PICKLES + 't.pkl', 'rb') as f:
        t = pickle.load(f)

    # 音素データを音声データ数に合わせて抽出
    phoneme = pd.concat([phoneme, pd.Series('sil')], ignore_index=True)
    phoneme = np.array([phoneme.iloc[int((time*FS))] for time in t])

    return f0, sp, phoneme


def make_variables(f0, sp, phoneme):
    """
    各種変数作成
    -----------------
    :param f0:      基本周波数
    :param sp:      スペクトル包絡
    :param phoneme: 音素
    :return:        説明変数、目的変数
    """





def make_model():
    """
    モデル作成
    ----------
    :return: 学習済みモデル
    """

    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, LSTM
    from keras.layers.wrappers import Bidirectional
    from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

    FRAME_SIZE = 2000
    BATCH_SIZE = 32
    EPOCHS = 60
    STATEFUL = False

    MFCC_SIZE = 40
    HIDDEN_SIZE = 128
    PHONEME_SIZE = 36

    if __name__ == '__main__':
        x_all = np.load('')
        y_all = np.load('')
        testsize = np.shape(x_all)[0] // 10
        testsize = testsize // BATCH_SIZE * BATCH_SIZE
        trainsize = (np.shape(x_all)[0] - testsize) // BATCH_SIZE * BATCH_SIZE
        x_train = x_all[-trainsize:]
        y_train = y_all[-trainsize:]
        x_test = x_all[:testsize]
        y_test = y_all[:testsize]

        model = Sequential()
        model.add(Bidirectional(LSTM(HIDDEN_SIZE, \
                                     return_sequences=True), input_shape=(None, MFCC_SIZE)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Dense(PHONEME_SIZE))
        model.add(Dropout(0.3))
        model.add(Activation('softmax'))

        es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        mc = ModelCheckpoint('', \
                             monitor='val_loss', save_best_only=False, \
                             save_weights_only=False, mode='auto')

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, \
                  validation_data=[x_test, y_test], callbacks=[es, mc], shuffle=True)


if __name__ == '__main__':
    main()