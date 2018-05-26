import pickle
import numpy as np
import pyworld as pw
from scipy.io import wavfile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


DIR_PICKLES = './pickles/'
FS = 16000


def main():
    # データのロード
    f0, sp, ap, phoneme = load_data()

    # 各学習に必要な変数作成
    x, y = make_variables(f0, sp, ap, phoneme)

    # モデル作成
    hidden_units = [128, 128, 128]
    lr = 0.01
    model = make_model(x, y, hidden_units, lr)

    # 学習実行（学習済みモデルはpickle保存）
    test_ratio = 0.2
    n_epoch = 50
    batch_size = 4096
    train_model(model, x, y, test_ratio, n_epoch, batch_size)

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

    # 非周期成分
    with open(DIR_PICKLES + 'ap.pkl', 'rb') as f:
        ap = pickle.load(f)

    # 音素
    with open(DIR_PICKLES + 'phoneme.pkl', 'rb') as f:
        phoneme = pickle.load(f)
        phoneme = np.array(phoneme)


    return f0, sp, ap, phoneme


def make_variables(f0, sp, ap, phoneme):
    """
    各種変数作成
    -----------------
    :param f0:      基本周波数
    :param sp:      スペクトル包絡
    :param ap:      非周期成分
    :param phoneme: 音素
    :return:        説明変数、目的変数
    """

    skip = 7
    x = np.concatenate([np.array([f0]).T, sp[:, ::skip], ap[:, ::skip]], axis=1)
    y = phoneme

    # one-hotエンコーディング実施
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = np.array(y).reshape(1, -1)
    y = y.transpose()
    ohe = OneHotEncoder()
    y = ohe.fit_transform(y).toarray().reshape((-1, 41))

    return x, y


def make_model(x, y, hidden_units, lr):
    """
    モデル作成
    ----------
    :param x:               説明変数(f0, spの2次元)
    :param y:               目的変数(phonemeの1次元)
    :param hidden_units:    隠れ層リスト
    :param lr:              学習率
    :return:                学習済みモデル
    """

    from keras.layers.core import Activation, Dropout
    from keras.models import Sequential
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv1D
    from keras.layers.pooling import MaxPooling1D

    model = Sequential()
    model.add(Conv1D(128, 16, padding='same', input_shape=(1, x.shape[1])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(y.shape[1], 8, padding='same', activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model


def train_model(model, x, y, test_ratio, n_epoch, batch_size):
    """
    学習実行
    --------
    :param model:       モデル
    :param x:           説明変数
    :param y:           目的変数
    :param test_ratio:  テストデータに使う比率
    :param n_epoch:     エポック数
    :param batch_size:  バッチサイズ
    """

    x = x.reshape((-1, 1, x.shape[1]))
    y = y.reshape((-1, 1, y.shape[1]))
    print(x.shape, y.shape)

    # 訓練データと評価データに分割
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_ratio, random_state=0)


    # 学習実行
    history = model.fit(X_train, Y_train, validation_split=0.2,
                        batch_size=batch_size, epochs=n_epoch, verbose=1)

    # 予測精度出力
    score = model.evaluate(X_test, Y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # モデルのpickle化
    model.save(DIR_PICKLES + 'model.h5')


def predict():
    """
    予測実行
    """

    path = DIR_PICKLES + 'model_481.h'
    model = load_model(path)

    fs, wav_data = wavfile.read('say.wav')
    wav_data = wav_data.astype(np.float)

    _f0, t = pw.dio(wav_data, FS)                                 # 基本周波数の抽出
    f0 = pw.stonemask(wav_data, _f0, t, FS)                       # 基本周波数の修正
    sp = pw.cheaptrick(wav_data, f0, t, FS, f0_floor=750)         # スペクトル包絡の抽出


if __name__ == '__main__':
    main()