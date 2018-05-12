import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping


DIR_PICKLES = './pickles/'
FS = 16000


def main():
    # データのロード
    f0, sp, phoneme = load_data()

    # 各学習に必要な変数作成
    x, y = make_variables(f0, sp, phoneme)

    # モデル作成
    hidden_units = [200, 200, 200]
    model = make_model(x, y, hidden_units)

    # 学習実行（学習済みモデルはpickle保存）
    test_ratio = 0.3
    n_epoch = 200
    batch_size = 200
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

    # 音素
    with open(DIR_PICKLES + 'phoneme.pkl', 'rb') as f:
        phoneme = pickle.load(f)

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

    x = np.concatenate([np.array([f0]).T, sp], axis=1)
    y = phoneme

    # onhe-hotエンコーディング実施
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = np.array(y).reshape(1, -1)
    y = y.transpose()
    ohe = OneHotEncoder()
    y = ohe.fit_transform(y).toarray()

    return x, y


def make_model(x, y, hidden_units):
    """
    モデル作成
    ----------
    :param x:               説明変数(f0, spの2次元)
    :param y:               目的変数(phonemeの1次元)
    :param hidden_units:    隠れ層リスト
    :return:                学習済みモデル
    """


    # 重み初期化用関数
    def weight_variable(shape):
        return np.sqrt(2.0 / shape[0]) * np.random.normal(size=shape)

    # モデル構築
    model = Sequential()
    model.add(Dense(hidden_units[0], input_dim=x.shape[1], kernel_initializer=weight_variable))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    for n_units in hidden_units[1:]:
        model.add(Dense(n_units))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
    model.add(Dense(y.shape[1], kernel_initializer=weight_variable))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])

    # モデル情報出力
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

    # 訓練データとテストデータに分割
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    # 学習実行
    hist = model.fit(X_train, Y_train, epochs=n_epoch,
                     batch_size=batch_size,
                     validation_split=test_ratio)

    # 予測精度出力
    loss_and_metrics = model.evaluate(X_test, Y_test)
    print(loss_and_metrics)

    # モデルのpickle化
    with open(DIR_PICKLES + 'model.pkl', mode='wb') as f:
        pickle.dump(model, f)

    # 学習の可視化
    plt.plot(range(n_epoch), hist.history['loss'], label='loss')
    plt.plot(range(n_epoch), hist.history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()