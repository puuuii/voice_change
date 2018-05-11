import pickle
import pyworld


def main():
    df = load_data()


def load_data():
    """
    pklからデータロード
    :return: データ
    """

    pkl_path = 'df.pkl'
    with open(pkl_path, 'rb') as pkl:
        return pickle.load(pkl)


if __name__ == '__main__':
    main()