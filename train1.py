import pickle


def main():
    pkl_path = 'df.pkl'
    with open(pkl_path, 'rb') as pkl:
        df = pickle.load(pkl)
        print(df[1000:1200])
        print(df[2000:2200])
        print(df[30000:30200])

if __name__ == '__main__':
    main()