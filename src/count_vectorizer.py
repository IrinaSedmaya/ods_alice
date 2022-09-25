def count_vectorizer(df):
    df.to_csv('train_sites.txt', sep=' ', index=None, header=None)

    with open('train_sites.txt') as inp_train_file:
        train_sites = cv.fit_transform(inp_train_file)
    return train_sites

