import pandas as pd
from sklearn import preprocessing


def normalize_nsl(input_file, output_file):
    nsl = pd.read_csv(input_file)
    col_to_drop = len(nsl.columns) - 1
    col_to_fix = len(nsl.columns) - 2
    nsl = nsl.drop(nsl.columns[col_to_drop], axis='columns')

    #nsl = nsl.drop_duplicates()

    nsl[nsl.columns[col_to_fix]] = nsl[nsl.columns[col_to_fix]].str.lower()
    nsl[nsl.columns[col_to_fix]] = nsl[nsl.columns[col_to_fix]]. \
        replace(to_replace='^((?!normal).)*$', value='attack', regex=True)

    le = preprocessing.LabelEncoder()
    for column in nsl.columns:
        if nsl[column].dtype == type(object):
            nsl[column] = le.fit_transform(nsl[column])

    nsl.to_csv(output_file, sep=',', encoding='utf-8', index=False)


normalize_nsl('data/NSL-KDD/test.csv', 'data/NSL-KDD/test_normalized.csv')
normalize_nsl('data/NSL-KDD/train.csv', 'data/NSL-KDD/train_normalized.csv')
normalize_nsl('data/UNSW/test.csv', 'data/UNSW/test_normalized.csv')
normalize_nsl('data/UNSW/train.csv', 'data/UNSW/train_normalized.csv')

# print(nsl.head())
