import pandas as pd
# from sklearn import preprocessing
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, f1_score

classifier = DecisionTreeClassifier()
training_file_name = 'data/UNSW/train_normalized.csv'
testing_file_name = 'data/UNSW/test_normalized.csv'

R = 0.09
np = 64
number_of_iterations = 200
U = 1
L = 0



def get_number_of_inputs():
    global number_of_inputs
    return number_of_inputs


def init():
    global data_train, data_test, target_train, target_test, number_of_inputs

    # kddcup = pd.read_csv('kddcup.csv')
    # le = preprocessing.LabelEncoder()
    # for column in kddcup.columns:
    #     if kddcup[column].dtype == type(object):
    #         kddcup[column] = le.fit_transform(kddcup[column])
    # print(kddcup.shape)
    # kddcup = kddcup.drop_duplicates()
    # kddcup.to_csv('kddcup_no_duplicates.csv', sep=',', encoding='utf-8', index=False)
    #
    # x = kddcup.values  # returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # kddcup = pd.DataFrame(x_scaled)
    #
    # kddcup.to_csv('kddcup_no_duplicates_normalized.csv', sep=',', encoding='utf-8', index=False)
    #
    # print(kddcup.shape)

    train_set = pd.read_csv(training_file_name)
    test_set = pd.read_csv(testing_file_name)

    # y = kddcup.iloc[:, 41]
    # x = kddcup.iloc[:, :41]
    # data_train, data_test, target_train, target_test = train_test_split(x, y)

    number_of_inputs = len(train_set.columns) - 1

    data_train = train_set.iloc[:, :number_of_inputs]
    target_train = train_set.iloc[:, number_of_inputs]

    data_test = test_set.iloc[:, :number_of_inputs]
    target_test = test_set.iloc[:, number_of_inputs]


def calc_fitness(px):
    select = []
    for i, xi in enumerate(px):
        if xi >= .5:
            select.append(i)
    if len(select) == 0:
        return 0, 1, 0

    td = data_train.iloc[:, select]
    dt = data_test.iloc[:, select]

    # classifier = GaussianNB()
    # classifier = RandomForestClassifier(n_estimators=10)

    pred = classifier.fit(td, target_train).predict(dt)
    r = recall_score(target_test, pred, average=None)

    return r[0], (1 - r[1]), len(select)


def acc__f_score(px):
    select = []
    for i, xi in enumerate(px):
        if xi >= .5:
            select.append(i)
    if len(select) == 0:
        return 0, 1, 0

    td = data_train.iloc[:, select]
    dt = data_test.iloc[:, select]

    pred = classifier.fit(td, target_train).predict(dt)
    acc = accuracy_score(target_test, pred)
    f_score = f1_score(target_test, pred, average='macro')
    return acc, f_score


def get_attr(px):
    select = []
    for i, xi in enumerate(px):
        if xi >= .5:
            select.append(i)
    return select
# x = kddcup.iloc[:, :41]
# return x.columns
