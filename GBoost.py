import datetime
import os
import pickle

import numpy as np
import pytz
from sklearn.ensemble import GradientBoostingClassifier as GBClassifier
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.orm import Session
import joined_tables_by_default as jtd
import multiprocessing as mp
import preprocessing as pre
from sklearn import metrics
import CreditScorePredictor as Csp
import json
import augmentation as aug
import lightgbm as lgb
import GBM
from itertools import repeat
from sklearn.metrics import accuracy_score

exper_name = 'tree-kde-h=.4-lightgbm'
engine = create_engine('postgresql://postgres:postgres@db', echo=True, future=True)
num_processor = 32

train_ratio = 0.8
with open('RiskModel.json', 'r') as f:
    risk_dict = json.load(f)

with open('random.json', 'r') as f:
    random_dict = json.load(f)

result = {'selected number of test defaults': [],
          'selected number of test loans': [],
          'selected number of train defaults': [],
          'selected number of train loans': [],

          'all year AUC': [],
          'augmentation AUC': []

          }


def get_loan_by_year(year):
    with Session(engine) as session:
        joined_tables = jtd.OriginationDataWithDefault
        query_result = session.execute(
            select(joined_tables).where(
                jtd.OriginationDataWithDefault.year == year)).all()
        print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
            "%Y-%m-%d %H:%M:%S") + " Query Complete. Data Got from database")

        return query_result


def loans_split(input_loans_x, input_loans_y):
    data_size = len(input_loans_x)
    is_defaults = np.vectorize(int)(input_loans_y)
    train_size = int(data_size * train_ratio)
    x_train = input_loans_x[:train_size]
    y_train = is_defaults[:train_size]
    x_test = input_loans_x[train_size:]
    y_test = is_defaults[train_size:]
    return (x_train, y_train), (x_test, y_test)


def preprocessing(loans_data):
    rng = np.random.default_rng(random_dict['random_seed'])
    rng.shuffle(loans_data)
    with mp.Pool(num_processor) as p:
        loans_data_list = p.map(pre.individual_x_y, loans_data)
        p.close()
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S") + " Queried Data Formatted")
    loans_data_array = np.array(loans_data_list)
    non_default_loans = loans_data_array[loans_data_array[:, 26] == False]
    default_loans = loans_data_array[loans_data_array[:, 26] == True]
    random_non_default_index = rng.choice(len(non_default_loans),
                                          int(len(non_default_loans) * risk_dict['non_default_loans_select_ratio']))
    random_non_default = non_default_loans[random_non_default_index]
    balanced_loans = np.concatenate((default_loans, random_non_default))
    rng.shuffle(balanced_loans)
    X = balanced_loans[:, :26]
    Y = balanced_loans[:, 26].astype('int')
    encoded_x = pre.encode(X)
    # TODO: conform if the classifier could be year dependent
    non_missed_credit_score_data = encoded_x[encoded_x[:, 0] != 999]
    missed_x_credit_factors = encoded_x[encoded_x[:, 0] == 999][:, 1:]
    if len(missed_x_credit_factors) > 0:
        credit_score_classifier = Csp.CreditScoreRegressionClassifier()
        credit_score_classifier.train(non_missed_credit_score_data)
        encoded_x[encoded_x[:, 0] == 999] = credit_score_classifier.predict(missed_x_credit_factors)
    encoded_x = encoded_x.astype(float)
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S") + " Data Pre-precessed")
    return encoded_x, Y


def evaluate(model, test_x, test_y):
    auc = metrics.roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    # TODO: conform that all type of y is considered, the following commented code would warn only one type of y is provided
    # fpr, tpr, thresholds = metrics.roc_curve(test_y, model.predict(test_x), pos_label=2)
    # metrics.auc(fpr, tpr)
    return auc


def test(year, clf, ):
    loans = get_loan_by_year(year)
    X, y = preprocessing(loans)

    train_set, test_set = loans_split(X, y)
    predict_clf = clf.predict(test_set[0])
    auc = evaluate(clf, test_set[0], test_set[1])
    score = clf.score(test_set[0], test_set[1])
    x_default_test = test_set[0][test_set[1] == 1]
    y_default_test = test_set[1][test_set[1] == 1]
    default_number = len(y_default_test)
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S ") + "Model Evaluated for year " + str(year))

    if default_number > 0:
        default_score = clf.score(x_default_test, y_default_test)
    else:
        default_score = None
    return auc, default_score, default_number


if __name__ == '__main__':
    all_years_score = []
    all_years_auc = []
    default_years_score = []
    selected_test_default_numbers = []
    selected_test_loan_numbers = []
    selected_train_loan_numbers = []
    selected_train_default_numbers = []
    aug_auc = []
    loans = get_loan_by_year(2004)
    X, y = preprocessing(loans)
    train_set, test_set = loans_split(X, y)
    density, density_estimate = aug.distribution_modifier(test_set)
    # PDF calculation with kernel density method 1
    pdf_test = density.pdf()
    pdf_train = density.pdf(aug.distribution_normalization(train_set[0]))
    # PDF calculation with kernel density method 2
    pdf_test_v2 = density_estimate.evaluate(aug.distribution_normalization(test_set[0]))
    pdf_train_v2 = density_estimate.evaluate(aug.distribution_normalization(train_set[0]))
    # n_estimators=risk_dict['n_estimators'], random_state=risk_dict['random_state'],
    #                        max_depth=risk_dict['max_depth'], min_samples_leaf=risk_dict['min_samples_leaf']
    """
    Prepare the data and weight in txt file
    """
    # current_cache_prefix_time_date = datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
    #     "%Y-%m-%d %H:%M:%S")
    # current_cache_path = 'cache/data/' + current_cache_prefix_time_date + '/'
    # os.mkdir(current_cache_path)
    # GBM.ndarray_to_cache(current_cache_path, '-train-x-', year=2004, array=train_set[0])
    # GBM.ndarray_to_cache(current_cache_path, '-train-y-', year=2004, array=train_set[1])
    # GBM.ndarray_to_cache(current_cache_path, '-train--', year=2004, array=train_set[1])

    gbm_aug = GBM.train(train_set, pdf_train_v2)
    gbm = GBM.train(train_set)

    # prediction result evaluation
    y_test_prob_aug = gbm_aug.predict_proba(test_set[0], num_iteration=gbm_aug.best_iteration_)
    current_aug_auc = metrics.roc_auc_score(test_set[1], y_test_prob_aug[:, 1])
    aug_auc.append(current_aug_auc)

    y_test = gbm_aug.predict(test_set[0], num_iteration=gbm_aug.best_iteration_)
    accuracy = accuracy_score(test_set[1], y_test)

    # test the original gbm without augmentation
    y_test_prob_original = gbm.predict_proba(test_set[0], num_iteration=gbm_aug.best_iteration_)
    current_auc_original = metrics.roc_auc_score(test_set[1], y_test_prob_original[:, 1])
    all_years_auc.append(current_auc_original)

    selected_test_loan_numbers.append(test_set[0].shape[0])
    selected_train_loan_numbers.append(train_set[0].shape[0])
    # clf = GBClassifier(max_depth=risk_dict['max_depth'], n_estimators=risk_dict['n_estimators'])
    # clf.fit(train_set[0], train_set[1])
    # print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
    #     "%Y-%m-%d %H:%M:%S") + " Gradient Boosting Model Trained")
    # predict_clf = clf.predict(test_set[0])
    # auc = evaluate(clf, test_set[0], test_set[1])
    # all_years_auc.append(auc)
    # print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
    #     "%Y-%m-%d %H:%M:%S") + "Model Evaluated for 2006")
    # score = clf.score(test_set[0], test_set[1])
    # all_years_score.append(score)
    x_default_test = test_set[0][test_set[1] == 1]
    y_default_test = test_set[1][test_set[1] == 1]
    default_number = len(y_default_test)
    selected_test_default_numbers.append(default_number)
    y_default_train = train_set[1][train_set[1] == 1]
    selected_train_default_numbers.append(len(y_default_train))
    # if default_number > 0:
    #     default_score = clf.score(x_default_test, y_default_test)
    # else:
    #     default_score = None
    # default_years_score.append(default_score)
    train_ratio = 0.7
    # with mp.Pool(num_processor) as p:
    #     test_roc, test_default_score, test_default_number = zip(
    #         *p.starmap(test, zip(list(range(2007, 2021)), repeat(clf))))
    # all_years_roc.extend(test_roc)
    # default_years_score.extend(test_default_score)
    # default_numbers.extend(test_default_number)
    for year in range(2005, 2018):
        loans = get_loan_by_year(year)
        X, y = preprocessing(loans)

        train_set, test_set = loans_split(X, y)
        """
        augmentation
        """
        density, density_estimate = aug.distribution_modifier(test_set)
        pdf_train_v2 = density_estimate.evaluate(aug.distribution_normalization(train_set[0]))
        gbm_aug = GBM.train(train_set, pdf_train_v2)
        # prediction result evaluation
        y_test_prob_aug = gbm_aug.predict_proba(test_set[0], num_iteration=gbm_aug.best_iteration_)
        current_aug_auc = metrics.roc_auc_score(test_set[1], y_test_prob_aug[:, 1])
        aug_auc.append(current_aug_auc)

        # predict_clf = clf.predict(test_set[0])
        # auc = evaluate(clf, test_set[0], test_set[1])
        y_test_prob_original = gbm.predict_proba(test_set[0], num_iteration=gbm.best_iteration_)
        current_auc_original = metrics.roc_auc_score(test_set[1], y_test_prob_original[:, 1])
        all_years_auc.append(current_auc_original)
        print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
            "%Y-%m-%d %H:%M:%S") + "Model Evaluated for year" + str(year))

        # score = clf.score(test_set[0], test_set[1])
        # all_years_score.append(score)

        selected_test_loan_numbers.append(test_set[0].shape[0])
        selected_train_loan_numbers.append(train_set[0].shape[0])

        x_default_test = test_set[0][test_set[1] == 1]
        y_default_test = test_set[1][test_set[1] == 1]
        default_number = len(y_default_test)
        y_default_train = train_set[1][train_set[1] == 1]
        selected_train_default_numbers.append(len(y_default_train))
        # if default_number > 0:
        #     default_score = clf.score(x_default_test, y_default_test)
        # else:
        #     default_score = None
        # default_years_score.append(default_score)
        selected_test_default_numbers.append(default_number)

    stored_file_name = "/workspace/FYP/codespace/output/" + datetime.datetime.now(
        tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d:%H:%M:%S") + exper_name + '.all.years.scores.pkl'
    with open(stored_file_name, 'wb') as f:
        # result = {'all year scores': all_years_score, 'default all years score': default_years_score,
        #           'number of defaults': default_numbers, 'all year AUC': all_years_auc}
        result['selected number of test defaults'] = selected_test_default_numbers
        result['selected number of test loans'] = selected_test_loan_numbers
        result['selected number of train defaults'] = selected_train_default_numbers
        result['selected number of train loans'] = selected_train_loan_numbers
        result['all year AUC'] = all_years_auc
        result['augmentation AUC'] = aug_auc
        pickle.dump(result, f)
        print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
            "%Y-%m-%d:%H:%M:%S") + " Result Stored as " + stored_file_name)
