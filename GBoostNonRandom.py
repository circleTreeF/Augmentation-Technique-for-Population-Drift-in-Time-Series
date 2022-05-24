import datetime
import os
import pickle
import argparse
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
import utility
from contextlib import closing
import gc

exper_name = 'naive-kde-bandwidth-0.3-to-0.511-step=0.001-nondefault-select-ratio=0.01-2006-2009-v4-non-random-GBDT-non-random-data-run-0'
engine = create_engine('postgresql://postgres:postgres@db', echo=True)
num_processor = 32

train_year = 2006
test_year = 2009

train_ratio = 0.8
with open('config/RiskModel.json', 'r') as f:
    risk_dict = json.load(f)

with open('config/random.json', 'r') as f:
    random_dict = json.load(f)

result = {'selected number of test defaults': [],
          'selected number of test loans': [],
          'selected number of train defaults': [],
          'selected number of train loans': [],

          'all year AUC': [],
          'augmentation AUC': [],
          'bandwidth': []
          }


def get_loan_by_year(session, year):
    joined_tables = jtd.OriginationDataWithDefault
    query_result = session.execute(
        select(joined_tables).where(
            jtd.OriginationDataWithDefault.year == year).order_by(joined_tables.loan_sequence_number)).all()
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S") + " Query Complete. Data Got from database")
    return query_result


"""
Input: X and Y for the whole data set
Output: (x train, y train), (x test, y test)
"""


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
    rng = np.random.default_rng(**random_dict)
    # rng.shuffle(loans_data)
    gc.collect()
    with closing(mp.Pool(num_processor)) as p:
        loans_data_list = p.map(pre.individual_x_y, loans_data)
        # clean up
        p.close()
        p.join()
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S") + " Queried Data Formatted")
    loans_data_array = np.array(loans_data_list)
    non_default_loans = loans_data_array[loans_data_array[:, 19] == False]
    default_loans = loans_data_array[loans_data_array[:, 19] == True]
    random_non_default_index = rng.choice(len(non_default_loans),
                                          int(len(non_default_loans) * risk_dict['non_default_loans_select_ratio']))
    random_non_default = non_default_loans[random_non_default_index]
    balanced_loans = np.concatenate((default_loans, random_non_default))
    rng.shuffle(balanced_loans)
    X = balanced_loans[:, :19]
    Y = balanced_loans[:, 19].astype('int')
    encoded_x = pre.encode(X)
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
    return auc


def test(year, clf, session):
    loans = get_loan_by_year(session, year)
    X, y = preprocessing(loans)

    train_set, test_set = loans_split(X, y)
    predict_clf = clf.predict(X_test)
    auc = evaluate(clf, X_test, y_test)
    score = clf.score(X_test, y_test)
    x_default_test = X_test[y_test == 1]
    y_default_test = y_test[y_test == 1]
    default_number = len(y_default_test)
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S ") + "Model Evaluated for year " + str(year))

    if default_number > 0:
        default_score = clf.score(x_default_test, y_default_test)
    else:
        default_score = None
    return auc, default_score, default_number


def finalize_session(_session):
    _session.expire_all()
    _session.close()


if __name__ == '__main__':
    all_years_score = []
    all_years_auc = []
    default_years_score = []
    selected_test_default_numbers = []
    selected_test_loan_numbers = []
    selected_train_loan_numbers = []
    selected_train_default_numbers = []
    aug_auc = []
    bandwidth_dict = []
    dens_out_path = 'output/augmentation-exper/bandwidth-exper/' + datetime.datetime.now(
        tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d-%H-%M-%S") + '/'
    os.mkdir(dens_out_path)
    # Read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--Normalization", help="Weight Normalization Approaches")
    args = parser.parse_args()

    # pca_machine= PCA(n_components=)
    with Session(engine) as session:
        train_loans = get_loan_by_year(session, train_year)
        test_loans = get_loan_by_year(session, test_year)
        for bandwidth in np.arange(0.3, 0.511, 0.001):

            gc.collect()
            X_train, y_train = preprocessing(train_loans)
            X_test, y_test = preprocessing(test_loans)
            """
            augmentation
            """
            density, density_estimate = aug.distribution_modifier(X_test, __bandwidth=bandwidth)
            pdf_train_v2 = density_estimate.evaluate(aug.distribution_normalization(X_train))
            utility.save_density_with_bandwidth(dens_out_path, pdf_train_v2, bandwidth)

            if args.Normalization == 'norm-1':
                weight = pdf_train_v2.shape[0] * pdf_train_v2 / np.sum(pdf_train_v2)
            else:
                if args.Normalization == 'norm-small':
                    weight = pdf_train_v2 / np.sum(pdf_train_v2)
            print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
                "%Y-%m-%d %H:%M:%S") + " Modeling start")
            gbm_aug = GBM.GBM()
            gbm = GBM.GBM()
            gbm_aug.train((X_train, y_train), weight)
            gbm.train((X_train, y_train))
            # prediction result evaluation
            y_test_prob_aug = gbm_aug.classifier_machine.predict_proba(X_test,
                                                                       num_iteration=gbm_aug.classifier_machine.best_iteration_)
            current_aug_auc = metrics.roc_auc_score(y_test, y_test_prob_aug[:, 1])
            aug_auc.append(current_aug_auc)


            y_test_prob_original = gbm.classifier_machine.predict_proba(X_test,
                                                                        num_iteration=gbm.classifier_machine.best_iteration_)
            current_auc_original = metrics.roc_auc_score(y_test, y_test_prob_original[:, 1])
            all_years_auc.append(current_auc_original)
            print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
                "%Y-%m-%d %H:%M:%S") + "Model Evaluated for bandwidth " + str(bandwidth))


            selected_test_loan_numbers.append(X_test.shape[0])
            selected_train_loan_numbers.append(X_train.shape[0])

            x_default_test = X_test[y_test == 1]
            y_default_test = y_test[y_test == 1]
            default_number = len(y_default_test)
            y_default_train = y_train[y_train == 1]
            selected_train_default_numbers.append(len(y_default_train))
            selected_test_default_numbers.append(default_number)
            bandwidth_dict.append(bandwidth)

    finalize_session(session)
    if args.Normalization == 'norm-1':
        stored_file_name = dens_out_path + exper_name + 'normalization-1' + '.exper.auc.json'
    else:
        if args.Normalization == 'norm-small':
            stored_file_name = dens_out_path + exper_name + 'normalization-small' + '.exper.auc.json'
    with open(stored_file_name, 'w+') as f:
        # result = {'all year scores': all_years_score, 'default all years score': default_years_score,
        #           'number of defaults': default_numbers, 'all year AUC': all_years_auc}
        result['selected number of test defaults'] = selected_test_default_numbers
        result['selected number of test loans'] = selected_test_loan_numbers
        result['selected number of train defaults'] = selected_train_default_numbers
        result['selected number of train loans'] = selected_train_loan_numbers
        result['all year AUC'] = all_years_auc
        result['augmentation AUC'] = aug_auc
        result['bandwidth'] = bandwidth_dict
        json.dump(result, f)
        print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
            "%Y-%m-%d:%H:%M:%S") + " Result Stored as " + stored_file_name)
