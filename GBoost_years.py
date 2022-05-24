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
from itertools import repeat
from sklearn.metrics import accuracy_score
import utility
from contextlib import closing
import gc
from sklearn.decomposition import PCA

exper_name = 'Native-kde-bandwidth=0.6-pca-d=50-years=2006-test=2006-2019-non-subsampling-test-time-v4-non-random-all-run1'
engine = create_engine('postgresql://postgres:postgres@10.182.20.32:28888', echo=True)
num_processor = 32

train_year = 2006
test_year = 2009
bandwidth = 0.6


with open('config/RiskModel.json', 'r') as f:
    risk_dict = json.load(f)

with open('config/random.json', 'r') as f:
    random_dict = json.load(f)

with open('config/pca.json', 'r') as f:
    pca_dict = json.load(f)

result = {'selected number of test defaults': [],
          'selected number of test loans': [],
          'selected number of train defaults': [],
          'selected number of train loans': [],

          'all year AUC': [],
          'augmentation AUC': [],
          'bandwidth': [],
          'test year':[]
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

# deprecated function
# def loans_split(input_loans_x, input_loans_y):
#     data_size = len(input_loans_x)
#     is_defaults = np.vectorize(int)(input_loans_y)
#     train_size = int(data_size * train_ratio)
#     x_train = input_loans_x[:train_size]
#     y_train = is_defaults[:train_size]
#     x_test = input_loans_x[train_size:]
#     y_test = is_defaults[train_size:]
#     return (x_train, y_train), (x_test, y_test)


def pca_dim_reduce(input_data, num_components):
    pca = PCA(n_components=int(num_components), **pca_dict)
    pca.fit(input_data)
    return pca, pca.transform(input_data)


def preprocessing_train(loans_data):
    rng = np.random.default_rng(**random_dict)
    rng.shuffle(loans_data)
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
    pca_transformer, reduced_x = pca_dim_reduce(encoded_x, num_components=args.Dimensions)
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S") + " Dimensions Reduced by PCA")
    return reduced_x, Y, pca_transformer


def preprocessing_test(loans_data, pca_transformer):
    rng = np.random.default_rng(**random_dict)
    rng.shuffle(loans_data)
    gc.collect()
    with closing(mp.Pool(num_processor)) as p:
        loans_data_list = p.map(pre.individual_x_y, loans_data)
        # clean up
        p.close()
        p.join()
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S") + " Queried Data Formatted")
    loans_data_array = np.array(loans_data_list)
    # Balance the non-default data and default by subsampling the non-default loans in a given ratio
    # non_default_loans = loans_data_array[loans_data_array[:, 19] == False]
    # default_loans = loans_data_array[loans_data_array[:, 19] == True]
    # random_non_default_index = rng.choice(len(non_default_loans),
    #                                       int(len(non_default_loans) * risk_dict['non_default_loans_select_ratio']))
    # random_non_default = non_default_loans[random_non_default_index]
    # balanced_loans = np.concatenate((default_loans, random_non_default))
    balanced_loans = loans_data_array
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
    reduced_x = pca_transformer.transform(encoded_x)
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S") + " Dimensions Reduced by PCA")
    return reduced_x, Y


def evaluate(model, test_x, test_y):
    auc = metrics.roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])

    return auc


def finalize_session(_session):
    _session.expire_all()
    _session.close()


if __name__ == '__main__':
    # Read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--Normalization", help="Weight Normalization Approaches")
    parser.add_argument("-d", "--Dimensions", help="Number of Dimensions after PCA")
    args = parser.parse_args()
    all_years_score = []
    all_years_auc = []
    default_years_score = []
    selected_test_default_numbers = []
    selected_test_loan_numbers = []
    selected_train_loan_numbers = []
    selected_train_default_numbers = []
    aug_auc = []
    bandwidth_dict = []
    test_year_dict = []
    out_path = 'output/augmentation-exper/all-years-expers/' + datetime.datetime.now(
        tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d-%H-%M-%S") + '-dim=' + str(
        args.Dimensions) + '/'
    os.mkdir(out_path)

    with Session(engine) as session:
        train_loans = get_loan_by_year(session, train_year)
        for test_year in np.arange(2009, 2021, 1):
            gc.collect()
            X_train, y_train, train_pca_transformer = preprocessing_train(train_loans)
            test_loans = get_loan_by_year(session, int(test_year))
            X_test, y_test = preprocessing_test(test_loans, train_pca_transformer)
            """
            augmentation
            """
            density, density_estimate = aug.distribution_modifier(X_test, __bandwidth=bandwidth)
            pdf_train_v2 = density_estimate.evaluate(aug.distribution_normalization(X_train))
            utility.save_density_with_bandwidth(out_path, pdf_train_v2, "{:.4f}".format(bandwidth))

            if args.Normalization == 'normalization-1':
                weight = pdf_train_v2.shape[0] * pdf_train_v2 / np.sum(pdf_train_v2)
            else:
                if args.Normalization == 'normalization-small':
                    weight = pdf_train_v2 / np.sum(pdf_train_v2)
            print("Modeling start")
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
                "%Y-%m-%d %H:%M:%S") + "Model Evaluated for bandwidth " + "{:.4f}".format(bandwidth))



            selected_test_loan_numbers.append(X_test.shape[0])
            selected_train_loan_numbers.append(X_train.shape[0])

            x_default_test = X_test[y_test == 1]
            y_default_test = y_test[y_test == 1]
            default_number = len(y_default_test)
            y_default_train = y_train[y_train == 1]
            selected_train_default_numbers.append(len(y_default_train))

            selected_test_default_numbers.append(default_number)
            bandwidth_dict.append(bandwidth)
            test_year_dict.append(test_year)

    finalize_session(session)
    if args.Normalization == 'normalization-1':
        stored_file_name = out_path + exper_name + '-dim=' + str(
            args.Dimensions) + '-normalization-1' + '.exper.auc.json'
    else:
        if args.Normalization == 'normalization-small':
            stored_file_name = out_path + exper_name + '-dim=' + str(
                args.Dimensions) + '-normalization-small' + '.exper.auc.json'
    with open(stored_file_name, 'w+') as f:
        result['selected number of test defaults'] = selected_test_default_numbers
        result['selected number of test loans'] = selected_test_loan_numbers
        result['selected number of train defaults'] = selected_train_default_numbers
        result['selected number of train loans'] = selected_train_loan_numbers
        result['all year AUC'] = all_years_auc
        result['augmentation AUC'] = aug_auc
        result['bandwidth'] = bandwidth_dict
        result['test year'] = test_year_dict
        json.dump(result, f)
        print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
            "%Y-%m-%d:%H:%M:%S") + " Result Stored as " + stored_file_name)
