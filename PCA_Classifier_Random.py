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
from scipy.stats import ks_2samp

exper_name = 'Native-kde-bandwidth-0.1-to-2.5-v2-pca-2006-v4-weight-1-non-random-GBDT-retro-train-random-test-all-random-data-run0'
engine = create_engine('postgresql://postgres:postgres@10.182.20.32:28888', echo=True, future=True)
num_processor = 32

train_year = 2006
test_year = 2009
TIME_STEP = 22 * 60
INIT_DATETIME = datetime.datetime(year=2022, month=3, day=15, hour=22, minute=30, second=11,
                                  tzinfo=pytz.timezone('Asia/Shanghai')).timestamp()
PREPROCESSING_BIAS = 5 * 60 + 8
count = 0
exper_name += '-test=' + str(test_year) + '-'

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
          'ks': [],
          'ks aug': [],
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


#
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


def db_class_to_array(loans_data):
    gc.collect()
    with closing(mp.Pool(num_processor)) as p:
        loans_data_list = p.map(pre.individual_x_y, loans_data)
        # clean up
        p.close()
        p.join()
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S") + " Queried Data Formatted")
    loans_data_array = np.array(loans_data_list)
    return loans_data_array


def preprocessing_non_pca(loans_data_array, select_ratio):
    # rng = np.random.default_rng(
    #     int(precessing_count * PREPROCESSING_BIAS+ count * TIME_STEP + INIT_DATETIME))
    rng = np.random.default_rng(
        int(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).timestamp() - start_time) + random_dict['seed'])
    # rng.shuffle(loans_data_array)
    non_default_loans = loans_data_array[loans_data_array[:, 19] == False]
    default_loans = loans_data_array[loans_data_array[:, 19] == True]
    random_non_default_index = rng.choice(len(non_default_loans),
                                          int(len(non_default_loans) * select_ratio),
                                          replace=False)
    random_non_default = non_default_loans[random_non_default_index]
    balanced_loans = np.concatenate((default_loans, random_non_default))
    rng.shuffle(balanced_loans)
    X = balanced_loans[:, :19]
    Y = balanced_loans[:, 19].astype('int')
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


def preprocessing_train(loans_data_array):
    # rng = np.random.default_rng(
    #     int(precessing_count * PREPROCESSING_BIAS+ count * TIME_STEP + INIT_DATETIME))
    rng = np.random.default_rng(
        TIME_STEP * count + PREPROCESSING_BIAS * precessing_count + random_dict['seed'])
    non_default_loans = loans_data_array[loans_data_array[:, 19] == False]
    default_loans = loans_data_array[loans_data_array[:, 19] == True]
    random_non_default_index = rng.choice(len(non_default_loans),
                                          int(len(non_default_loans) * risk_dict[
                                              'non_default_loans_select_ratio_train']),
                                          replace=False)
    random_non_default = non_default_loans[random_non_default_index]
    balanced_loans = np.concatenate((default_loans, random_non_default))
    rng.shuffle(balanced_loans)
    X = balanced_loans[:, :19]
    Y = balanced_loans[:, 19].astype('int')
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
    pca_transformer, reduced_x = pca_dim_reduce(encoded_x, num_components=args.Dimensions)
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S") + " Dimensions Reduced by PCA")
    return reduced_x, Y, pca_transformer


def preprocessing_test(loans_data_array, pca_transformer):

    rng = np.random.default_rng(TIME_STEP * count + PREPROCESSING_BIAS * precessing_count + int(random_dict['seed']))
    # rng = np.random.default_rng()
    non_default_loans = loans_data_array[loans_data_array[:, 19] == False]
    default_loans = loans_data_array[loans_data_array[:, 19] == True]
    random_non_default_index = rng.choice(len(non_default_loans),
                                          int(len(non_default_loans) * risk_dict[
                                              'non_default_loans_select_ratio_test']),
                                          replace=False)
    random_non_default = non_default_loans[random_non_default_index]
    balanced_loans = np.concatenate((default_loans, random_non_default))

    # balanced_loans = loans_data_array

    rng.shuffle(balanced_loans)
    X = balanced_loans[:, :19]
    Y = balanced_loans[:, 19].astype('int')
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
    reduced_x = pca_transformer.transform(encoded_x)
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S") + " Dimensions Reduced by PCA")
    return reduced_x, Y


def evaluate(model, test_x, test_y):
    auc = metrics.roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    # TODO: conform that all type of y is considered, the following commented code would warn only one type of y is provided
    # fpr, tpr, thresholds = metrics.roc_curve(test_y, model.predict(test_x), pos_label=2)
    # metrics.auc(fpr, tpr)
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
    ks_values_base = []
    ks_values_aug = []
    start_time = datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).timestamp()
    out_path = 'output/augmentation-exper/pca-exper/' + str(test_year) + '/' + datetime.datetime.now(
        tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d-%H-%M-%S") + '-dim=' + str(
        args.Dimensions) + '/'
    os.mkdir(out_path)

    with Session(engine) as session:
        train_loans = get_loan_by_year(session, train_year)
        training_year_loans_array = db_class_to_array(train_loans)

        test_loans = get_loan_by_year(session, test_year)
        testing_year_loans_array = db_class_to_array(test_loans)

        for bandwidth in np.concatenate(
                (np.arange(0.1, 0.2, 0.02), np.arange(0.2, 1.0, 0.02), np.arange(1.0, 2.55, 0.05))):
            gc.collect()
            precessing_count = 0
            X_train, y_train, train_pca_transformer = preprocessing_train(training_year_loans_array)
            precessing_count += 1
            X_test, y_test = preprocessing_test(testing_year_loans_array, train_pca_transformer)
            precessing_count += 1
            X_train_non_pca, y_train_non_pca = preprocessing_non_pca(training_year_loans_array,
                                                                     risk_dict['non_default_loans_select_ratio_train'])
            precessing_count += 1
            X_test_non_pca, y_test_non_pca = preprocessing_non_pca(testing_year_loans_array,
                                                                   risk_dict['non_default_loans_select_ratio_test'])

            """
            augmentation
            """
            density, density_estimate = aug.distribution_modifier(X_test, __bandwidth=bandwidth)
            pdf_train_v2 = density_estimate.evaluate(aug.distribution_normalization(X_train))
            utility.save_density_with_bandwidth(out_path, pdf_train_v2, "{:.4f}".format(bandwidth))

            if args.Normalization == 'norm-1':
                weight = pdf_train_v2.shape[0] * pdf_train_v2 / np.sum(pdf_train_v2)
            else:
                if args.Normalization == 'norm-small':
                    weight = pdf_train_v2 / np.sum(pdf_train_v2)
            print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
                "%Y-%m-%d %H:%M:%S") + " Modeling start")
            gbm_aug = GBM.GBM()
            gbm_aug.train((X_train, y_train), weight)
            # prediction result evaluation
            y_test_prob_aug = gbm_aug.classifier_machine.predict_proba(X_test,
                                                                       num_iteration=gbm_aug.classifier_machine.best_iteration_)
            current_aug_auc = metrics.roc_auc_score(y_test, y_test_prob_aug[:, 1])
            aug_auc.append(current_aug_auc)
            current_ks_aug = ks_2samp(y_test_prob_aug[:, 0], y_test_prob_aug[:, 1])
            ks_values_aug.append(current_ks_aug)
            gbm = GBM.GBM()
            gbm.train((X_train_non_pca, y_train_non_pca))
            y_test_prob_original = gbm.classifier_machine.predict_proba(X_test_non_pca,
                                                                        num_iteration=gbm.classifier_machine.best_iteration_)
            current_auc_original = metrics.roc_auc_score(y_test_non_pca, y_test_prob_original[:, 1])
            all_years_auc.append(current_auc_original)
            current_ks = ks_2samp(y_test_prob_original[:, 0], y_test_prob_original[:, 1])
            ks_values_base.append(current_ks)
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
            count += 1

    finalize_session(session)
    if args.Normalization == 'norm-1':
        stored_file_name = out_path + exper_name + '-dim=' + str(
            args.Dimensions) + '-norm-1' + '.exper.auc.json'
    else:
        if args.Normalization == 'norm-small':
            stored_file_name = out_path + exper_name + '-dim=' + str(
                args.Dimensions) + '-norm-small' + '.exper.auc.json'
    with open(stored_file_name, 'w+') as f:
        result['selected number of test defaults'] = selected_test_default_numbers
        result['selected number of test loans'] = selected_test_loan_numbers
        result['selected number of train defaults'] = selected_train_default_numbers
        result['selected number of train loans'] = selected_train_loan_numbers
        result['all year AUC'] = all_years_auc
        result['augmentation AUC'] = aug_auc
        result['ks'] = ks_values_base
        result['ks aug'] = ks_values_aug
        result['bandwidth'] = bandwidth_dict
        json.dump(result, f)
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d:%H:%M:%S") + " Result Stored as " + stored_file_name)
