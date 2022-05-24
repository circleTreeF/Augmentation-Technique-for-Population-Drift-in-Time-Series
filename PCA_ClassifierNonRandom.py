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

num_processor = 32

train_year = 2006
test_year = 2009
exper_name = 'Native-kde-bandwidth-0.1-to-1.5-pca-v4-weight-v1-and-v2-non-random-GBDT-non-random-data-run1' + str(
    train_year) + '-' + str(test_year)+ '-seed='
engine = create_engine('postgresql://postgres:postgres@10.182.20.32:28888', echo=True, future=True)

with open('config/RiskModel.json', 'r') as f:
    risk_dict = json.load(f)

with open('config/random.json', 'r') as f:
    random_dict = json.load(f)

with open('config/pca.json', 'r') as f:
    pca_dict = json.load(f)

exper_name+=str(random_dict['seed'])
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




def pca_dim_reduce(input_data, num_components):
    pca = PCA(n_components=int(num_components), **pca_dict)
    pca.fit(input_data)
    return pca, pca.transform(input_data)


def preprocessing_non_pca(loans_data_array, select_ratio):
    rng = np.random.default_rng(**random_dict)
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


def preprocessing_train(loans_data_array):
    rng = np.random.default_rng(**random_dict)
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
    rng = np.random.default_rng(**random_dict)
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
    ks_values_base = []
    ks_values_aug = []
    bandwidth_dict = []
    start_time = datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).timestamp()
    out_path = 'output/augmentation-exper/pca-exper/non-random/' + datetime.datetime.now(
        tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d-%H-%M-%S") + '-dim=' + str(
        args.Dimensions) + '/'
    os.mkdir(out_path)

    # pca_machine= PCA(n_components=)
    with Session(engine) as session:
        train_loans = get_loan_by_year(session, train_year)
        training_year_loans_array = db_class_to_array(train_loans)
        test_loans = get_loan_by_year(session, test_year)
        testing_year_loans_array = db_class_to_array(test_loans)


        for bandwidth in np.concatenate(
                (np.arange(0.1, 0.2, 0.02), np.arange(0.2, 1.0, 0.02), np.arange(1.0, 2.55, 0.05))):
            gc.collect()
            X_train, y_train, train_pca_transformer = preprocessing_train(training_year_loans_array)
            X_test, y_test = preprocessing_test(testing_year_loans_array, train_pca_transformer)
            X_train_non_pca, y_train_non_pca = preprocessing_non_pca(training_year_loans_array,
                                                                     risk_dict['non_default_loans_select_ratio_train'])
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
            gbm = GBM.GBM()
            gbm_aug.train((X_train, y_train), weight)
            gbm.train((X_train_non_pca, y_train_non_pca))
            # prediction result evaluation
            y_test_prob_aug = gbm_aug.classifier_machine.predict_proba(X_test,
                                                                       num_iteration=gbm_aug.classifier_machine.best_iteration_)
            current_aug_auc = metrics.roc_auc_score(y_test, y_test_prob_aug[:, 1])
            aug_auc.append(current_aug_auc)
            current_ks_aug = ks_2samp(y_test_prob_aug[:, 0], y_test_prob_aug[:, 1])
            ks_values_aug.append(current_ks_aug)
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
            # if default_number > 0:
            #     default_score = clf.score(x_default_test, y_default_test)
            # else:
            #     default_score = None
            # default_years_score.append(default_score)
            selected_test_default_numbers.append(default_number)
            bandwidth_dict.append(bandwidth)

        finalize_session(session)
    if args.Normalization == 'norm-1':
        stored_file_name = out_path + exper_name + '-dim=' + str(
            args.Dimensions) + '-norm-1' + '.exper.auc.json'
    else:
        if args.Normalization == 'norm-small':
            stored_file_name = out_path + exper_name + '-dim=' + str(
                args.Dimensions) + '-norm-small' + '.exper.auc.json'
    with open(stored_file_name, 'w+') as f:
        # result = {'all year scores': all_years_score, 'default all years score': default_years_score,
        #           'number of defaults': default_numbers, 'all year AUC': all_years_auc}
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
