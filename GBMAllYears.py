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

exper_name = 'Native-kde-bandwidth=0.6-pca-train-2006-test-all-years-train-ratio=1-v4-weight-norm-1-non-random-GBDT-random-data-fix-repeat-year-run0'
engine = create_engine('postgresql://postgres:postgres@db', echo=True, future=True)
num_processor = 32

train_year = 2006
test_year = 2009
bandwidth = 0.5
train_ratio = 0.8

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


def preprocessing_init_year(loans_data_array):
    rng = np.random.default_rng()
    non_default_loans = loans_data_array[loans_data_array[:, 19] == False]
    default_loans = loans_data_array[loans_data_array[:, 19] == True]
    random_non_default_index_train = rng.choice(len(non_default_loans),
                                                int(len(non_default_loans) * risk_dict[
                                              'non_default_loans_select_ratio_train']), replace=False)
    random_non_default_train = non_default_loans[random_non_default_index_train]

    random_default_index_train = rng.choice(len(default_loans), int(len(default_loans) * train_ratio), replace=False)
    random_default = default_loans[random_default_index_train]
    balanced_loans_train = np.concatenate((random_default, random_non_default_train))
    rng.shuffle(balanced_loans_train)

    X_train = balanced_loans_train[:, :19]
    Y_train = balanced_loans_train[:, 19].astype('int')
    encoded_x_train = pre.encode(X_train)
    non_missed_credit_score_data = encoded_x_train[encoded_x_train[:, 0] != 999]
    missed_x_credit_factors = encoded_x_train[encoded_x_train[:, 0] == 999][:, 1:]
    if len(missed_x_credit_factors) > 0:
        credit_score_classifier = Csp.CreditScoreRegressionClassifier()
        credit_score_classifier.train(non_missed_credit_score_data)
        encoded_x_train[encoded_x_train[:, 0] == 999] = credit_score_classifier.predict(missed_x_credit_factors)
    encoded_x_train = encoded_x_train.astype(float)

    # get testing data set when training year
    default_test_index = np.delete(np.arange(len(default_loans)), random_default_index_train)
    default_test = default_loans[default_test_index]
    non_default_test_index = np.delete(np.arange(len(non_default_loans)), random_non_default_index_train)
    non_default_test = non_default_loans[non_default_test_index]
    balanced_loans_test = np.concatenate((default_test, non_default_test))
    rng.shuffle(balanced_loans_test)

    X_test = balanced_loans_test[:, :19]
    Y_test = balanced_loans_test[:, 19].astype('int')
    encoded_x_test = pre.encode(X_test)
    non_missed_credit_score_data = encoded_x_test[encoded_x_test[:, 0] != 999]
    missed_x_credit_factors = encoded_x_test[encoded_x_test[:, 0] == 999][:, 1:]
    if len(missed_x_credit_factors) > 0:
        credit_score_classifier = Csp.CreditScoreRegressionClassifier()
        credit_score_classifier.train(non_missed_credit_score_data)
        encoded_x_test[encoded_x_test[:, 0] == 999] = credit_score_classifier.predict(missed_x_credit_factors)
    encoded_x_test = encoded_x_test.astype(float)
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S") + " Data Pre-processed")
    return encoded_x_train, Y_train, encoded_x_test, Y_test


def preprocessing_non_pca(loans_data_array, select_ratio):
    rng = np.random.default_rng()
    non_default_loans = loans_data_array[loans_data_array[:, 19] == False]
    default_loans = loans_data_array[loans_data_array[:, 19] == True]
    random_non_default_index = rng.choice(len(non_default_loans),
                                          int(len(non_default_loans) * select_ratio), replace=False)
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
        "%Y-%m-%d %H:%M:%S") + " Data Pre-processed")
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
    rng = np.random.default_rng(int(random_dict['seed']))
    non_default_loans = loans_data_array[loans_data_array[:, 19] == False]
    default_loans = loans_data_array[loans_data_array[:, 19] == True]
    random_non_default_index = rng.choice(len(non_default_loans),
                                          int(len(non_default_loans) * risk_dict[
                                              'non_default_loans_select_ratio_train']), replace=False)
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
        "%Y-%m-%d %H:%M:%S") + " Data Pre-processed")
    pca_transformer, reduced_x = pca_dim_reduce(encoded_x, num_components=args.Dimensions)
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S") + " Dimensions Reduced by PCA")
    return reduced_x, Y, pca_transformer


def preprocessing_test(loans_data, pca_transformer):
    rng = np.random.default_rng(int(random_dict['seed']))
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
    random_non_default_index = rng.choice(len(non_default_loans), int(len(non_default_loans) * risk_dict[
        'non_default_loans_select_ratio_test']), replace=False)
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
        "%Y-%m-%d %H:%M:%S") + " Data Pre-processed")
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
    out_path = 'output/augmentation-exper/all-years-expers/' + datetime.datetime.now(
        tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d-%H-%M-%S") + '-dim=' + str(
        args.Dimensions) + '/'
    os.mkdir(out_path)

    # pca_machine= PCA(n_components=)
    with Session(engine) as session:
        train_loans = get_loan_by_year(session, train_year)
        train_loans_array = db_class_to_array(train_loans)
        X_train_non_pca_init_year, y_train_non_pca_init_year, X_test_non_pca_init_year, y_test_non_pca_init_year = preprocessing_init_year(
            train_loans_array)
        gbm = GBM.GBM()
        gbm.train((X_train_non_pca_init_year, y_train_non_pca_init_year))

        y_test_prob_original = gbm.classifier_machine.predict_proba(X_test_non_pca_init_year,
                                                                    num_iteration=gbm.classifier_machine.best_iteration_)
        current_auc_original = metrics.roc_auc_score(y_test_non_pca_init_year, y_test_prob_original[:, 1])
        all_years_auc.append(current_auc_original)
        aug_auc.append(current_auc_original)
        selected_test_loan_numbers.append(X_test_non_pca_init_year.shape[0])
        selected_train_loan_numbers.append(X_train_non_pca_init_year.shape[0])

        x_default_test = X_test_non_pca_init_year[y_test_non_pca_init_year == 1]
        y_default_test = y_test_non_pca_init_year[y_test_non_pca_init_year == 1]
        default_number = len(y_default_test)
        y_default_train = y_train_non_pca_init_year[y_train_non_pca_init_year == 1]
        selected_train_default_numbers.append(len(y_default_train))
        selected_test_default_numbers.append(default_number)
        bandwidth_dict.append(bandwidth)


        for test_year in np.arange(2007, 2021, 1):
            gc.collect()
            test_loans = get_loan_by_year(session, int(test_year))
            testing_year_loans_array = db_class_to_array(test_loans)
            X_train_non_pca, y_train_non_pca = preprocessing_non_pca(train_loans_array,0.01)
            X_test_non_pca, y_test_non_pca = preprocessing_non_pca(testing_year_loans_array,0.01)
            # train_set, test_set = loans_split(X, y)
            """
            augmentation
            """
            density, density_estimate = aug.distribution_modifier(X_test_non_pca, __bandwidth=bandwidth)
            pdf_train_v2 = density_estimate.evaluate(aug.distribution_normalization(X_train_non_pca))
            utility.save_density_with_bandwidth(out_path, pdf_train_v2, "{:.4f}".format(test_year))

            if args.Normalization == 'norm-1':
                weight = pdf_train_v2.shape[0] * pdf_train_v2 / np.sum(pdf_train_v2)
            else:
                if args.Normalization == 'norm-small':
                    weight = pdf_train_v2 / np.sum(pdf_train_v2)
            print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
                "%Y-%m-%d %H:%M:%S") + " Modeling start")

            gbm = GBM.GBM()
            gbm.train((X_train_non_pca, y_train_non_pca))
            y_test_prob_original = gbm.classifier_machine.predict_proba(X_test_non_pca,
                                                                        num_iteration=gbm.classifier_machine.best_iteration_)
            current_auc_original = metrics.roc_auc_score(y_test_non_pca, y_test_prob_original[:, 1])
            all_years_auc.append(current_auc_original)
            # prediction result evaluation
            gbm_aug = GBM.GBM()
            gbm_aug.train((X_train_non_pca, y_train_non_pca), weight)
            y_test_prob_aug = gbm_aug.classifier_machine.predict_proba(X_test_non_pca,
                                                                       num_iteration=gbm_aug.classifier_machine.best_iteration_)
            current_aug_auc = metrics.roc_auc_score(y_test_non_pca, y_test_prob_aug[:, 1])
            aug_auc.append(current_aug_auc)

            # predict_clf = clf.predict(X_test)
            # auc = evaluate(clf, X_test, y_test)
            print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
                "%Y-%m-%d %H:%M:%S") + "Model Evaluated for bandwidth " + "{:.4f}".format(bandwidth))

            # score = clf.score(X_test, y_test)
            # all_years_score.append(score)

            selected_test_loan_numbers.append(X_test_non_pca.shape[0])
            selected_train_loan_numbers.append(X_train_non_pca.shape[0])

            x_default_test = X_test_non_pca[y_test_non_pca == 1]
            y_default_test = y_test_non_pca[y_test_non_pca == 1]
            default_number = len(y_default_test)
            y_default_train = y_train_non_pca[y_train_non_pca == 1]
            selected_train_default_numbers.append(len(y_default_train))
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
        result['bandwidth'] = bandwidth_dict
        json.dump(result, f)
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d:%H:%M:%S") + " Result Stored as " + stored_file_name)
