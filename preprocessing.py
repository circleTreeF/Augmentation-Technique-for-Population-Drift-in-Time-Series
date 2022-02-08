import numpy as np

import joined_tables_by_default as jtd
import month_performance_table as mpt
import Origination_Table_Sample as Ots
import Origination_Table as Ot
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
from sqlalchemy import select, update
import sklearn.preprocessing as pre
import pytz
import datetime
from decimal import Decimal

raw_data_path = 'raw_data/'
num_processor = 32

DEFAULT_DEFAULT = False

CATEGORIES = [['Y', 'N', '9'], [Decimal('1'), Decimal('2'), Decimal('3'), Decimal('4'), Decimal('99')],
              ['P', 'I', 'S', '9'], ['R', 'B', 'C', 'T', '9'], ['Y', 'N'], ['FRM', 'ARM'],
              ['AL', 'AK', 'AS', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN',
               'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM',
               'NY', 'NC', 'ND', 'MP', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA',
               'VI', 'WA', 'WV', 'WI', 'WY'],
              ['CO', 'PU', 'MH', 'SF', 'CP', '99'],
              ['P', 'C', 'N', 'R', '9'], ['Y', None], ['H', Decimal('9')], ['Y', None],
              [Decimal('1'), Decimal('2'), Decimal('3'), Decimal('9')], ['Y', 'N']]

'''
Determine if the given sequence result in default in sample dataset
'''


def add_jointed_table_with_quarter_and_default(year, quarter, query_session, current_object):
    query_session.add(
        jtd.OriginationDataWithDefault(credit_score=current_object[0], first_payment_date=current_object[1],
                                       first_time_homebuyer_flag=current_object[2],
                                       maturity_date=current_object[3],
                                       msa=current_object[4], mortgage_insurance_percentage=current_object[5],
                                       number_of_units=current_object[6], occupancy_status=current_object[7],
                                       original_CLTV=current_object[8], original_RTI_ratio=current_object[9],
                                       original_UPB=current_object[10], original_LTV=current_object[11],
                                       original_interest_rate=current_object[12], channel=current_object[13],
                                       ppm_flag=current_object[14], amortization_type=current_object[15],
                                       property_state=current_object[16], property_type=current_object[17],
                                       postal_code=current_object[18], loan_sequence_number=current_object[19],
                                       loan_purpose=current_object[20], original_loan_term=current_object[21],
                                       number_of_borrowers=current_object[22], seller_name=current_object[23],
                                       servicer_name=current_object[24],
                                       super_conforming_flag=current_object[25],
                                       pre_harp_loan_sequence_number=current_object[26],
                                       program_indicator=current_object[27], harp_indicator=current_object[28],
                                       property_valuation_method=current_object[29],
                                       interest_only_indicator=current_object[30], default=False, year=year,
                                       quarter=quarter
                                       ))
    # query_session.flush()
    print('Standard Data in Year %d added' % year)


def construct_default_joined_by_built_db(session):
    for origination in session.query(Ot.OriginationData).all():
        # seq = origination.loan_sequence_number
        # TODO: check the dimensions of the returned values of the list and make sure it contains 24 entities only
        # zero_balance_code_list = session.query(mpt.Month_Performance_Data_Sample.zero_balance_code).filter(
        #     mpt.Month_Performance_Data_Sample.loan_sequence_number.__eq__(seq)).all()[:23]
        #
        # default = len([*filter(lambda x: x >= 3, zero_balance_code_list)]) > 0
        session.add(jtd.OriginationDataWithDefault(credit_score=origination.credit_score,
                                                   first_payment_date=origination.first_payment_date,
                                                   first_time_homebuyer_flag=origination.first_time_homebuyer_flag,
                                                   maturity_date=origination.maturity_date,
                                                   msa=origination.msa,
                                                   mortgage_insurance_percentage=origination.mortgage_insurance_percentage,
                                                   number_of_units=origination.number_of_units,
                                                   occupancy_status=origination.occupancy_status,
                                                   original_CLTV=origination.original_CLTV,
                                                   original_RTI_ratio=origination.original_RTI_ratio,
                                                   original_UPB=origination.original_UPB,
                                                   original_LTV=origination.original_LTV,
                                                   original_interest_rate=origination.original_interest_rate,
                                                   channel=origination.channel,
                                                   ppm_flag=origination.ppm_flag,
                                                   amortization_type=origination.amortization_type,
                                                   property_state=origination.property_state,
                                                   property_type=origination.property_type,
                                                   postal_code=origination.postal_code,
                                                   loan_sequence_number=origination.loan_sequence_number,
                                                   loan_purpose=origination.loan_purpose,
                                                   original_loan_term=origination.original_loan_term,
                                                   number_of_borrowers=origination.number_of_borrowers,
                                                   seller_name=origination.seller_name,
                                                   servicer_name=origination.servicer_name,
                                                   super_conforming_flag=origination.super_conforming_flag,
                                                   pre_harp_loan_sequence_number=origination.pre_harp_loan_sequence_number,
                                                   program_indicator=origination.program_indicator,
                                                   harp_indicator=origination.harp_indicator,
                                                   property_valuation_method=origination.property_valuation_method,
                                                   interest_only_indicator=origination.interest_only_indicator,
                                                   default=DEFAULT_DEFAULT, year=origination.year,
                                                   quarter=origination.quarter))
        print("standard entity of a loan inserted into the jointed table")
        session.commit()
        session.flush()


def update_default_in_join_by_perf(session):
    for year in range(2020, 2021):
        month_perf_name_prefix = raw_data_path + "standard/annual/historical_data_" + year.__str__() + "/historical_data_time_" + year.__str__()

        for quarter in range(4, 5):
            month_perf_file_name = month_perf_name_prefix + "Q" + quarter.__str__() + ".txt"
            month_perf_df = pd.read_csv(month_perf_file_name, delimiter='|', header=None, dtype=object)
            # month_perf_df.where(pd.notnull(month_perf_df), None, inplace=True)
            # for i in range(month_perf_df.shape[0]):
            #     # get the loan seq number
            #     seq = session.query(Ots.OriginationDataSample).all().iloc[i, 19]
            #     perf_of_seq = month_perf_df[month_perf_df[0] == seq]
            #
            #     zero_balance_code_list = perf_of_seq[8]
            #     default = len([*filter(lambda x: x >= 3, zero_balance_code_list)]) > 0
            #
            #     add_jointed_table_with_quarter_and_default(year, quarter, query_session, origination_df.iloc[i, :])

            loan_with_zero_balance_code = month_perf_df[month_perf_df[8].notnull()]
            default_perf = loan_with_zero_balance_code[pd.to_numeric(loan_with_zero_balance_code[8]) >= 3]
            # default_seq = default_perf[0]
            # default_seq_with_date = session.execute(select(jtd.OriginationDataWithDefault.loan_sequence_number,
            #                                                jtd.OriginationDataWithDefault.first_payment_date).where(
            #     jtd.OriginationDataWithDefault.loan_sequence_number.in_(default_seq)))
            # default_seq_with_date_df = pd.DataFrame(data=default_seq_with_date.all())
            # filter all loan sequence number whose monthly reporting period is far than 24 months to the first payment date
            valid_default_seq = default_perf[pd.to_numeric(default_perf[4]) <= 24][0].drop_duplicates()
            # FIXME: replace with the feature loan age in the perf, not considering 1 year has 12 months only simple minus not applicable
            # valid_default_seq =default_seq_with_date_df[0][pd.to_numeric(default_perf.reset_index(drop=True)[default_perf.reset_index(drop=True)[0]==default_seq_with_date_df[0]][1])-pd.to_numeric(default_seq_with_date_df[1])<=24]
            session.execute(update(jtd.OriginationDataWithDefault).where(
                jtd.OriginationDataWithDefault.loan_sequence_number.in_(valid_default_seq)).values(default=True))
            session.commit()
            session.flush()


# def check_if_default_given_perf(seq, year):


def insert_origination_standard_with_quarter(year, query_session, sample=False):
    if sample:
        origination_file_name = raw_data_path + "standard/annual/sample/sample_" + year.__str__() + "/sample_orig_" + year.__str__() + ".txt"
        # insert_origination_from_txt(file_name, year, query_session, sample)
    else:
        # TODO: confirm the file name convention
        origination_file_name_prefix = raw_data_path + "standard/annual/historical_data_" + year.__str__() + "/historical_data_" + year.__str__()
        month_perf_name_prefix = raw_data_path + "standard/annual/historical_data_" + year.__str__() + "/historical_data_time_" + year.__str__()

        for quarter in range(1, 5):
            if year == 2020 and quarter == 4:
                break
            origination_file_name = origination_file_name_prefix + "Q" + quarter.__str__() + ".txt"
            month_perf_file_name = month_perf_name_prefix + "Q" + quarter.__str__() + ".txt"

            insert_origination_from_txt_with_quarter(origination_file_name, month_perf_name_prefix, year, query_session,
                                                     quarter, sample)


def insert_origination_from_txt_with_quarter(origination_file_name, month_perf_file_name, year, query_session, quarter,
                                             sample):
    # content_arr = np.genfromtxt(file_name, delimiter='|', dtype=object)
    origination_df = pd.read_csv(origination_file_name, delimiter='|', header=None, dtype=object)
    month_perf_df = pd.read_csv(month_perf_file_name, delimiter='|', header=None, dtype=object)
    origination_df[25].fillna(' ', inplace=True)
    origination_df[26].fillna(' ', inplace=True)
    origination_df[28].fillna(' ', inplace=True)
    # replace all nan with None and NULL in the database
    month_perf_df.where(pd.notnull(month_perf_df), None, inplace=True)

    for i in range(origination_df.shape[0]):
        # get the loan seq number
        seq = origination_df.iloc[i, 19]
        perf_of_seq = month_perf_df[month_perf_df[0] == seq]

        zero_balance_code_list = perf_of_seq[8]
        default = len([*filter(lambda x: x >= 3, zero_balance_code_list)]) > 0

        add_jointed_table_with_quarter_and_default(year, quarter, query_session, origination_df.iloc[i, :])

    # session.flush()
    # session.commit()


def individual_x_y(individual_obj):
    obj = individual_obj[0]
    data = [float(obj.credit_score),
            obj.first_time_homebuyer_flag,
            float(obj.msa),
            float(obj.mortgage_insurance_percentage,),
            obj.number_of_units,
            obj.occupancy_status,
            float(obj.original_CLTV),
            float(obj.original_RTI_ratio),
            float(obj.original_UPB),
            float(obj.original_LTV),
            float(obj.original_interest_rate),
            obj.channel,
            obj.ppm_flag,
            obj.amortization_type,
            obj.property_state,
            obj.property_type,
            obj.loan_purpose,
            float(obj.original_loan_term),
            float(obj.number_of_borrowers),
            obj.super_conforming_flag,
            obj.program_indicator,
            obj.harp_indicator,
            obj.property_valuation_method,
            obj.interest_only_indicator,
            obj.quarter,
            obj.year,
            obj.default]
    return data


def encode(data):
    # FIXME: make sure the encoder is consistent for both training and testing
    categorical_data = [1, 4, 5, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23]
    onehot_encoder = pre.OneHotEncoder(handle_unknown='ignore', sparse=False, categories=CATEGORIES)
    onehot_encoder.fit(data[:, categorical_data])
    encoded_categorical_feature = onehot_encoder.transform(data[:, categorical_data])
    encoded_data = np.concatenate(
        (data[:, [0, 2, 3, 6, 7, 8, 9, 10,17, 18, 24, 25]], encoded_categorical_feature),
        axis=1)
    print(datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(
        "%Y-%m-%d %H:%M:%S") + " Data Encoding Complete.")
    return encoded_data


if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:postgres@db', echo=True, future=True)
    jtd.Base.registry.metadata.create_all(engine)
    with Session(engine) as session:
        construct_default_joined_by_built_db(session)
        # update_default_in_join_by_perf(session)
