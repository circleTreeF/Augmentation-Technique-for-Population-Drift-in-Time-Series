import sqlalchemy as map
from sqlalchemy import create_engine
import numpy as np
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
from sqlalchemy import ForeignKey
import Origination_Table_Sample as Ots
import Origination_Table as Ot
import multiprocessing as mp
import os
import joined_tables_by_default as jtd

raw_data_path = 'raw_data/'
num_processor = 32


def insert_origination_standard(year, query_session, sample=False):
    if sample:
        file_name = raw_data_path + "standard/annual/sample/sample_" + year.__str__() + "/sample_orig_" + year.__str__() + ".txt"
        insert_origination_from_txt(file_name, year, query_session, sample)
    else:
        # TODO: confirm the file name convention
        file_name_prefix = raw_data_path + "standard/annual/historical_data_" + year.__str__() + "/historical_data_" + year.__str__()
        for quarter in range(1, 5):
            if year == 2020 and quarter == 4:
                break
            file_name = file_name_prefix + "Q" + quarter.__str__() + ".txt"
            insert_origination_from_txt(file_name, year, query_session, sample)


def insert_origination_standard_with_quarter(year, query_session, sample=False):
    if sample:
        file_name = raw_data_path + "standard/annual/sample/sample_" + year.__str__() + "/sample_orig_" + year.__str__() + ".txt"
        insert_origination_from_txt(file_name, year, query_session, sample)
    else:
        # TODO: confirm the file name convention
        file_name_prefix = raw_data_path + "standard/annual/historical_data_" + year.__str__() + "/historical_data_" + year.__str__()
        for quarter in range(4, 5):
            # if year == 2020 and quarter == 4:
            #     break
            file_name = file_name_prefix + "Q" + quarter.__str__() + ".txt"
            insert_origination_from_txt_with_quarter(file_name, year, query_session, quarter, sample)


def insert_origination_from_txt(file_name, year, query_session, sample):
    content_arr = np.genfromtxt(file_name, delimiter='|', dtype=object)
    content_df = pd.read_csv(file_name, delimiter='|', header=None, dtype=object)
    # replace all nan for super_conforming_flag with 'N'
    content_df[4].fillna('00000', inplace=True)
    content_df[12].replace('.', '0.0', inplace=True)
    content_df.fillna(' ', inplace=True)
    # content_df[25].fillna(' ', inplace=True)
    # content_df[26].fillna(' ', inplace=True)
    # content_df[28].fillna(' ', inplace=True)
    # with mp.Pool(num_processor) as pool:
    if sample:
        for i in range(content_df.shape[0]):
            add_object_sample(year, query_session, content_df.iloc[i, :])
        #     pool.apply_async(add_object_sample, args=(year, query_session, content_df.iloc[i, :]))
        # pool.close()
        # pool.join()
    else:
        for i in range(content_df.shape[0]):
            add_origination_object(year, query_session, content_df.iloc[i, :])
            # pool.apply_async(add_object, args=(year, query_session, content_df.iloc[i, :]))
        # pool.close()
        # pool.join()
    session.flush()
    session.commit()


def insert_origination_from_txt_with_quarter(file_name, year, query_session, quarter, sample):
    # content_arr = np.genfromtxt(file_name, delimiter='|', dtype=object)
    content_df = pd.read_csv(file_name, delimiter='|', header=None, dtype=object)
    content_df[12].replace('.', '0.0', inplace=True)

    # replace all nan for super_conforming_flag with 'N'
    content_df[4].fillna('00000', inplace=True)
    content_df.where(pd.notnull(content_df), None, inplace=True)

    # content_df[25].fillna(' ', inplace=True)
    # content_df[26].fillna(' ', inplace=True)
    # content_df[28].fillna(' ', inplace=True)
    # with mp.Pool(num_processor) as pool:
    if sample:
        for i in range(content_df.shape[0]):
            add_object_sample(year, query_session, content_df.iloc[i, :])
        #     pool.apply_async(add_object_sample, args=(year, query_session, content_df.iloc[i, :]))
        # pool.close()
        # pool.join()
    else:
        for i in range(content_df.shape[0]):
            add_origination_object_with_quarter(year, quarter, query_session, content_df.iloc[i, :])
            # pool.apply_async(add_object, args=(year, query_session, content_df.iloc[i, :]))
        # pool.close()
        # pool.join()
    session.flush()
    session.commit()


'''
This function is to add single object into the cache memory for the database query
'''


def add_object_sample(year, query_session, current_object):
    query_session.add(Ots.OriginationDataSample(credit_score=current_object[0], first_payment_date=current_object[1],
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
                                                interest_only_indicator=current_object[30], year=year
                                                ))
    # query_session.flush()
    # query_session.commit()
    print('Data Sample in Year %d added' % year)


def add_origination_object(year, query_session, current_object):
    query_session.add(Ot.OriginationData(credit_score=current_object[0], first_payment_date=current_object[1],
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
                                         interest_only_indicator=current_object[30], year=year
                                         ))
    # query_session.flush()
    print('Standard Data in Year %d added' % year)


def add_origination_object_with_quarter(year, quarter, query_session, current_object):
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
    query_session.commit()
    query_session.flush()
    print('Standard Data in Year %d added' % year)


if __name__ == '__main__':
    print("Construction Starts")
    engine = create_engine('postgresql://postgres:postgres@db', echo=True, future=True)
    # _session_ = sessionmaker(bind=engine, future=True)
    with Session(engine) as session:
        # print(session.info)
        Ots.Base.registry.metadata.create_all(engine)
        Ot.Base.registry.metadata.create_all(engine)
        jtd.Base.registry.metadata.create_all(engine)
        for year in range(2020, 2021):
            insert_origination_standard_with_quarter(year, session)
            # insert_origination_standard(year, session)
        print("Session Ends")
        # session.commit()
