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

raw_data_path = 'raw_data/'
num_processor = 32


def insert_origination(year, query_session, sample=False):
    if sample:
        file_name = raw_data_path + "sample/sample_" + year.__str__() + "/sample_orig_" + year.__str__() + ".txt"
    else:
        # TODO: confirm the file name convention
        file_name = raw_data_path + year.__str__() + ".txt"
    content_arr = np.genfromtxt(file_name, delimiter='|', dtype=object)
    content_df = pd.read_csv(file_name, delimiter='|', header=None, dtype=object)
    with mp.Pool(processes=32) as pool:
        if sample:
            for i in range(content_df.shape[0]):
                pool.apply_async(add_object_sample, args=(year, query_session, content_df.iloc[i, :]))
        else:
            for i in range(content_df.shape[0]):
                pool.apply_async(add_object, args=(year, query_session, content_df.iloc[i, :]))
        pool.close()
        pool.join()


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
    query_session.flush()
    print('Data in Year %d added' % year)


def add_object(year, query_session, current_object):
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
    query_session.flush()


if __name__ == '__main__':
    print("Construction Starts")
    engine = create_engine('postgresql://postgres:postgres@db', echo=True, future=True)
    # _session_ = sessionmaker(bind=engine, future=True)
    with Session(engine) as session:
        # print(session.info)
        Ots.Base.registry.metadata.create_all(engine)
        Ot.Base.registry.metadata.create_all(engine)
        insert_origination(2020, session, sample=True)
        print("Session Ends")
        # session.commit()
