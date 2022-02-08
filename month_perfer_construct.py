import sqlalchemy as map
from sqlalchemy import create_engine
import numpy as np
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
from sqlalchemy import ForeignKey
import month_performance_table as Perf

import multiprocessing as mp
import os

raw_data_path = 'raw_data/'
num_processor = 32


def insert_performance_standard(year, query_session, sample=False):
    if sample:
        file_name = raw_data_path + "standard/annual/sample/sample_" + year.__str__() + "/sample_svcg_" + year.__str__() + ".txt"
        insert_performance_from_txt(file_name, year, query_session, sample)
    else:
        file_name_prefix = raw_data_path + "standard/annual/historical_data_" + year.__str__() + "/historical_data_time_" + year.__str__()
        for quarter in range(1, 5):
            if year == 2020 and quarter == 4:
                break
            file_name = file_name_prefix + "Q" + quarter.__str__() + ".txt"
            insert_performance_from_txt_with_quarter(file_name, year, query_session, quarter, sample)


def insert_performance_from_txt(file_name, year, query_session, sample):
    content_df = pd.read_csv(file_name, delimiter='|', header=None, dtype=object)
    # replace all nan for super_conforming_flag with 'N'

    content_df = content_df.where(pd.notnull(content_df), None)
    if sample:
        for i in range(content_df.shape[0]):
            add_object_sample(year, query_session, content_df.iloc[i, :])
        # pool.join()
    session.flush()
    session.commit()


def insert_performance_from_txt_with_quarter(file_name, year, query_session, quarter, sample):
    # content_arr = np.genfromtxt(file_name, delimiter='|', dtype=object)
    content_df = pd.read_csv(file_name, delimiter='|', header=None, dtype=object)
    # replace all nan for super_conforming_flag with 'N'
    content_df = content_df.where(pd.notnull(content_df), None)
    # content_df[25].fillna(' ', inplace=True)
    # content_df[26].fillna(' ', inplace=True)
    # content_df[28].fillna(' ', inplace=True)
    if sample:
        for i in range(content_df.shape[0]):
            add_object_sample(year, query_session, content_df.iloc[i, :])

    else:
        for i in range(content_df.shape[0]):
            add_performance_object_with_quarter(year, quarter, query_session, content_df.iloc[i, :])



def add_object_sample(year, query_session, current_object):
    query_session.add(
        Perf.Month_Performance_Data_Sample(loan_sequence_number=current_object[0],
                                           monthly_reporting_period=current_object[1],
                                           current_actual_UPB=current_object[2],
                                           current_loan_delinquency_status=current_object[3],
                                           loan_age=current_object[4],
                                           remaining_months_to_legal_maturity=current_object[5],
                                           repurchase_flag=current_object[6], modification_flag=current_object[7],
                                           zero_balance_code=current_object[8],
                                           zero_balance_effective_date=current_object[9],
                                           current_interest_rate=current_object[10],
                                           current_deferred_UPB=current_object[11],
                                           ddlpi=current_object[12], mi_recoveries=current_object[13],
                                           net_sales_proceeds=current_object[14], non_mi_recoveries=current_object[15],
                                           expenses=current_object[16], legal_costs=current_object[17],
                                           maintenance_and_preservation_costs=current_object[18],
                                           taxes_and_insurance=current_object[19],
                                           miscellaneous_expenses=current_object[20],
                                           actual_loss_calculation=current_object[21],
                                           modification_cost=current_object[22],
                                           step_modification_flag=current_object[23],
                                           deferred_payment_plan=current_object[24], eltv=current_object[25],
                                           zero_balance_removal_ups=current_object[26],
                                           delinquent_accrued_interest=current_object[27],
                                           delinquency_due_to_disaster=current_object[28],
                                           borrower_assistance_status_code=current_object[29], year=year
                                           ))
    session.flush()
    session.commit()
    print('Sample Performance Data Sample in Year %d added' % year)


# def add_performance_object_with_quarter(year, query_session, current_object):
#     query_session.add(Perf.Month_Performance_Data(loan_sequence_number=current_object[0], monthly_reporting_period=current_object[1],
#                            current_actual_UPB=current_object[2], current_loan_delinquency_status=current_object[3],
#                            loan_age=current_object[4], remaining_months_to_legal_maturity=current_object[5],
#                            repurchase_flag=current_object[6], modification_flag=current_object[7],
#                            zero_balance_code=current_object[8], zero_balance_effective_date=current_object[9],
#                            current_interest_rate=current_object[10], current_deferred_UPB=current_object[11],
#                            ddlpi=current_object[12], mi_recoveries=current_object[13],
#                            net_sales_proceeds=current_object[14], non_mi_recoveries=current_object[15],
#                            expenses=current_object[16], legal_costs=current_object[17],
#                            maintenance_and_preservation_costs=current_object[18],
#                            taxes_and_insurance=current_object[19], miscellaneous_expenses=current_object[20],
#                            actual_loss_calculation=current_object[21],
#                            modification_cost=current_object[22], step_modification_flag=current_object[23],
#                            deferred_payment_plan=current_object[24], eltv=current_object[25],
#                            zero_balance_removal_ups=current_object[26], delinquent_accrued_interest=current_object[27],
#                            delinquency_due_to_disaster=current_object[28],
#                            borrower_assistance_status_code=current_object[29], year=year
#                            ))
#     print('Standard Data in Year %d added' % year)


def add_performance_object_with_quarter(year, quarter, query_session, current_object):
    query_session.add(
        Perf.Month_Performance_Data(loan_sequence_number=current_object[0], monthly_reporting_period=current_object[1],
                                    current_actual_UPB=current_object[2],
                                    current_loan_delinquency_status=current_object[3],
                                    loan_age=current_object[4], remaining_months_to_legal_maturity=current_object[5],
                                    repurchase_flag=current_object[6], modification_flag=current_object[7],
                                    zero_balance_code=current_object[8], zero_balance_effective_date=current_object[9],
                                    current_interest_rate=current_object[10], current_deferred_UPB=current_object[11],
                                    ddlpi=current_object[12], mi_recoveries=current_object[13],
                                    net_sales_proceeds=current_object[14], non_mi_recoveries=current_object[15],
                                    expenses=current_object[16], legal_costs=current_object[17],
                                    maintenance_and_preservation_costs=current_object[18],
                                    taxes_and_insurance=current_object[19], miscellaneous_expenses=current_object[20],
                                    actual_loss_calculation=current_object[21],
                                    modification_cost=current_object[22], step_modification_flag=current_object[23],
                                    deferred_payment_plan=current_object[24], eltv=current_object[25],
                                    zero_balance_removal_ups=current_object[26],
                                    delinquent_accrued_interest=current_object[27],
                                    delinquency_due_to_disaster=current_object[28],
                                    borrower_assistance_status_code=current_object[29], year=year, quarter=quarter
                                    ))

    session.flush()
    session.commit()
    print('Performance Standard Data in Year %d Quarter %d added' % (year, quarter))


if __name__ == '__main__':
    print("Construction Starts")
    engine = create_engine('postgresql://postgres:postgres@db', echo=True, future=True)
    # _session_ = sessionmaker(bind=engine, future=True)
    with Session(engine) as session:
        # print(session.info)
        Perf.Base.registry.metadata.create_all(engine)
        for year in range(1999, 2021):
            insert_performance_standard(year, session)
            # insert_origination_standard(year, session)
        print("Session Ends")
