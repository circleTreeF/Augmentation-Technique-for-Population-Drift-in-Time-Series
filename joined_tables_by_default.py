from sqlalchemy import Column, Integer, String, Float, Numeric, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class OriginationDataWithDefault(Base):
    __tablename__ = 'origination_data_with_default'

    credit_score = Column(Integer)
    first_payment_date = Column(Numeric(6))
    first_time_homebuyer_flag = Column(String(1))
    maturity_date = Column(Numeric(6))
    msa = Column(Numeric(5))
    mortgage_insurance_percentage = Column(Numeric(3))
    number_of_units = Column(Numeric(2))
    occupancy_status = Column(String(1))
    original_CLTV = Column(Numeric(3))
    original_RTI_ratio = Column(Numeric(3))
    original_UPB = Column(Numeric(12))
    original_LTV = Column(Numeric(3))
    original_interest_rate = Column(Float)
    channel = Column(String(1))
    ppm_flag = Column(String(1))
    amortization_type = Column(String(5))
    property_state = Column(String(2))
    property_type = Column(String(2))
    postal_code = Column(Numeric(5))
    loan_sequence_number = Column(String(12), primary_key=True)
    loan_purpose = Column(String(1))
    original_loan_term = Column(Numeric(3))
    number_of_borrowers = Column(Numeric(2))
    seller_name = Column(String(60))
    servicer_name = Column(String(60))
    super_conforming_flag = Column(String(1))
    pre_harp_loan_sequence_number = Column(String(12))
    program_indicator = Column(String(1))
    harp_indicator = Column(String(1))
    property_valuation_method = Column(Numeric(1))
    interest_only_indicator = Column(String(1))
    default = Column(Boolean, default=False)
    year = Column(Numeric(4))
    quarter = Column(Numeric(1))
