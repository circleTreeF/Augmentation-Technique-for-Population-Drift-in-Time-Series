from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, CHAR, Float, Date, Numeric

Base = declarative_base()
engine = create_engine('postgresql://postgres:postgres@db', echo=True,future=True)


class Month_Performance_Data(Base):
    __tablename__ = 'Monthly Performance Data'

    loan_sequence_number = Column('Loan Sequence Number', String(12), primary_key=True)
    monthly_reporting_period = Column('Monthly Reporting Period', Date, primary_key=True)
    current_actual_UPB = Column('Current Actual UPB', Numeric(12))
    current_loan_delinquency_status = Column('Current Loan Delinquency Status', String(3))
    loan_age = Column('Loan Age', Numeric(3))
    remaining_months_to_legal_maturity = Column('Remaining Months To Legal Maturity', Numeric(3))
    repurchase_flag = Column('Repurchase Flag', String(1))
    modification_flag = Column('Modification Flag', String(1))
    zero_balance_code = Column('Zero Balance Code', Numeric(2))
    zero_balance_effective_date = Column('Zero Balance Effective Date', Date)
    current_interest_rate = Column('Current Interest Rate', Numeric(8))
    current_deferred_UPB = Column('Current Deferred UPB', Numeric(12))
    ddlpi = Column('Due Date Of Last Paid Installment', Date)
    mi_recoveries = Column('MI Recoveries', Numeric(12))
    net_sales_proceeds = Column('Net Sales Proceeds', String(14))
    non_mi_recoveries = Column('Non Mi Recoveries', Numeric(12))
    expenses = Column('Expenses', Numeric(12))
    legal_costs = Column('Legal Costs', Numeric(12))
    maintenance_and_preservation_costs = Column('Maintenance and Preservation Costs', Numeric(12))
    taxes_and_insurance = Column('Taxes and Insurance', Numeric(12))
    miscellaneous_expenses = Column('Miscellaneous Expenses', Numeric(12))
    actual_loss_calculation = Column('Actual Loss Calculation', Numeric(12))
    modification_cost = Column('Modification Cost', Numeric(12))
    step_modification_flag = Column('Step Modification Flag', String(1))
    deferred_payment_plan = Column('Deferred Payment Plan', String(1))
    eltv = Column('Estimated Load To Value', Numeric(4))
    zero_balance_removal_ups = Column('Zero Balance Removal UPB', Numeric(12))
    delinquent_accrued_interest = Column('Delinquent Accrued Interest', Numeric(12))
    delinquency_due_to_disaster = Column('Delinquency Due To Disaster', String(1))
    borrower_assistance_status_code = Column('Borrower Assistance Status Code', String(1))
