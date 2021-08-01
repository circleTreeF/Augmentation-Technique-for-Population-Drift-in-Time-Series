from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date, Numeric
from sqlalchemy.orm import relationship
from Origination_Table_Sample import OriginationDataSample

Base = declarative_base()


class origination_set(Base):
    __tablename__ = 'origination_data_set'

    year = Column(Numeric(4))
    origination_table = Column(OriginationDataSample)
