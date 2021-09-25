from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date, Numeric
from sqlalchemy import create_engine
import numpy as np
from sqlalchemy.orm import sessionmaker, Session
import test_data as td

Base = declarative_base()

class Test_data(Base):
    __tablename__ = 'test_data'

    numeric = Column(Numeric(6), primary_key=True)
    inte=Column(Integer)
    str=Column(String(1))


if __name__ == '__main__':
    print("Construction Starts")
    engine = create_engine('postgresql://postgres:postgres@db', echo=True, future=True)
    # _session_ = sessionmaker(bind=engine, future=True)
    with Session(engine) as session:
        # print(session.info)
        Base.registry.metadata.create_all(engine)
        session.add(Test_data(numeric='.',inte='11',str='.'))
        session.flush()
        session.commit()
