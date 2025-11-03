from sqlalchemy import Column, Integer, String
from db.connection import Base

class User(Base):
    __tablename__ = "users"
    uid = Column(Integer, primary_key=True)  # 기존 데이터베이스의 컬럼명과 일치
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
