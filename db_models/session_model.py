from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from db.connection import Base
from datetime import datetime

class SessionToken(Base):
    __tablename__ = "sessions"
    session_id = Column(String(64), primary_key=True)
    uid = Column(Integer, ForeignKey('users.uid'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

    user = relationship("User", backref="sessions")
