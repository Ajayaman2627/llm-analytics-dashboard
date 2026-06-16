from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class LLMCall(Base):
    __tablename__ = "llm_calls"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    timestamp       = Column(DateTime, default=datetime.utcnow)
    model           = Column(String(100), nullable=False)
    prompt          = Column(Text, nullable=False)
    response        = Column(Text, nullable=False)
    prompt_tokens   = Column(Integer, default=0)
    response_tokens = Column(Integer, default=0)
    total_tokens    = Column(Integer, default=0)
    latency_ms      = Column(Float, default=0.0)
    cost_usd        = Column(Float, default=0.0)
    session_label   = Column(String(200), default="default")

# Cost per 1000 tokens for each model
MODEL_COSTS = {
    "gpt-4o-mini":               0.000165,
    "gpt-4o":                    0.005,
    "Llama-3.2-11B-Vision-Instruct": 0.0,
    "mistral-small-2503":            0.0,
}

def get_engine(db_path="llm_analytics.db"):
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
