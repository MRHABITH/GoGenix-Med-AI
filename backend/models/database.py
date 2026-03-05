from sqlalchemy import Integer, String, Float, DateTime, Text, Column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from datetime import datetime
import os

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    disease_type = Column(String)
    prediction_label = Column(String)
    confidence_score = Column(Float)
    risk_level = Column(String)
    patient_age = Column(String, nullable=True)
    guidance_notes = Column(Text)
    heatmap_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/medical_ai")
# engine = create_engine(DATABASE_URL) # Uncomment when DB is ready
