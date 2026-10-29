from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    contact = Column(String, nullable=True)

    records = relationship("MedicalRecord", back_populates="patient")

class MedicalRecord(Base):
    __tablename__ = "medical_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    filename = Column(String)
    file_path = Column(String)
    extracted_text = Column(Text, nullable=True)
    department = Column(String, index=True, nullable=True)
    status = Column(String, default="PENDING") # PENDING, REVIEW, PROCESSED
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="records")
    logs = relationship("ProcessingLog", back_populates="record")

class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True, index=True)
    record_id = Column(Integer, ForeignKey("medical_records.id"))
    step = Column(String)
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    record = relationship("MedicalRecord", back_populates="logs")

class RPAJob(Base):
    __tablename__ = "rpa_jobs"

    id = Column(Integer, primary_key=True, index=True)
    record_id = Column(Integer, ForeignKey("medical_records.id"))
    status = Column(String, default="QUEUED") # QUEUED, COMPLETED, FAILED
    payload = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
