\
from __future__ import annotations
import datetime as dt
from typing import Optional, List, Tuple

from sqlalchemy import (
    create_engine, Column, Integer, String, LargeBinary, ForeignKey, DateTime, func, select
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DB_URL = "sqlite:///faces.db"

engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()

class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), unique=True, nullable=False, index=True)
    info = Column(String(250), nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    faces = relationship("FaceEmb", back_populates="person", cascade="all, delete-orphan")

class FaceEmb(Base):
    __tablename__ = "faces"
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False, index=True)
    embedding = Column(LargeBinary, nullable=False)  # float32 bytes
    img = Column(LargeBinary, nullable=True)         # small cropped face image (JPEG/PNG)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    person = relationship("Person", back_populates="faces")

def init_db() -> None:
    Base.metadata.create_all(bind=engine)

def get_or_create_person(session, name: str, info: Optional[str] = None) -> Person:
    person = session.execute(select(Person).where(Person.name == name)).scalar_one_or_none()
    if person:
        if info and person.info != info:
            person.info = info
            session.commit()
        return person
    person = Person(name=name, info=info)
    session.add(person)
    session.commit()
    session.refresh(person)
    return person

def add_face(session, person_id: int, embedding_bytes: bytes, face_img_bytes: Optional[bytes]) -> FaceEmb:
    face = FaceEmb(person_id=person_id, embedding=embedding_bytes, img=face_img_bytes)
    session.add(face)
    session.commit()
    session.refresh(face)
    return face

def list_persons_with_counts(session) -> List[Tuple[Person, int]]:
    stmt = (
        select(Person, func.count(FaceEmb.id))
        .join(FaceEmb, FaceEmb.person_id == Person.id, isouter=True)
        .group_by(Person.id)
        .order_by(Person.name.asc())
    )
    return list(session.execute(stmt).all())

def list_faces_for_person(session, person_id: int) -> List[FaceEmb]:
    stmt = select(FaceEmb).where(FaceEmb.person_id == person_id).order_by(FaceEmb.created_at.desc())
    return [row[0] for row in session.execute(stmt).all()]

def all_faces(session) -> List[FaceEmb]:
    stmt = select(FaceEmb).order_by(FaceEmb.created_at.desc())
    return [row[0] for row in session.execute(stmt).all()]

def delete_person(session, person_id: int) -> None:
    person = session.get(Person, person_id)
    if person:
        session.delete(person)
        session.commit()

def delete_face(session, face_id: int) -> None:
    face = session.get(FaceEmb, face_id)
    if face:
        session.delete(face)
        session.commit()
