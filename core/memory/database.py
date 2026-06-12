import os
from sqlalchemy import create_engine, Column, String, Integer, JSON, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Character(Base):
    """
    Persistent Character Memory.
    Character visual DNA must remain consistent across generations.
    """
    __tablename__ = 'characters'
    
    id = Column(String, primary_key=True)
    canonical_name = Column(String, nullable=False)
    aliases = Column(JSON, default=list)
    status = Column(String, default="alive")
    first_appearance = Column(String)
    last_appearance = Column(String)
    
    # Immutable Visual DNA
    id = Column(String, primary_key=True)
    canonical_name = Column(String, unique=True)
    visual_dna = Column(JSON) # e.g. {"hair": "black", "clothing": "blue robes"}
    dynamic_state = Column(JSON, nullable=True) # e.g. current location, injuries

class Location(Base):
    __tablename__ = 'locations'
    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_name = Column(String, unique=True)
    description = Column(String)
    
class WorldConcept(Base):
    __tablename__ = 'world_concepts'
    id = Column(Integer, primary_key=True, autoincrement=True)
    concept_type = Column(String) # e.g., 'currency', 'sect', 'skill'
    name = Column(String, unique=True)
    description = Column(String)

class MemoryEngine:
    def __init__(self, project_dir: str):
        self.db_path = os.path.join(project_dir, 'memory', 'novel_memory.db')
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_character(self, character_id: str, name: str, dna: dict):
        with self.Session() as session:
            # Check if exists
            existing = session.query(Character).filter_by(canonical_name=name).first()
            if existing:
                return
            char = Character(id=character_id, canonical_name=name, visual_dna=dna)
            session.add(char)
            session.commit()

    def get_character_by_name(self, name: str) -> dict:
        """Fetch a character's data, including visual DNA, by their name."""
        with self.Session() as session:
            # Simple match for now. In reality, should check aliases too.
            char = session.query(Character).filter(Character.canonical_name.ilike(f"%{name}%")).first()
            if char:
                return {
                    "id": char.id,
                    "canonical_name": char.canonical_name,
                    "visual_dna": char.visual_dna,
                    "dynamic_state": char.dynamic_state
                }
            return None

    def add_location(self, name: str, description: str):
        with self.Session() as session:
            existing = session.query(Location).filter_by(canonical_name=name).first()
            if existing:
                return
            loc = Location(canonical_name=name, description=description)
            session.add(loc)
            session.commit()

    def add_world_concept(self, concept_type: str, name: str, description: str):
        with self.Session() as session:
            existing = session.query(WorldConcept).filter_by(name=name).first()
            if existing:
                return
            concept = WorldConcept(concept_type=concept_type, name=name, description=description)
            session.add(concept)
            session.commit()
