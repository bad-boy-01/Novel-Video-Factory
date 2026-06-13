import os
from sqlalchemy import create_engine, Column, String, Integer, JSON, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Character(Base):
    """
    Persistent Character Memory.
    Character visual DNA must remain consistent across generations.
    """
    __tablename__ = 'characters'
    
    id = Column(String, primary_key=True)
    canonical_name = Column(String, nullable=False, unique=True)
    aliases = Column(JSON, default=list)
    status = Column(String, default="alive")
    first_appearance = Column(String)
    last_appearance = Column(String)
    
    # Immutable Visual DNA
    visual_dna = Column(JSON) # e.g. {"hair": "black", "clothing": "blue robes"}
    dynamic_state = Column(JSON, nullable=True) # e.g. current location, injuries

class CharacterState(Base):
    __tablename__ = 'character_states'
    id = Column(Integer, primary_key=True, autoincrement=True)
    character_id = Column(String, ForeignKey('characters.id'))
    chapter_start = Column(Integer, default=1)
    chapter_end = Column(Integer, nullable=True)
    
    # Step 2: Visual DNA overrides
    state_visual_dna = Column(JSON, default=dict)
    
    # Step 3: Extended fields
    age = Column(String, nullable=True)
    outfit_id = Column(String, nullable=True)
    hairstyle_id = Column(String, nullable=True)
    expression_profile = Column(String, nullable=True)
    reference_image = Column(String, nullable=True)
    pose_reference = Column(String, nullable=True)
    style_profile = Column(String, nullable=True)

class TimelineEvent(Base):
    __tablename__ = 'timeline_events'
    id = Column(Integer, primary_key=True, autoincrement=True)
    character_id = Column(String, ForeignKey('characters.id'))
    chapter = Column(Integer)
    event_type = Column(String) # e.g. "injury", "breakthrough", "status_change"
    description = Column(String)
    visual_impact = Column(JSON, nullable=True) # Override DNA temporarily/permanently

class Relationship(Base):
    __tablename__ = 'relationships'
    id = Column(Integer, primary_key=True, autoincrement=True)
    char1_id = Column(String, ForeignKey('characters.id'))
    char2_id = Column(String, ForeignKey('characters.id'))
    relationship_type = Column(String) # "master-disciple", "enemy", "romantic"
    staging_metadata = Column(JSON, nullable=True) # rules for spatial positioning

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

    def get_character_by_name(self, name: str, chapter: int = None) -> dict:
        """Fetch a character's data, including visual DNA, by their name."""
        with self.Session() as session:
            # Simple match for now. In reality, should check aliases too.
            char = session.query(Character).filter(Character.canonical_name.ilike(f"%{name}%")).first()
            if char:
                # Base DNA
                merged_dna = dict(char.visual_dna or {})
                
                # Fetch the active CharacterState
                if chapter is not None:
                    active_state = session.query(CharacterState).filter(
                        CharacterState.character_id == char.id,
                        CharacterState.chapter_start <= chapter,
                        (CharacterState.chapter_end >= chapter) | (CharacterState.chapter_end == None)
                    ).first()
                else:
                    active_state = session.query(CharacterState).filter(
                        CharacterState.character_id == char.id
                    ).order_by(CharacterState.id.desc()).first()
                
                state_data = {}
                if active_state:
                    # Merge DNA overrides
                    if active_state.state_visual_dna:
                        for k, v in active_state.state_visual_dna.items():
                            merged_dna[k] = v
                    
                    # Gather state metadata
                    state_data = {
                        "age": active_state.age,
                        "outfit_id": active_state.outfit_id,
                        "hairstyle_id": active_state.hairstyle_id,
                        "expression_profile": active_state.expression_profile,
                        "reference_image": active_state.reference_image,
                        "pose_reference": active_state.pose_reference,
                        "style_profile": active_state.style_profile
                    }
                
                return {
                    "id": char.id,
                    "canonical_name": char.canonical_name,
                    "visual_dna": merged_dna,
                    "dynamic_state": char.dynamic_state,
                    "state_metadata": state_data
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
