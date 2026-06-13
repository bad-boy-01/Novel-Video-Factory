import os
from typing import Dict, List
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
    background_path = Column(String, nullable=True)
    
class WorldConcept(Base):
    __tablename__ = 'world_concepts'
    id = Column(Integer, primary_key=True, autoincrement=True)
    concept_type = Column(String) # e.g., 'currency', 'sect', 'skill'
    name = Column(String, unique=True)
    description = Column(String)

class MemoryEngine:
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.db_path = os.path.join(project_dir, 'memory', 'novel_memory.db')
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def get_all_characters(self) -> List[Dict]:
        """Fetch all characters currently in the database."""
        session = self.Session()
        try:
            chars = session.query(Character).all()
            return [
                {
                    "id": char.id,
                    "canonical_name": char.canonical_name,
                    "visual_dna": char.visual_dna
                } for char in chars
            ]
        finally:
            session.close()

    def add_character(self, character_id: str, name: str, dna: dict) -> bool:
        session = self.Session()
        try:
            # Check if exists
            existing = session.query(Character).filter_by(canonical_name=name).first()
            if existing:
                return False
            char = Character(id=character_id, canonical_name=name, visual_dna=dna)
            session.add(char)
            session.commit()
            return True
        finally:
            session.close()

    def get_character_by_name(self, name: str, chapter: int = None) -> dict:
        """
        V3 Upgrade: Visual State Engine.
        Automatically merges Base DNA with CharacterState overrides.
        """
        session = self.Session()
        try:
            # Case-insensitive partial match
            char = session.query(Character).filter(Character.canonical_name.ilike(f"%{name}%")).first()
            if not char:
                return None

            # 1. Start with Base DNA
            merged_dna = dict(char.visual_dna or {})
            
            # 2. Fetch Active State
            if chapter is not None:
                active_state = session.query(CharacterState).filter(
                    CharacterState.character_id == char.id,
                    CharacterState.chapter_start <= chapter,
                    (CharacterState.chapter_end >= chapter) | (CharacterState.chapter_end == None)
                ).order_by(CharacterState.id.desc()).first()
            else:
                active_state = session.query(CharacterState).filter(
                    CharacterState.character_id == char.id
                ).order_by(CharacterState.id.desc()).first()
            
            state_meta = {}
            if active_state:
                # 3. Automatic Merge: State overrides base attributes
                if active_state.state_visual_dna:
                    for k, v in active_state.state_visual_dna.items():
                        if v and str(v).lower() not in ['none', 'not specified']:
                            merged_dna[k] = v
                
                # 4. Map Extended Visual Fields (V3 Migration)
                state_meta = {
                    "age": active_state.age,
                    "outfit_id": active_state.outfit_id,
                    "hairstyle_id": active_state.hairstyle_id,
                    "expression": active_state.expression_profile,
                    "reference_image": active_state.reference_image,
                    "pose": active_state.pose_reference,
                    "style": active_state.style_profile
                }
                
                # Inject extended fields into DNA if they exist
                for k, v in state_meta.items():
                    if v:
                        merged_dna[k] = v

            return {
                "id": char.id,
                "canonical_name": char.canonical_name,
                "visual_dna": merged_dna,
                "state_metadata": state_meta
            }
        finally:
            session.close()

    def add_location(self, name: str, description: str) -> bool:
        session = self.Session()
        try:
            existing = session.query(Location).filter_by(canonical_name=name).first()
            if existing:
                return False
            loc = Location(canonical_name=name, description=description)
            session.add(loc)
            session.commit()
            return True
        finally:
            session.close()

    def add_relationship(self, name1: str, name2: str, rel_type: str, staging: str = "") -> bool:
        session = self.Session()
        try:
            # V3 Upgrade: Relationship Hierarchy
            # Higher number = higher priority. We don't overwrite high priority with low.
            PRIORITY = {
                'family': 100, 'spouse': 100, 'husband': 100, 'wife': 100,
                'brother': 90, 'sister': 90, 'parent': 90, 'child': 90, 'father': 90, 'mother': 90,
                'master-disciple': 80, 'teacher': 80, 'student': 80,
                'enemy': 70, 'rival': 70,
                'friend': 50, 'ally': 50,
                'neutral': 10, 'acquaintance': 10, 'unknown': 0
            }
            
            def get_priority(t):
                t = t.lower()
                for key, val in PRIORITY.items():
                    if key in t: return val
                return 5 # Default for unknown types
            
            # Resolve IDs
            c1 = session.query(Character).filter(Character.canonical_name == name1).first()
            c2 = session.query(Character).filter(Character.canonical_name == name2).first()
            if not c1 or not c2:
                return False
                
            # Check if exists (either way)
            existing = session.query(Relationship).filter(
                ((Relationship.char1_id == c1.id) & (Relationship.char2_id == c2.id)) |
                ((Relationship.char1_id == c2.id) & (Relationship.char2_id == c1.id))
            ).first()
            
            new_priority = get_priority(rel_type)
            
            if existing:
                old_priority = get_priority(existing.relationship_type)
                if new_priority >= old_priority:
                    existing.relationship_type = rel_type
                    if staging:
                        existing.staging_metadata = {"staging": staging}
                else:
                    logger.debug(f"Ignoring lower priority relationship update: {rel_type} vs {existing.relationship_type}")
                    return False
            else:
                rel = Relationship(char1_id=c1.id, char2_id=c2.id, relationship_type=rel_type, staging_metadata={"staging": staging})
                session.add(rel)
            
            session.commit()
            return True
        finally:
            session.close()

    def get_all_relationships(self) -> List[dict]:
        """Fetch all relationships with character names for LLM context."""
        session = self.Session()
        try:
            rels = session.query(Relationship).all()
            results = []
            for r in rels:
                c1 = session.query(Character).filter_by(id=r.char1_id).first()
                c2 = session.query(Character).filter_by(id=r.char2_id).first()
                if c1 and c2:
                    results.append({
                        "char1": c1.canonical_name,
                        "char2": c2.canonical_name,
                        "type": r.relationship_type
                    })
            return results
        finally:
            session.close()

    def get_relationship_staging(self, names: List[str]) -> str:
        """Returns staging tags based on relationships between characters."""
        if len(names) < 2:
            return ""
            
        session = self.Session()
        try:
            staging_tags = []
            # Check all pairs
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    n1, n2 = names[i], names[j]
                    c1 = session.query(Character).filter(Character.canonical_name == n1).first()
                    c2 = session.query(Character).filter(Character.canonical_name == n2).first()
                    if c1 and c2:
                        rel = session.query(Relationship).filter(
                            ((Relationship.char1_id == c1.id) & (Relationship.char2_id == c2.id)) |
                            ((Relationship.char1_id == c2.id) & (Relationship.char2_id == c1.id))
                        ).first()
                        if rel:
                            # Map relationship to visual staging
                            rel_type = rel.relationship_type.lower()
                            staging_note = rel.staging_metadata.get('staging', '') if rel.staging_metadata else ''
                            
                            if 'enemy' in rel_type or 'aggressive' in staging_note:
                                staging_tags.append("combat stance, weapons drawn, glaring at each other")
                            elif 'master-disciple' in rel_type or 'respectful' in staging_note:
                                staging_tags.append("respectful distance, bow, standing slightly behind")
                            elif 'romantic' in rel_type or 'close' in staging_note:
                                staging_tags.append("standing close together, holding hands, intimate")
                            elif 'family' in rel_type or 'protective' in staging_note:
                                staging_tags.append("protective posture, standing side by side")
                                
            return ", ".join(list(set(staging_tags)))
        finally:
            session.close()

    def update_location_background(self, name: str, path: str) -> bool:
        session = self.Session()
        try:
            loc = session.query(Location).filter_by(canonical_name=name).first()
            if loc:
                loc.background_path = path
                session.commit()
                return True
            return False
        finally:
            session.close()

    def add_world_concept(self, concept_type: str, name: str, description: str) -> bool:
        session = self.Session()
        try:
            existing = session.query(WorldConcept).filter_by(name=name).first()
            if existing:
                return False
            concept = WorldConcept(concept_type=concept_type, name=name, description=description)
            session.add(concept)
            session.commit()
            return True
        finally:
            session.close()

    def get_all_locations(self) -> List[dict]:
        session = self.Session()
        try:
            locs = session.query(Location).all()
            return [{"canonical_name": l.canonical_name, "description": l.description, "background_path": l.background_path} for l in locs]
        finally:
            session.close()
