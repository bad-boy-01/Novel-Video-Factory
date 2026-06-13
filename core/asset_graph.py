import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class NodeDependency:
    def __init__(self, node_id: str, version: str, hash_val: str):
        self.node_id = node_id
        self.version = version
        self.hash_val = hash_val

class AssetGraph:
    """
    Phase 0: Asset Dependency Graph.
    Tracks exact lineage of generated artifacts to support intelligent incremental rendering.
    """
    def __init__(self, project_dir: str):
        self.graph_path = os.path.join(project_dir, 'output', 'asset_graph.json')
        self.nodes = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save(self):
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        with open(self.graph_path, 'w', encoding='utf-8') as f:
            json.dump(self.nodes, f, indent=2)

    def register_node(self, node_id: str, node_type: str, version: str, hash_val: str, dependencies: List[NodeDependency], outputs: List[str]):
        """
        Registers an asset in the graph. 
        e.g., node_id: "scene341.json", type: "scene", deps: [CharacterState12, LocationState5]
        """
        self.nodes[node_id] = {
            "type": node_type,
            "version": version,
            "hash": hash_val,
            "dependencies": [{"id": d.node_id, "version": d.version, "hash": d.hash_val} for d in dependencies],
            "outputs": outputs
        }
        self.save()
        logger.debug(f"Registered node {node_id} in Asset Graph.")

    def check_invalidation(self, node_id: str, current_dependencies: List[NodeDependency]) -> bool:
        """
        Checks if a node's dependencies have changed. If true, the node needs regeneration.
        """
        if node_id not in self.nodes:
            return True # Missing, needs generation
            
        recorded_deps = {d["id"]: d for d in self.nodes[node_id]["dependencies"]}
        
        for curr_dep in current_dependencies:
            if curr_dep.node_id not in recorded_deps:
                return True # New dependency
            rec_dep = recorded_deps[curr_dep.node_id]
            if rec_dep["hash"] != curr_dep.hash_val or rec_dep["version"] != curr_dep.version:
                return True # Dependency changed
                
        return False
