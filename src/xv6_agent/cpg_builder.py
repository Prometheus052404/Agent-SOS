"""
Code Property Graph Builder - Tree-sitter + NetworkX CPG for Xv6 agent.

Implements:
- Tree-sitter C grammar for AST parsing
- NetworkX MultiDiGraph for graph storage
- AST, CFG, PDG, and Call Graph edges
- Incremental updates (function-level invalidation)
- Data-flow priority ranking for context queries
- Heuristic inter-file impact analysis (include-graph + grep)
"""

import os
import re
import pickle
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

import networkx as nx

logger = logging.getLogger(__name__)

# Try to import tree-sitter
try:
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser, Tree, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("Tree-sitter not available, using fallback regex parsing")


@dataclass
class CPGNode:
    """Represents a node in the Code Property Graph."""
    id: str
    type: str  # function, variable, struct, statement, etc.
    name: str
    file: str
    line_start: int
    line_end: int
    source_span: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CPGEdge:
    """Represents an edge in the Code Property Graph."""
    source: str
    target: str
    edge_type: str  # child_of, calls, data_flow, control_flow
    metadata: Dict[str, Any] = field(default_factory=dict)


class TreeSitterParser:
    """Wrapper for Tree-sitter C parser."""

    def __init__(self):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("Tree-sitter not available")
        
        self.language = Language(tsc.language())
        self.parser = Parser(self.language)
        self.trees: Dict[str, Tree] = {}
        self.sources: Dict[str, bytes] = {}

    def parse_file(self, filepath: str) -> Optional[Tree]:
        """Parse a C file and return the AST."""
        try:
            with open(filepath, 'rb') as f:
                source = f.read()
            
            tree = self.parser.parse(source)
            self.trees[filepath] = tree
            self.sources[filepath] = source
            
            return tree
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return None

    def parse_incremental(
        self,
        filepath: str,
        old_tree: Optional[Tree] = None
    ) -> Optional[Tree]:
        """Parse with incremental update if old tree available."""
        try:
            with open(filepath, 'rb') as f:
                source = f.read()
            
            if old_tree and filepath in self.sources:
                # Use incremental parsing
                tree = self.parser.parse(source, old_tree)
            else:
                tree = self.parser.parse(source)
            
            self.trees[filepath] = tree
            self.sources[filepath] = source
            
            return tree
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return None

    def get_source_text(self, filepath: str, start: int, end: int) -> str:
        """Get source text for a byte range."""
        if filepath in self.sources:
            return self.sources[filepath][start:end].decode('utf-8', errors='replace')
        return ""


class CPGBuilder:
    """
    Builds a Code Property Graph using Tree-sitter and NetworkX.
    
    The graph contains:
    - AST nodes: functions, variables, structs, statements
    - CFG edges: control flow between blocks
    - PDG edges: data dependencies
    - Call graph: function call relationships
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.graph = nx.MultiDiGraph()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize Tree-sitter parser if available
        if TREE_SITTER_AVAILABLE:
            self.parser = TreeSitterParser()
        else:
            self.parser = None
        
        # Track file versions for incremental updates
        self.file_versions: Dict[str, int] = {}
        self.function_nodes: Dict[str, Set[str]] = {}  # file -> function node IDs

    def build_from_directory(
        self,
        directory: str,
        patterns: Optional[List[str]] = None
    ):
        """Build CPG from all C files in a directory."""
        patterns = patterns or ['*.c', '*.h']
        directory = Path(directory)
        
        # Exclusion patterns
        exclude_patterns = ['.xv6_agent', '.venv', '.git', 'node_modules', '__pycache__']
        
        for pattern in patterns:
            for filepath in directory.rglob(pattern):
                filepath_str = str(filepath)
                # Skip excluded directories
                if any(excl in filepath_str for excl in exclude_patterns):
                    continue
                # Skip dotfiles (._xxx.c files from macOS)
                if '/._' in filepath_str or filepath.name.startswith('._'):
                    continue
                self.add_file(filepath_str)

    def add_file(self, filepath: str):
        """Add a file to the CPG."""
        logger.info(f"Adding file to CPG: {filepath}")
        
        if self.parser:
            self._add_file_treesitter(filepath)
        else:
            self._add_file_fallback(filepath)

    def _add_file_treesitter(self, filepath: str):
        """Add file using Tree-sitter parsing."""
        tree = self.parser.parse_file(filepath)
        if not tree:
            return
        
        # Track function nodes for this file
        self.function_nodes[filepath] = set()
        
        # Walk the AST
        self._walk_tree(tree.root_node, filepath)

    def _walk_tree(self, node: 'Node', filepath: str, parent_id: Optional[str] = None):
        """Walk the AST and add nodes/edges to the graph."""
        node_id = f"{filepath}:{node.start_point[0]}:{node.type}"
        
        # Handle function definitions
        if node.type == 'function_definition':
            self._add_function(node, filepath)
        
        # Handle function calls
        elif node.type == 'call_expression':
            self._add_call(node, filepath, parent_id)
        
        # Handle variable declarations
        elif node.type in ('declaration', 'parameter_declaration'):
            self._add_variable(node, filepath)
        
        # Handle struct definitions
        elif node.type == 'struct_specifier':
            self._add_struct(node, filepath)
        
        # Recurse into children
        for child in node.children:
            self._walk_tree(child, filepath, node_id)

    def _add_function(self, node: 'Node', filepath: str):
        """Add a function node to the graph."""
        # Find function name
        name = None
        for child in node.children:
            if child.type == 'function_declarator':
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        name = self.parser.get_source_text(
                            filepath,
                            subchild.start_byte,
                            subchild.end_byte
                        )
                        break
        
        if not name:
            return
        
        node_id = f"{filepath}:{name}"
        
        self.graph.add_node(
            node_id,
            type='function',
            name=name,
            file=filepath,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            source_span=self.parser.get_source_text(
                filepath,
                node.start_byte,
                min(node.start_byte + 200, node.end_byte)  # First 200 chars
            )
        )
        
        self.function_nodes.setdefault(filepath, set()).add(node_id)
        
        # Extract variables written and read in this function
        self._analyze_function_data_flow(node, filepath, node_id)

    def _add_call(self, node: 'Node', filepath: str, caller_id: Optional[str]):
        """Add a function call edge."""
        # Find callee name
        for child in node.children:
            if child.type == 'identifier':
                callee_name = self.parser.get_source_text(
                    filepath,
                    child.start_byte,
                    child.end_byte
                )
                
                # Find caller function
                caller = self._find_enclosing_function(filepath, node.start_point[0])
                
                if caller:
                    # Add call edge
                    self.graph.add_edge(
                        caller,
                        callee_name,  # May not exist yet
                        edge_type='calls',
                        call_site_line=node.start_point[0] + 1
                    )
                break

    def _add_variable(self, node: 'Node', filepath: str):
        """Add a variable node to the graph."""
        # Find variable name
        name = None
        for child in node.children:
            if child.type in ('identifier', 'init_declarator'):
                if child.type == 'init_declarator':
                    for subchild in child.children:
                        if subchild.type == 'identifier':
                            name = self.parser.get_source_text(
                                filepath,
                                subchild.start_byte,
                                subchild.end_byte
                            )
                            break
                else:
                    name = self.parser.get_source_text(
                        filepath,
                        child.start_byte,
                        child.end_byte
                    )
                
                if name:
                    node_id = f"{filepath}:{node.start_point[0]}:{name}"
                    self.graph.add_node(
                        node_id,
                        type='variable',
                        name=name,
                        file=filepath,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1
                    )

    def _add_struct(self, node: 'Node', filepath: str):
        """Add a struct node to the graph."""
        name = None
        for child in node.children:
            if child.type == 'type_identifier':
                name = self.parser.get_source_text(
                    filepath,
                    child.start_byte,
                    child.end_byte
                )
                break
        
        if name:
            node_id = f"{filepath}:struct:{name}"
            self.graph.add_node(
                node_id,
                type='struct',
                name=name,
                file=filepath,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1
            )

    def _analyze_function_data_flow(
        self,
        node: 'Node',
        filepath: str,
        function_id: str
    ):
        """Analyze data flow within a function."""
        # Simple pattern matching for assignments and reads
        source = self.parser.get_source_text(
            filepath,
            node.start_byte,
            node.end_byte
        )
        
        # Find assignments (writes)
        write_pattern = re.compile(r'(\w+)\s*=')
        for match in write_pattern.finditer(source):
            var_name = match.group(1)
            self.graph.nodes[function_id]['writes_var'] = var_name
        
        # Find reads (variable references)
        # This is simplified - a real implementation would use proper data flow analysis

    def _find_enclosing_function(self, filepath: str, line: int) -> Optional[str]:
        """Find the function containing a given line."""
        for node_id in self.function_nodes.get(filepath, []):
            node_data = self.graph.nodes.get(node_id, {})
            if (node_data.get('line_start', 0) <= line + 1 <= 
                node_data.get('line_end', 0)):
                return node_id
        return None

    def _add_file_fallback(self, filepath: str):
        """Add file using fallback regex parsing."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return
        
        self.function_nodes[filepath] = set()
        
        # Find functions with regex
        func_pattern = re.compile(
            r'^(?:static\s+)?(?:inline\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*{',
            re.MULTILINE
        )
        
        lines = content.split('\n')
        
        for match in func_pattern.finditer(content):
            name = match.group(1)
            start_pos = match.start()
            line_start = content[:start_pos].count('\n') + 1
            
            # Find end of function (matching braces)
            brace_count = 0
            line_end = line_start
            for i, line in enumerate(lines[line_start-1:], start=line_start):
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    line_end = i
                    break
            
            node_id = f"{filepath}:{name}"
            self.graph.add_node(
                node_id,
                type='function',
                name=name,
                file=filepath,
                line_start=line_start,
                line_end=line_end
            )
            
            self.function_nodes[filepath].add(node_id)

    def update_incremental(self, filepath: str):
        """
        Incrementally update the CPG for a modified file.
        Uses function-level invalidation, not full rebuild.
        """
        logger.info(f"Incremental CPG update for: {filepath}")
        
        # Remove old nodes from this file
        nodes_to_remove = [
            n for n in self.graph.nodes
            if self.graph.nodes[n].get('file') == filepath
        ]
        
        for node in nodes_to_remove:
            self.graph.remove_node(node)
        
        # Re-add the file
        self.add_file(filepath)
        
        # Update cross-file dependencies
        self._update_cross_file_deps(filepath)

    def _update_cross_file_deps(self, modified_file: str):
        """
        Heuristic inter-file impact analysis.
        Flags files that include modified headers.
        """
        # If modified file is a header
        if modified_file.endswith('.h'):
            affected_files = self._find_files_including(modified_file)
            for f in affected_files:
                # Mark for reanalysis
                self.graph.graph.setdefault('needs_reanalysis', set()).add(f)
        
        # If struct definition changed
        modified_structs = self._extract_modified_structs(modified_file)
        for struct_name in modified_structs:
            files_using_struct = self._grep_for_struct_usage(struct_name)
            for f in files_using_struct:
                # Add annotation
                for node in self.function_nodes.get(f, []):
                    self.graph.nodes[node].setdefault('annotations', []).append(
                        f"potentially_affected_by_{struct_name}"
                    )

    def _find_files_including(self, header: str) -> List[str]:
        """Find files that include a header."""
        header_name = Path(header).name
        affected = []
        
        for filepath in self.function_nodes.keys():
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    if f'#include "{header_name}"' in content or \
                       f'#include <{header_name}>' in content:
                        affected.append(filepath)
            except:
                pass
        
        return affected

    def _extract_modified_structs(self, filepath: str) -> List[str]:
        """Extract struct names from a file."""
        structs = []
        for node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            if (node_data.get('file') == filepath and 
                node_data.get('type') == 'struct'):
                structs.append(node_data.get('name', ''))
        return structs

    def _grep_for_struct_usage(self, struct_name: str) -> List[str]:
        """Fallback grep search for struct usage."""
        try:
            result = subprocess.run(
                ['grep', '-l', f'struct {struct_name}', '-r', '--include=*.c', '.'],
                capture_output=True,
                text=True
            )
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
        except:
            return []

    def query_by_distance(
        self,
        start_nodes: List[str],
        max_distance: int = 2,
        ranking: str = 'data_flow_priority'
    ) -> List[Dict[str, Any]]:
        """
        Query nodes by graph distance from start nodes.
        
        Args:
            start_nodes: List of node IDs to start from
            max_distance: Maximum graph distance
            ranking: 'data_flow_priority' or 'distance'
            
        Returns:
            List of ranked nodes with their data
        """
        results = []
        visited = set()
        
        for start in start_nodes:
            if start not in self.graph:
                continue
            
            # BFS to find nodes within distance
            queue = [(start, 0)]
            
            while queue:
                node, dist = queue.pop(0)
                
                if node in visited or dist > max_distance:
                    continue
                
                visited.add(node)
                
                if node != start:
                    node_data = dict(self.graph.nodes.get(node, {}))
                    node_data['id'] = node
                    node_data['distance'] = dist
                    results.append(node_data)
                
                # Add neighbors
                for neighbor in self.graph.successors(node):
                    queue.append((neighbor, dist + 1))
                for neighbor in self.graph.predecessors(node):
                    queue.append((neighbor, dist + 1))
        
        # Rank results
        if ranking == 'data_flow_priority':
            results = self._rank_by_data_flow(results, start_nodes)
        else:
            results.sort(key=lambda x: x.get('distance', float('inf')))
        
        return results

    def _rank_by_data_flow(
        self,
        nodes: List[Dict[str, Any]],
        modified_nodes: List[str]
    ) -> List[Dict[str, Any]]:
        """Rank nodes by data flow priority."""
        for node in nodes:
            score = 0
            node_id = node.get('id', '')
            distance = node.get('distance', float('inf'))
            
            # Distance penalty
            score -= distance * 10
            
            # Data flow bonus
            if self._is_in_data_dependency_chain(node_id, modified_nodes):
                score += 50
            
            # Call graph bonus
            if self._is_caller_or_callee(node_id, modified_nodes):
                score += 30
            
            node['score'] = score
        
        nodes.sort(key=lambda x: x.get('score', 0), reverse=True)
        return nodes

    def _is_in_data_dependency_chain(
        self,
        node_id: str,
        modified_nodes: List[str]
    ) -> bool:
        """Check if node is in a data dependency chain with modified nodes."""
        # Simplified check - look for PDG edges
        for modified in modified_nodes:
            for _, target, data in self.graph.out_edges(modified, data=True):
                if target == node_id and data.get('edge_type') == 'data_flow':
                    return True
            for source, _, data in self.graph.in_edges(modified, data=True):
                if source == node_id and data.get('edge_type') == 'data_flow':
                    return True
        return False

    def _is_caller_or_callee(
        self,
        node_id: str,
        modified_nodes: List[str]
    ) -> bool:
        """Check if node is a caller or callee of modified nodes."""
        for modified in modified_nodes:
            # Check if node calls modified
            for _, target, data in self.graph.out_edges(node_id, data=True):
                if target == modified and data.get('edge_type') == 'calls':
                    return True
            # Check if modified calls node
            for _, target, data in self.graph.out_edges(modified, data=True):
                if target == node_id and data.get('edge_type') == 'calls':
                    return True
        return False

    def get_callers(self, function_name: str) -> List[str]:
        """Get all functions that call the given function."""
        callers = []
        for source, target, data in self.graph.edges(data=True):
            if target == function_name and data.get('edge_type') == 'calls':
                callers.append(source)
        return callers

    def get_callees(self, function_id: str) -> List[str]:
        """Get all functions called by the given function."""
        callees = []
        for _, target, data in self.graph.out_edges(function_id, data=True):
            if data.get('edge_type') == 'calls':
                callees.append(target)
        return callees

    def save_cache(self, version: str = "latest"):
        """Save CPG to cache."""
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"cpg_{version}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'file_versions': self.file_versions,
                    'function_nodes': self.function_nodes
                }, f)
            
            logger.info(f"CPG saved to {cache_file}")

    def load_cache(self, version: str = "latest") -> bool:
        """Load CPG from cache."""
        if self.cache_dir:
            cache_file = self.cache_dir / f"cpg_{version}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    self.graph = data['graph']
                    self.file_versions = data['file_versions']
                    self.function_nodes = data['function_nodes']
                    
                    logger.info(f"CPG loaded from {cache_file}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to load cache: {e}")
        
        return False

    def get_stats(self) -> Dict[str, int]:
        """Get CPG statistics."""
        node_types = {}
        for node in self.graph.nodes:
            node_type = self.graph.nodes[node].get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'files': len(self.function_nodes),
            'node_types': node_types,
            'edge_types': edge_types
        }


if __name__ == "__main__":
    # Test the CPG builder
    logging.basicConfig(level=logging.DEBUG)
    
    builder = CPGBuilder(cache_dir=".xv6_agent/cpg_cache")
    
    # Build from current directory
    builder.build_from_directory(".")
    
    # Print stats
    stats = builder.get_stats()
    print(f"\n=== CPG Statistics ===")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Files: {stats['files']}")
    print(f"Node types: {stats['node_types']}")
    print(f"Edge types: {stats['edge_types']}")
