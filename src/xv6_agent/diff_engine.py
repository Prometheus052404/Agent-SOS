"""
Diff Engine Module - Two-level change computation for Xv6 agent.

Implements:
- Line delta computation using unified diff
- Semantic delta computation using AST comparison
- Integration with CPG updates
"""

import re
import difflib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LineDelta:
    """Represents line-level changes."""
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    context_lines: int = 3
    unified_diff: str = ""


@dataclass
class SemanticDelta:
    """Represents AST-level changes."""
    functions_added: List[str] = field(default_factory=list)
    functions_modified: List[Dict[str, Any]] = field(default_factory=list)
    functions_deleted: List[str] = field(default_factory=list)
    global_vars_changed: List[str] = field(default_factory=list)
    structs_modified: List[str] = field(default_factory=list)
    includes_changed: List[str] = field(default_factory=list)


@dataclass
class FileDiff:
    """Complete diff for a single file."""
    file: str
    timestamp: str
    line_delta: LineDelta
    semantic_delta: SemanticDelta
    affected_callsites: List[str] = field(default_factory=list)


class DiffEngine:
    """
    Computes line-level and semantic-level diffs between file versions.
    """

    def __init__(self, context_lines: int = 3):
        self.context_lines = context_lines
        
        # Regex patterns for C parsing
        self.function_pattern = re.compile(
            r'^(?:static\s+)?(?:inline\s+)?'
            r'(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*{'
        )
        self.struct_pattern = re.compile(r'struct\s+(\w+)\s*{')
        self.global_var_pattern = re.compile(
            r'^(?:static\s+)?(?:const\s+)?(?:\w+)\s+(\w+)\s*[=;]'
        )
        self.include_pattern = re.compile(r'#include\s*[<"]([^>"]+)[>"]')

    def compute_diff(
        self,
        old_content: Optional[str],
        new_content: str,
        filepath: str
    ) -> FileDiff:
        """
        Compute both line-level and semantic-level diff.
        
        Args:
            old_content: Previous file content (or None for new files)
            new_content: Current file content
            filepath: Path to the file
            
        Returns:
            FileDiff with both line and semantic deltas
        """
        timestamp = datetime.now().isoformat()
        
        # Handle new file case
        if old_content is None:
            old_content = ""
        
        # Compute line delta
        line_delta = self._compute_line_delta(old_content, new_content, filepath)
        
        # Compute semantic delta
        semantic_delta = self._compute_semantic_delta(old_content, new_content)
        
        # Find affected callsites (functions that call modified functions)
        affected_callsites = self._find_affected_callsites(
            semantic_delta.functions_modified,
            new_content
        )
        
        return FileDiff(
            file=filepath,
            timestamp=timestamp,
            line_delta=line_delta,
            semantic_delta=semantic_delta,
            affected_callsites=affected_callsites
        )

    def _compute_line_delta(
        self,
        old_content: str,
        new_content: str,
        filepath: str
    ) -> LineDelta:
        """Compute line-level diff using unified diff."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        # Generate unified diff
        diff = list(difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"{filepath} (old)",
            tofile=f"{filepath} (new)",
            n=self.context_lines
        ))
        
        # Extract added and removed lines
        added = []
        removed = []
        
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                added.append(line[1:].rstrip())
            elif line.startswith('-') and not line.startswith('---'):
                removed.append(line[1:].rstrip())
        
        return LineDelta(
            added=added,
            removed=removed,
            context_lines=self.context_lines,
            unified_diff=''.join(diff)
        )

    def _compute_semantic_delta(
        self,
        old_content: str,
        new_content: str
    ) -> SemanticDelta:
        """Compute AST-level semantic changes."""
        # Extract components from both versions
        old_funcs = self._extract_functions(old_content)
        new_funcs = self._extract_functions(new_content)
        
        old_structs = self._extract_structs(old_content)
        new_structs = self._extract_structs(new_content)
        
        old_globals = self._extract_globals(old_content)
        new_globals = self._extract_globals(new_content)
        
        old_includes = self._extract_includes(old_content)
        new_includes = self._extract_includes(new_content)
        
        # Compare functions
        old_func_names = set(old_funcs.keys())
        new_func_names = set(new_funcs.keys())
        
        functions_added = list(new_func_names - old_func_names)
        functions_deleted = list(old_func_names - new_func_names)
        
        # Check for modified functions
        functions_modified = []
        for name in old_func_names & new_func_names:
            if old_funcs[name] != new_funcs[name]:
                # Count changed lines
                old_lines = old_funcs[name].splitlines()
                new_lines = new_funcs[name].splitlines()
                matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
                ratio = matcher.ratio()
                
                functions_modified.append({
                    'name': name,
                    'lines_changed': abs(len(new_lines) - len(old_lines)),
                    'similarity': ratio,
                    'impact': 'high' if ratio < 0.7 else 'medium' if ratio < 0.9 else 'low'
                })
        
        # Compare structs
        structs_modified = list(
            (old_structs.keys() | new_structs.keys()) -
            (old_structs.keys() & new_structs.keys())
        )
        for name in old_structs.keys() & new_structs.keys():
            if old_structs[name] != new_structs[name]:
                structs_modified.append(name)
        
        # Compare globals
        global_vars_changed = list(
            (set(old_globals) ^ set(new_globals)) |
            {g for g in (set(old_globals) & set(new_globals))
             if old_content.count(g) != new_content.count(g)}
        )
        
        # Compare includes
        includes_changed = list(set(old_includes) ^ set(new_includes))
        
        return SemanticDelta(
            functions_added=functions_added,
            functions_modified=functions_modified,
            functions_deleted=functions_deleted,
            global_vars_changed=global_vars_changed,
            structs_modified=structs_modified,
            includes_changed=includes_changed
        )

    def _extract_functions(self, content: str) -> Dict[str, str]:
        """Extract function definitions and their bodies."""
        functions = {}
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            match = self.function_pattern.match(line.strip())
            
            if match:
                func_name = match.group(1)
                
                # Find the end of the function (matching braces)
                brace_count = line.count('{') - line.count('}')
                func_lines = [line]
                j = i + 1
                
                while j < len(lines) and brace_count > 0:
                    func_lines.append(lines[j])
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1
                
                functions[func_name] = '\n'.join(func_lines)
                i = j
            else:
                i += 1
        
        return functions

    def _extract_structs(self, content: str) -> Dict[str, str]:
        """Extract struct definitions."""
        structs = {}
        
        for match in self.struct_pattern.finditer(content):
            struct_name = match.group(1)
            start = match.start()
            
            # Find matching closing brace
            brace_count = 0
            end = start
            for i, char in enumerate(content[start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = start + i + 1
                        break
            
            structs[struct_name] = content[start:end]
        
        return structs

    def _extract_globals(self, content: str) -> List[str]:
        """Extract global variable names."""
        globals_list = []
        
        for line in content.split('\n'):
            line = line.strip()
            # Skip if inside a function (simple heuristic)
            if line.startswith('//') or line.startswith('/*'):
                continue
            
            match = self.global_var_pattern.match(line)
            if match:
                globals_list.append(match.group(1))
        
        return globals_list

    def _extract_includes(self, content: str) -> List[str]:
        """Extract #include statements."""
        return self.include_pattern.findall(content)

    def _find_affected_callsites(
        self,
        modified_functions: List[Dict[str, Any]],
        content: str
    ) -> List[str]:
        """Find functions that call the modified functions."""
        affected = set()
        
        for func_info in modified_functions:
            func_name = func_info['name']
            # Simple pattern to find calls
            call_pattern = re.compile(rf'\b{func_name}\s*\(')
            
            # Find which functions contain calls to this function
            current_func = None
            brace_count = 0
            
            for line in content.split('\n'):
                # Track current function
                func_match = self.function_pattern.match(line.strip())
                if func_match:
                    current_func = func_match.group(1)
                    brace_count = line.count('{') - line.count('}')
                elif current_func:
                    brace_count += line.count('{') - line.count('}')
                    if brace_count <= 0:
                        current_func = None
                
                # Check for calls
                if current_func and current_func != func_name:
                    if call_pattern.search(line):
                        affected.add(current_func)
        
        return list(affected)

    def to_json(self, diff: FileDiff) -> Dict[str, Any]:
        """Convert FileDiff to JSON-serializable dict."""
        return {
            'file': diff.file,
            'timestamp': diff.timestamp,
            'line_delta': {
                'added': diff.line_delta.added,
                'removed': diff.line_delta.removed,
                'context_lines': diff.line_delta.context_lines
            },
            'semantic_delta': {
                'functions_added': diff.semantic_delta.functions_added,
                'functions_modified': diff.semantic_delta.functions_modified,
                'functions_deleted': diff.semantic_delta.functions_deleted,
                'global_vars_changed': diff.semantic_delta.global_vars_changed,
                'structs_modified': diff.semantic_delta.structs_modified,
                'includes_changed': diff.semantic_delta.includes_changed
            },
            'affected_callsites': diff.affected_callsites
        }


# Convenience function
def compute_file_diff(
    old_content: Optional[str],
    new_content: str,
    filepath: str
) -> FileDiff:
    """Compute diff between two file versions."""
    engine = DiffEngine()
    return engine.compute_diff(old_content, new_content, filepath)


if __name__ == "__main__":
    # Test the diff engine
    old_code = '''
#include "types.h"

void foo(int x) {
    int y = x + 1;
    return;
}

void bar() {
    foo(5);
}
'''

    new_code = '''
#include "types.h"
#include "defs.h"

void foo(int x, int z) {
    int y = x + z;
    printf("result: %d", y);
    return;
}

void bar() {
    foo(5, 10);
}

void baz() {
    bar();
}
'''

    diff = compute_file_diff(old_code, new_code, "test.c")
    
    print("=== Line Delta ===")
    print(f"Added: {len(diff.line_delta.added)} lines")
    print(f"Removed: {len(diff.line_delta.removed)} lines")
    
    print("\n=== Semantic Delta ===")
    print(f"Functions added: {diff.semantic_delta.functions_added}")
    print(f"Functions modified: {diff.semantic_delta.functions_modified}")
    print(f"Includes changed: {diff.semantic_delta.includes_changed}")
    
    print(f"\nAffected callsites: {diff.affected_callsites}")
