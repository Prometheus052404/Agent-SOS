"""
Context Assembler Module - Prompt compilation for Xv6 agent.

Implements:
- 3000 token budget enforcement
- Priority-ranked assembly (system, errors, code, textbook)
- CPG distance-ranked code snippets
- Vector search for textbook chunks
- User consent flow before first API call
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContextPart:
    """A part of the assembled context."""
    name: str
    content: str
    tokens: int
    priority: int


class TokenEstimator:
    """Estimates token count for text."""

    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token

    def estimate(self, text: str) -> int:
        """Estimate token count for text."""
        return max(1, int(len(text) / self.chars_per_token))


class ContextAssembler:
    """
    Assembles context for LLM prompts with token budget.
    
    Token allocation (3000 total):
    - System prompt: 500 tokens
    - Error context: 700 tokens
    - Code context: 1200 tokens
    - Textbook chunks: 600 tokens
    """

    def __init__(
        self,
        max_tokens: int = 3000,
        token_allocation: Optional[Dict[str, int]] = None
    ):
        self.max_tokens = max_tokens
        self.token_allocation = token_allocation or {
            'system_prompt': 500,
            'error_context': 700,
            'code_context': 1200,
            'textbook_chunks': 600
        }
        self.token_estimator = TokenEstimator()

    def assemble_context(
        self,
        user_query: str,
        session_state: Dict[str, Any],
        code_snippets: List[Dict[str, Any]] = None,
        textbook_chunks: List[str] = None,
        cpg_results: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assemble context for LLM prompt.
        
        Args:
            user_query: The user's question
            session_state: Current session state
            code_snippets: Relevant code snippets
            textbook_chunks: Relevant textbook content
            cpg_results: Results from CPG query
            
        Returns:
            Assembled context with prompt and metadata
        """
        context_parts = []
        
        # 1. System prompt (mandatory)
        system_prompt = self._build_system_prompt(session_state)
        context_parts.append(ContextPart(
            name='system',
            content=system_prompt,
            tokens=self.token_estimator.estimate(system_prompt),
            priority=1
        ))
        
        # 2. Error context (high priority)
        error_context = self._build_error_context(session_state)
        if error_context:
            context_parts.append(ContextPart(
                name='errors',
                content=error_context,
                tokens=self.token_estimator.estimate(error_context),
                priority=2
            ))
        
        # 3. Code context (CPG ranked)
        code_context = self._build_code_context(
            code_snippets or [],
            cpg_results or [],
            session_state
        )
        if code_context:
            context_parts.append(ContextPart(
                name='code',
                content=code_context,
                tokens=self.token_estimator.estimate(code_context),
                priority=3
            ))
        
        # 4. Textbook chunks
        textbook_context = self._build_textbook_context(textbook_chunks or [])
        if textbook_context:
            context_parts.append(ContextPart(
                name='textbook',
                content=textbook_context,
                tokens=self.token_estimator.estimate(textbook_context),
                priority=4
            ))
        
        # 5. User query
        query_tokens = self.token_estimator.estimate(user_query)
        context_parts.append(ContextPart(
            name='query',
            content=user_query,
            tokens=query_tokens,
            priority=5
        ))
        
        # Enforce budget
        final_parts = self._enforce_budget(context_parts)
        
        # Build final prompt
        prompt = self._build_prompt(final_parts, user_query)
        
        # Calculate total tokens
        total_tokens = sum(p.tokens for p in final_parts)
        
        return {
            'prompt': prompt,
            'total_tokens': total_tokens,
            'parts': {p.name: p.tokens for p in final_parts},
            'within_budget': total_tokens <= self.max_tokens
        }

    def _build_system_prompt(self, session_state: Dict[str, Any]) -> str:
        """Build the system prompt."""
        task_id = session_state.get('task_id', 'unknown')
        current_state = session_state.get('current_state', 'NOT_STARTED')
        progress = session_state.get('progress', 0.0)
        
        return f"""You are an Xv6 Teaching Assistant.

Student Progress: {progress*100:.0f}% through {task_id}
Current Subtask: {current_state}

Your role:
- Guide the student to discover solutions, don't give answers
- Explain underlying OS principles first
- Reference xv6 book chapters when relevant
- Never show more than 5 lines of code
- Use questions to prompt understanding

Constraints:
- Explain WHY before HOW
- Point to relevant book sections
- Guide discovery, don't solve"""

    def _build_error_context(self, session_state: Dict[str, Any]) -> str:
        """Build error context from build results."""
        last_build = session_state.get('last_build', {})
        
        if not last_build:
            return ""
        
        parts = []
        
        # Compiler errors
        errors = last_build.get('errors', [])
        if errors:
            parts.append("COMPILER ERRORS:")
            for i, error in enumerate(errors[:3]):  # Top 3 errors
                parts.append(
                    f"  {error.get('file', '?')}:{error.get('line', '?')}: "
                    f"{error.get('message', 'unknown error')}"
                )
        
        # Panics
        panics = last_build.get('panics', [])
        if panics:
            parts.append("\nKERNEL PANICS:")
            for panic in panics[:2]:  # Top 2 panics
                parts.append(f"  panic: {panic.get('message', 'unknown')}")
                if panic.get('epc'):
                    parts.append(f"  epc: {panic['epc']}")
                if panic.get('backtrace'):
                    parts.append(f"  backtrace: {' -> '.join(panic['backtrace'][:5])}")
        
        return '\n'.join(parts)

    def _build_code_context(
        self,
        code_snippets: List[Dict[str, Any]],
        cpg_results: List[Dict[str, Any]],
        session_state: Dict[str, Any]
    ) -> str:
        """Build code context from snippets and CPG results."""
        parts = []
        token_budget = self.token_allocation['code_context']
        tokens_used = 0
        
        # Add modified files info
        diff_info = session_state.get('diff_engine', {})
        files_changed = diff_info.get('files_changed', [])
        
        if files_changed:
            parts.append(f"MODIFIED FILES: {', '.join(files_changed)}")
            tokens_used += self.token_estimator.estimate(parts[-1])
        
        # Add code snippets (ranked by score/distance)
        all_snippets = []
        
        for snippet in code_snippets:
            all_snippets.append({
                'content': snippet.get('content', ''),
                'file': snippet.get('file', ''),
                'score': snippet.get('score', 0)
            })
        
        for result in cpg_results:
            if result.get('source_span'):
                all_snippets.append({
                    'content': result['source_span'],
                    'file': result.get('file', ''),
                    'score': result.get('score', 0)
                })
        
        # Sort by score (higher is better)
        all_snippets.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Add snippets within budget
        parts.append("\nCODE CONTEXT:")
        
        for snippet in all_snippets:
            content = snippet['content']
            snippet_tokens = self.token_estimator.estimate(content)
            
            if tokens_used + snippet_tokens > token_budget:
                break
            
            file_name = Path(snippet['file']).name if snippet['file'] else 'unknown'
            parts.append(f"\n// {file_name}")
            parts.append(content)
            tokens_used += snippet_tokens
        
        return '\n'.join(parts)

    def _build_textbook_context(self, chunks: List[str]) -> str:
        """Build textbook context from retrieved chunks."""
        if not chunks:
            return ""
        
        parts = ["RELEVANT TEXTBOOK SECTIONS:"]
        token_budget = self.token_allocation['textbook_chunks']
        tokens_used = 0
        
        for chunk in chunks:
            chunk_tokens = self.token_estimator.estimate(chunk)
            
            if tokens_used + chunk_tokens > token_budget:
                break
            
            parts.append(f"\n{chunk}")
            tokens_used += chunk_tokens
        
        return '\n'.join(parts)

    def _enforce_budget(
        self,
        parts: List[ContextPart]
    ) -> List[ContextPart]:
        """Enforce token budget by trimming lower priority parts."""
        # Sort by priority (lower number = higher priority)
        parts.sort(key=lambda p: p.priority)
        
        total_tokens = sum(p.tokens for p in parts)
        
        if total_tokens <= self.max_tokens:
            return parts
        
        # Trim from lowest priority
        result = []
        tokens_remaining = self.max_tokens
        
        for part in parts:
            if part.tokens <= tokens_remaining:
                result.append(part)
                tokens_remaining -= part.tokens
            else:
                # Truncate this part
                ratio = tokens_remaining / part.tokens
                truncated_content = part.content[:int(len(part.content) * ratio)]
                
                if truncated_content:
                    result.append(ContextPart(
                        name=part.name,
                        content=truncated_content + "\n[truncated]",
                        tokens=tokens_remaining,
                        priority=part.priority
                    ))
                
                break
        
        return result

    def _build_prompt(
        self,
        parts: List[ContextPart],
        user_query: str
    ) -> str:
        """Build the final prompt string."""
        sections = []
        
        for part in parts:
            if part.name == 'system':
                sections.append(f"[SYSTEM]\n{part.content}")
            elif part.name == 'errors':
                sections.append(f"[ERRORS]\n{part.content}")
            elif part.name == 'code':
                sections.append(f"[CODE]\n{part.content}")
            elif part.name == 'textbook':
                sections.append(f"[KNOWLEDGE]\n{part.content}")
        
        sections.append(f"[USER QUERY]\n{user_query}")
        
        return '\n\n'.join(sections)


class ConsentManager:
    """Manages user consent for LLM API calls."""

    def __init__(self, consent_file: str = ".xv6_agent/consent.json"):
        self.consent_file = Path(consent_file)
        self.consent_given = self._load_consent()

    def _load_consent(self) -> bool:
        """Load consent status from file."""
        if self.consent_file.exists():
            try:
                import json
                with open(self.consent_file, 'r') as f:
                    data = json.load(f)
                    return data.get('consent', False)
            except:
                pass
        return False

    def _save_consent(self, consent: bool):
        """Save consent status to file."""
        import json
        self.consent_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.consent_file, 'w') as f:
            json.dump({'consent': consent}, f)

    def request_consent(self, code_preview: str) -> bool:
        """
        Request user consent before first API call.
        
        Args:
            code_preview: Preview of code that will be sent
            
        Returns:
            True if consent given
        """
        if self.consent_given:
            return True
        
        print("\n" + "=" * 60)
        print("⚠️  ABOUT TO SEND CODE TO LLM API")
        print("=" * 60)
        print("\nThe following code snippet will be sent:")
        print("-" * 60)
        print(code_preview[:500] + "..." if len(code_preview) > 500 else code_preview)
        print("-" * 60)
        
        try:
            response = input("\nProceed with API call? [Y/n]: ").strip().lower()
            consent = response in ('', 'y', 'yes')
            
            if consent:
                self.consent_given = True
                self._save_consent(True)
                print("✓ Consent recorded. Future calls won't prompt.")
            
            return consent
            
        except (EOFError, KeyboardInterrupt):
            return False

    def revoke_consent(self):
        """Revoke previously given consent."""
        self.consent_given = False
        self._save_consent(False)


def create_context_assembler(max_tokens: int = 3000) -> ContextAssembler:
    """Create a context assembler instance."""
    return ContextAssembler(max_tokens=max_tokens)


if __name__ == "__main__":
    # Test the context assembler
    logging.basicConfig(level=logging.DEBUG)
    
    assembler = create_context_assembler()
    
    # Test assembly
    session_state = {
        'task_id': 'lab_locks',
        'current_state': 'IMPLEMENTING_ACQUIRE',
        'progress': 0.3,
        'last_build': {
            'errors': [],
            'panics': [{'message': 'sched locks', 'epc': '0x80001234'}]
        },
        'diff_engine': {
            'files_changed': ['proc.c', 'spinlock.c']
        }
    }
    
    code_snippets = [
        {
            'file': 'spinlock.c',
            'content': 'void acquire(struct spinlock *lk) {\n  // ...\n}',
            'score': 0.9
        }
    ]
    
    textbook_chunks = [
        "[xv6-book Ch.4.2] Spinlocks provide mutual exclusion by spinning..."
    ]
    
    result = assembler.assemble_context(
        user_query="Why is it panicking with 'sched locks'?",
        session_state=session_state,
        code_snippets=code_snippets,
        textbook_chunks=textbook_chunks
    )
    
    print("\n=== Assembled Context ===")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Within budget: {result['within_budget']}")
    print(f"\nParts: {result['parts']}")
    print(f"\n--- Prompt ---\n{result['prompt'][:1000]}...")
