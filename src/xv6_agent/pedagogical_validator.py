"""
Pedagogical Validator Module - Response safety for Xv6 agent.

Implements:
- Code reveal check (<5 lines)
- Progressive disclosure check
- Principle/fix ratio check (≥2.0)
- Invariant check with dangerous patterns
- Confidence scoring via meta-prompt
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of pedagogical validation."""
    passed: bool
    checks: Dict[str, bool]
    warnings: List[str]
    blocked_reason: Optional[str] = None


@dataclass
class ConfidenceScore:
    """Confidence score for a response."""
    value: float
    reasoning: str


class PedagogicalValidator:
    """
    Validates LLM responses for pedagogical safety.
    
    4-layer validation:
    1. Code Reveal Check: Reject >5 lines of C code
    2. Progressive Disclosure: Check for forbidden phrases
    3. Principle/Fix Ratio: Require ≥2.0 principles per fix
    4. Invariant Check: Warn on dangerous patterns
    """

    def __init__(
        self,
        max_code_lines: int = 5,
        min_principle_ratio: float = 2.0,
        forbidden_phrases: Optional[List[str]] = None,
        invariant_patterns: Optional[List[Dict[str, str]]] = None
    ):
        self.max_code_lines = max_code_lines
        self.min_principle_ratio = min_principle_ratio
        
        self.forbidden_phrases = forbidden_phrases or [
            "the answer is",
            "exactly",
            "specifically do this",
            "change line",
            "add this code",
            "replace with",
            "copy this",
            "here's the fix",
            "here's the solution"
        ]
        
        self.invariant_patterns = invariant_patterns or [
            {
                'regex': r'sleep.*holding.*lock',
                'warning': 'Never sleep while holding spinlock'
            },
            {
                'regex': r'acquire.*acquire',
                'warning': 'Potential double-acquire deadlock'
            },
            {
                'regex': r'release.*while.*not.*held',
                'warning': 'Releasing unheld lock'
            }
        ]
        
        # Principle markers (explanatory)
        self.principle_markers = [
            "because",
            "the reason",
            "consider what happens",
            "the kernel needs",
            "this is important",
            "the invariant",
            "understanding",
            "the principle",
            "fundamentally",
            "the key insight"
        ]
        
        # Fix markers (solution-giving)
        self.fix_markers = [
            "change line",
            "add this code",
            "replace with",
            "modify to",
            "update the",
            "add the following",
            "insert",
            "remove and add"
        ]

    def validate(self, response: str) -> ValidationResult:
        """
        Run all validation checks on a response.
        
        Args:
            response: The LLM response to validate
            
        Returns:
            ValidationResult with pass/fail status
        """
        checks = {}
        warnings = []
        blocked_reason = None
        
        # Check 1: Code reveal
        code_check, code_lines = self._check_code_reveal(response)
        checks['code_reveal'] = code_check
        if not code_check:
            blocked_reason = f"Response contains {code_lines} lines of code (max: {self.max_code_lines})"
        
        # Check 2: Progressive disclosure
        disclosure_check, forbidden = self._check_progressive_disclosure(response)
        checks['progressive_disclosure'] = disclosure_check
        if not disclosure_check:
            warnings.append(f"Contains forbidden phrase: '{forbidden}'")
        
        # Check 3: Principle/fix ratio
        ratio_check, ratio = self._check_principle_ratio(response)
        checks['principle_ratio'] = ratio_check
        if not ratio_check:
            warnings.append(f"Principle/fix ratio {ratio:.1f} < {self.min_principle_ratio}")
        
        # Check 4: Invariant check
        invariant_check, inv_warnings = self._check_invariants(response)
        checks['invariant_check'] = invariant_check
        warnings.extend(inv_warnings)
        
        # Overall pass
        passed = checks['code_reveal']  # Only code reveal blocks
        
        return ValidationResult(
            passed=passed,
            checks=checks,
            warnings=warnings,
            blocked_reason=blocked_reason
        )

    def _check_code_reveal(self, response: str) -> Tuple[bool, int]:
        """
        Check if response contains too many lines of code.
        
        Returns:
            (passed, line_count)
        """
        # Find code blocks
        code_block_pattern = re.compile(r'```(?:c|C)?\n(.*?)```', re.DOTALL)
        
        total_code_lines = 0
        
        for match in code_block_pattern.finditer(response):
            code = match.group(1)
            # Count non-empty lines
            lines = [l for l in code.split('\n') if l.strip()]
            total_code_lines += len(lines)
        
        # Also check for indented code (4 spaces or tab)
        indented_pattern = re.compile(r'^(?:    |\t).+$', re.MULTILINE)
        indented_lines = len(indented_pattern.findall(response))
        
        # Only count indented lines if there are no code blocks
        if total_code_lines == 0:
            total_code_lines = indented_lines
        
        passed = total_code_lines <= self.max_code_lines
        
        return passed, total_code_lines

    def _check_progressive_disclosure(self, response: str) -> Tuple[bool, Optional[str]]:
        """
        Check for forbidden phrases that give away solutions.
        
        Returns:
            (passed, forbidden_phrase_found)
        """
        response_lower = response.lower()
        
        for phrase in self.forbidden_phrases:
            if phrase.lower() in response_lower:
                return False, phrase
        
        return True, None

    def _check_principle_ratio(self, response: str) -> Tuple[bool, float]:
        """
        Check ratio of explanatory to solution-giving language.
        
        Returns:
            (passed, ratio)
        """
        response_lower = response.lower()
        
        principle_count = sum(
            1 for marker in self.principle_markers
            if marker.lower() in response_lower
        )
        
        fix_count = sum(
            1 for marker in self.fix_markers
            if marker.lower() in response_lower
        )
        
        # Avoid division by zero
        if fix_count == 0:
            ratio = float('inf') if principle_count > 0 else 1.0
        else:
            ratio = principle_count / fix_count
        
        passed = ratio >= self.min_principle_ratio or fix_count == 0
        
        return passed, ratio

    def _check_invariants(self, response: str) -> Tuple[bool, List[str]]:
        """
        Check for dangerous pattern warnings.
        
        Returns:
            (passed, warnings)
        """
        warnings = []
        response_lower = response.lower()
        
        for pattern in self.invariant_patterns:
            regex = pattern['regex']
            if re.search(regex, response_lower, re.IGNORECASE):
                warnings.append(f"⚠️ {pattern['warning']}")
        
        # Invariant check always passes, just adds warnings
        return True, warnings

    def sanitize_response(self, response: str) -> str:
        """
        Sanitize a response by removing problematic content.
        
        Args:
            response: The original response
            
        Returns:
            Sanitized response
        """
        result = response
        
        # Remove large code blocks
        code_pattern = re.compile(r'```(?:c|C)?\n(.*?)```', re.DOTALL)
        
        for match in code_pattern.finditer(response):
            code = match.group(1)
            lines = code.split('\n')
            
            if len(lines) > self.max_code_lines:
                # Replace with truncated version
                truncated = '\n'.join(lines[:self.max_code_lines])
                truncated += f'\n// [... {len(lines) - self.max_code_lines} more lines truncated ...]'
                result = result.replace(match.group(0), f'```c\n{truncated}\n```')
        
        return result


class ConfidenceScorer:
    """Scores confidence using meta-prompt pattern."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def score(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> ConfidenceScore:
        """
        Score confidence of a response.
        
        If LLM is available, uses meta-prompt.
        Otherwise, uses heuristic scoring.
        """
        if self.llm_client and self.llm_client.is_available():
            return self._score_with_llm(response, context)
        else:
            return self._score_heuristic(response, context)

    def _score_with_llm(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> ConfidenceScore:
        """Score using LLM meta-prompt."""
        meta_prompt = f"""
Rate your confidence (0.0-1.0) that the following explanation:
1. Correctly identifies the root cause
2. Guides the student without revealing the answer
3. Is appropriate for their current progress level

Context:
- Current task: {context.get('task_id', 'unknown')} / {context.get('current_state', 'unknown')}
- Recent errors: {context.get('last_build', {}).get('errors', [])[:1]}
- Recent panics: {context.get('last_build', {}).get('panics', [])[:1]}

Explanation to rate:
{response[:500]}

Output ONLY a JSON object: {{"confidence": 0.0-1.0, "reasoning": "..."}}
"""
        
        try:
            result = self.llm_client.generate(
                prompt=meta_prompt,
                system_prompt="You are an evaluation system. Output only JSON."
            )
            
            # Parse JSON response
            import json
            data = json.loads(result.content)
            
            return ConfidenceScore(
                value=float(data.get('confidence', 0.5)),
                reasoning=data.get('reasoning', 'No reasoning provided')
            )
            
        except Exception as e:
            logger.warning(f"Meta-prompt scoring failed: {e}")
            return self._score_heuristic(response, context)

    def _score_heuristic(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> ConfidenceScore:
        """Score using heuristics."""
        score = 0.5  # Base score
        reasons = []
        
        response_lower = response.lower()
        
        # Positive signals
        if 'because' in response_lower:
            score += 0.1
            reasons.append("Includes explanation")
        
        if 'chapter' in response_lower or 'book' in response_lower:
            score += 0.1
            reasons.append("References textbook")
        
        if 'consider' in response_lower or 'think about' in response_lower:
            score += 0.1
            reasons.append("Prompts reflection")
        
        # Negative signals
        if '```' in response:
            score -= 0.1
            reasons.append("Contains code blocks")
        
        if any(p in response_lower for p in ['fix:', 'solution:', 'answer:']):
            score -= 0.2
            reasons.append("May reveal solution")
        
        # Clamp to valid range
        score = max(0.0, min(1.0, score))
        
        return ConfidenceScore(
            value=score,
            reasoning='; '.join(reasons) if reasons else 'Heuristic evaluation'
        )


def create_validator() -> PedagogicalValidator:
    """Create a pedagogical validator instance."""
    return PedagogicalValidator()


if __name__ == "__main__":
    # Test the validator
    logging.basicConfig(level=logging.DEBUG)
    
    validator = create_validator()
    
    # Test cases
    test_responses = [
        # Good response
        """
        The panic occurs because sleep() is being called while holding 
        a spinlock. The kernel's invariant requires that sleep() only be 
        called without spinlocks held (except the condition lock).
        
        Consider: What happens to the CPU when sleep() is called? 
        What if another process needs that spinlock?
        
        Check Chapter 4.2 for the sleep/wakeup pattern.
        """,
        
        # Bad response (too much code)
        """
        Here's the fix:
        ```c
        void acquire(struct spinlock *lk) {
            pushcli();
            while(xchg(&lk->locked, 1) != 0)
                ;
            __sync_synchronize();
            lk->cpu = mycpu();
        }
        
        void release(struct spinlock *lk) {
            lk->cpu = 0;
            __sync_synchronize();
            xchg(&lk->locked, 0);
            popcli();
        }
        ```
        """,
        
        # Borderline response
        """
        The answer is in how sleep() expects locks to be handled.
        You need to release the lock before yielding the CPU.
        """
    ]
    
    for i, response in enumerate(test_responses):
        print(f"\n=== Test Case {i + 1} ===")
        result = validator.validate(response)
        print(f"Passed: {result.passed}")
        print(f"Checks: {result.checks}")
        print(f"Warnings: {result.warnings}")
        if result.blocked_reason:
            print(f"Blocked: {result.blocked_reason}")
