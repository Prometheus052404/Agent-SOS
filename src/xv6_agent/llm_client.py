"""
LLM Client Module - LLM integration for Xv6 agent.

Implements:
- OpenAI GPT-4o client
- Anthropic Claude 3.5 Sonnet support
- Groq support for fast inference
- Fallback template system
- Response parsing with retry logic
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import LLM clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    success: bool
    error: Optional[str] = None


class FallbackTemplates:
    """Template-based fallback responses."""

    def __init__(self, template_dir: str = ".xv6_agent/templates"):
        self.template_dir = Path(template_dir)
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load templates from disk."""
        templates = {
            # Default templates
            'panic_sched_locks': """
The "sched locks" panic occurs when the scheduler is called while 
locks are still held. This violates the kernel's invariant that 
sleep() and sched() require specific lock states.

Consider:
1. What locks are held when you call sleep()?
2. Does sleep() expect the caller to hold locks?
3. Check Chapter 4.2 of the xv6 book for the sleep/wakeup pattern.

The key insight is understanding when to release and reacquire locks.
""",
            'panic_acquire': """
The "acquire" panic typically means you're trying to acquire a lock 
that's already held, or there's a lock ordering violation.

Consider:
1. Is the same lock being acquired twice?
2. Are you following consistent lock ordering?
3. Check if any interrupt handlers acquire this lock.

Review Chapter 4 on locking to understand the acquire/release pattern.
""",
            'general_help': """
Without more context, here are some general debugging tips:

1. Check the panic message and backtrace carefully
2. Look at what function caused the panic
3. Review recent code changes near that function
4. Check the xv6 book chapter related to this subsystem

What specific error or panic are you seeing?
""",
            'compile_error': """
For compile errors, carefully read the error message:
1. Check the file and line number
2. Look for typos or missing declarations
3. Ensure headers are properly included
4. Verify function signatures match declarations

The compiler is usually quite specific about what's wrong.
"""
        }
        
        # Load custom templates from disk
        if self.template_dir.exists():
            for template_file in self.template_dir.glob("*.txt"):
                name = template_file.stem
                templates[name] = template_file.read_text()
        
        return templates

    def get_response(self, context: Dict[str, Any]) -> Optional[str]:
        """Get a template response based on context."""
        # Check for panics
        last_build = context.get('last_build', {})
        panics = last_build.get('panics', [])
        
        if panics:
            panic_msg = panics[0].get('message', '').lower()
            
            if 'sched' in panic_msg and 'lock' in panic_msg:
                return self.templates.get('panic_sched_locks')
            elif 'acquire' in panic_msg:
                return self.templates.get('panic_acquire')
        
        # Check for compile errors
        errors = last_build.get('errors', [])
        if errors:
            return self.templates.get('compile_error')
        
        # Default
        return self.templates.get('general_help')


class LLMClient:
    """
    LLM client with support for multiple providers.
    """

    def __init__(
        self,
        provider: str = "groq",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 800,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 2.0
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Set model based on provider
        if model:
            self.model = model
        elif self.provider == "openai":
            self.model = "gpt-4o"
        elif self.provider == "anthropic":
            self.model = "claude-3-5-sonnet-20241022"
        elif self.provider == "groq":
            self.model = "llama-3.3-70b-versatile"
        else:
            self.model = "gpt-4o"
        
        # Get API key
        if api_key:
            self.api_key = api_key
        elif self.provider == "openai":
            self.api_key = os.environ.get("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif self.provider == "groq":
            self.api_key = os.environ.get("GROQ_API_KEY")
        else:
            self.api_key = None
        
        # Initialize client
        self.client = self._init_client()
        
        # Fallback templates
        self.fallback = FallbackTemplates()

    def _init_client(self):
        """Initialize the appropriate client."""
        if not self.api_key:
            logger.warning(f"No API key for {self.provider}")
            return None
        
        try:
            if self.provider == "openai" and OPENAI_AVAILABLE:
                return openai.OpenAI(api_key=self.api_key)
            elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
                return anthropic.Anthropic(api_key=self.api_key)
            elif self.provider == "groq" and GROQ_AVAILABLE:
                return Groq(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {e}")
        
        return None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            context: Optional context for fallback
            
        Returns:
            LLMResponse with content or error
        """
        start_time = time.time()
        
        # Check if client is available
        if not self.client:
            logger.warning("No LLM client available, using fallback")
            return self._fallback_response(context, start_time)
        
        # Retry loop
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                response = self._call_api(prompt, system_prompt)
                latency = (time.time() - start_time) * 1000
                
                return LLMResponse(
                    content=response['content'],
                    model=self.model,
                    tokens_used=response.get('tokens', 0),
                    latency_ms=latency,
                    success=True
                )
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
        
        # All retries failed
        logger.error(f"All LLM retries failed: {last_error}")
        return self._fallback_response(context, start_time, last_error)

    def _call_api(
        self,
        prompt: str,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Make the actual API call."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        if self.provider == "openai":
            return self._call_openai(messages)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages, system_prompt)
        elif self.provider == "groq":
            return self._call_groq(messages)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, messages: List[Dict]) -> Dict[str, Any]:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return {
            'content': response.choices[0].message.content,
            'tokens': response.usage.total_tokens if response.usage else 0
        }

    def _call_anthropic(
        self,
        messages: List[Dict],
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Call Anthropic API."""
        # Filter out system message (Anthropic uses separate param)
        user_messages = [m for m in messages if m['role'] != 'system']
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt or "You are a helpful assistant.",
            messages=user_messages
        )
        
        return {
            'content': response.content[0].text,
            'tokens': response.usage.input_tokens + response.usage.output_tokens
        }

    def _call_groq(self, messages: List[Dict]) -> Dict[str, Any]:
        """Call Groq API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return {
            'content': response.choices[0].message.content,
            'tokens': response.usage.total_tokens if response.usage else 0
        }

    def _fallback_response(
        self,
        context: Optional[Dict[str, Any]],
        start_time: float,
        error: Optional[str] = None
    ) -> LLMResponse:
        """Generate a fallback response using templates."""
        content = self.fallback.get_response(context or {})
        latency = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=content or "I'm unable to provide assistance at this time.",
            model="fallback_template",
            tokens_used=0,
            latency_ms=latency,
            success=False,
            error=error or "No LLM client available"
        )

    def is_available(self) -> bool:
        """Check if LLM client is available."""
        return self.client is not None


def create_llm_client(
    provider: str = "groq",
    model: Optional[str] = None
) -> LLMClient:
    """Create an LLM client instance."""
    return LLMClient(provider=provider, model=model)


if __name__ == "__main__":
    # Test the LLM client
    logging.basicConfig(level=logging.DEBUG)
    
    # Try Groq first (usually fastest)
    client = create_llm_client(provider="groq")
    
    if client.is_available():
        print(f"Using {client.provider} with model {client.model}")
        
        response = client.generate(
            prompt="What does 'sched locks' panic mean in xv6?",
            system_prompt="You are an xv6 teaching assistant. Be concise."
        )
        
        print(f"\n=== Response ===")
        print(f"Success: {response.success}")
        print(f"Model: {response.model}")
        print(f"Latency: {response.latency_ms:.0f}ms")
        print(f"\nContent:\n{response.content}")
    else:
        print("No LLM client available, testing fallback...")
        
        response = client.generate(
            prompt="test",
            context={'last_build': {'panics': [{'message': 'sched locks'}]}}
        )
        
        print(f"\n=== Fallback Response ===")
        print(response.content)
