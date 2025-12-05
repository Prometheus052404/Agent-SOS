"""
Session Context Module - Session state management for Xv6 agent.

Implements:
- Session state schema
- JSONL logging to .xv6_agent/logs/
- Build result tracking (errors, panics)
- Confidence history tracking
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class BuildInfo:
    """Information about a build attempt."""
    timestamp: str
    success: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    panics: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DiffInfo:
    """Information about file changes."""
    files_changed: List[str] = field(default_factory=list)
    semantic_delta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionContext:
    """Complete session context for the agent."""
    # Session identification
    session_id: str
    session_start: str
    
    # Task tracking
    task_id: str = ""
    current_state: str = "NOT_STARTED"
    progress: float = 0.0
    
    # Build information
    last_build: Optional[BuildInfo] = None
    
    # Diff information
    diff_engine: DiffInfo = field(default_factory=DiffInfo)
    
    # LLM tracking
    llm_consent_given: bool = False
    llm_confidence_history: List[float] = field(default_factory=list)
    
    # Query history
    queries: List[Dict[str, Any]] = field(default_factory=list)
    
    # Hints given
    hints_given: List[str] = field(default_factory=list)


class SessionLogger:
    """Logs session events in JSONL format."""

    def __init__(self, log_dir: str = ".xv6_agent/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"session_{date_str}.jsonl"

    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        session_id: Optional[str] = None
    ):
        """Log an event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'session_id': session_id,
            **data
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def log_build(self, build_info: BuildInfo, session_id: str):
        """Log a build event."""
        self.log_event('build', asdict(build_info), session_id)

    def log_query(
        self,
        query: str,
        response: str,
        confidence: float,
        session_id: str
    ):
        """Log a query event."""
        self.log_event('query', {
            'query': query,
            'response_preview': response[:200] if response else '',
            'confidence': confidence
        }, session_id)

    def log_transition(
        self,
        from_state: str,
        to_state: str,
        session_id: str
    ):
        """Log a state transition."""
        self.log_event('transition', {
            'from_state': from_state,
            'to_state': to_state
        }, session_id)

    def get_session_events(
        self,
        session_id: str,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get events for a session."""
        events = []
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('session_id') == session_id:
                            if event_type is None or event.get('type') == event_type:
                                events.append(event)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        
        return events


class SessionManager:
    """Manages session context and logging."""

    def __init__(
        self,
        context_file: str = ".xv6_agent/session_context.json",
        log_dir: str = ".xv6_agent/logs"
    ):
        self.context_file = Path(context_file)
        self.context_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = SessionLogger(log_dir)
        self.context: Optional[SessionContext] = None
        
        # Load or create session
        self._load_or_create_session()

    def _load_or_create_session(self):
        """Load existing session or create new one."""
        if self.context_file.exists():
            try:
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct context
                self.context = SessionContext(
                    session_id=data.get('session_id', self._generate_session_id()),
                    session_start=data.get('session_start', datetime.now().isoformat()),
                    task_id=data.get('task_id', ''),
                    current_state=data.get('current_state', 'NOT_STARTED'),
                    progress=data.get('progress', 0.0),
                    llm_consent_given=data.get('llm_consent_given', False),
                    llm_confidence_history=data.get('llm_confidence_history', []),
                    queries=data.get('queries', []),
                    hints_given=data.get('hints_given', [])
                )
                
                # Reconstruct build info
                if data.get('last_build'):
                    self.context.last_build = BuildInfo(**data['last_build'])
                
                # Reconstruct diff info
                if data.get('diff_engine'):
                    self.context.diff_engine = DiffInfo(**data['diff_engine'])
                
                logger.info(f"Loaded session: {self.context.session_id}")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load session, creating new: {e}")
        
        # Create new session
        self.context = SessionContext(
            session_id=self._generate_session_id(),
            session_start=datetime.now().isoformat()
        )
        self._save_context()
        logger.info(f"Created new session: {self.context.session_id}")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def _save_context(self):
        """Save context to disk."""
        if not self.context:
            return
        
        data = {
            'session_id': self.context.session_id,
            'session_start': self.context.session_start,
            'task_id': self.context.task_id,
            'current_state': self.context.current_state,
            'progress': self.context.progress,
            'last_build': asdict(self.context.last_build) if self.context.last_build else None,
            'diff_engine': asdict(self.context.diff_engine),
            'llm_consent_given': self.context.llm_consent_given,
            'llm_confidence_history': self.context.llm_confidence_history,
            'queries': self.context.queries[-10:],  # Keep last 10 queries
            'hints_given': self.context.hints_given[-20:]  # Keep last 20 hints
        }
        
        try:
            with open(self.context_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save context: {e}")

    def get_context(self) -> SessionContext:
        """Get the current session context."""
        return self.context

    def update_build_result(
        self,
        success: bool,
        errors: List[Dict[str, Any]] = None,
        panics: List[Dict[str, Any]] = None,
        warnings: List[Dict[str, Any]] = None
    ):
        """Update with a new build result."""
        build_info = BuildInfo(
            timestamp=datetime.now().isoformat(),
            success=success,
            errors=errors or [],
            panics=panics or [],
            warnings=warnings or []
        )
        
        self.context.last_build = build_info
        self.logger.log_build(build_info, self.context.session_id)
        self._save_context()

    def update_diff(
        self,
        files_changed: List[str],
        semantic_delta: Dict[str, Any] = None
    ):
        """Update with new diff information."""
        self.context.diff_engine = DiffInfo(
            files_changed=files_changed,
            semantic_delta=semantic_delta or {}
        )
        self._save_context()

    def update_task_state(
        self,
        task_id: str,
        current_state: str,
        progress: float
    ):
        """Update task tracking information."""
        old_state = self.context.current_state
        
        self.context.task_id = task_id
        self.context.current_state = current_state
        self.context.progress = progress
        
        if old_state != current_state:
            self.logger.log_transition(
                old_state,
                current_state,
                self.context.session_id
            )
        
        self._save_context()

    def add_query(
        self,
        query: str,
        response: str,
        confidence: float
    ):
        """Record a query and response."""
        self.context.queries.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response_preview': response[:200] if response else '',
            'confidence': confidence
        })
        
        self.context.llm_confidence_history.append(confidence)
        
        self.logger.log_query(
            query,
            response,
            confidence,
            self.context.session_id
        )
        
        self._save_context()

    def add_hint(self, hint: str):
        """Record a hint given."""
        self.context.hints_given.append(hint)
        self._save_context()

    def set_consent(self, consent: bool):
        """Set LLM API consent."""
        self.context.llm_consent_given = consent
        self._save_context()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the session."""
        return {
            'session_id': self.context.session_id,
            'session_start': self.context.session_start,
            'task_id': self.context.task_id,
            'current_state': self.context.current_state,
            'progress': f"{self.context.progress * 100:.0f}%",
            'last_build_success': self.context.last_build.success if self.context.last_build else None,
            'files_changed': len(self.context.diff_engine.files_changed),
            'queries_count': len(self.context.queries),
            'hints_count': len(self.context.hints_given),
            'avg_confidence': (
                sum(self.context.llm_confidence_history) /
                len(self.context.llm_confidence_history)
                if self.context.llm_confidence_history else 0.0
            )
        }

    def reset_session(self):
        """Reset to a new session."""
        self.context = SessionContext(
            session_id=self._generate_session_id(),
            session_start=datetime.now().isoformat()
        )
        self._save_context()
        logger.info(f"Reset to new session: {self.context.session_id}")


def create_session_manager(
    context_file: str = ".xv6_agent/session_context.json"
) -> SessionManager:
    """Create a session manager instance."""
    return SessionManager(context_file=context_file)


if __name__ == "__main__":
    # Test the session manager
    logging.basicConfig(level=logging.DEBUG)
    
    manager = create_session_manager()
    
    # Update build result
    manager.update_build_result(
        success=False,
        errors=[{'file': 'proc.c', 'line': 42, 'message': 'undefined variable'}],
        panics=[{'message': 'sched locks'}]
    )
    
    # Add a query
    manager.add_query(
        query="Why is it panicking?",
        response="The panic occurs because...",
        confidence=0.75
    )
    
    # Print summary
    print("\n=== Session Summary ===")
    print(json.dumps(manager.get_summary(), indent=2))
