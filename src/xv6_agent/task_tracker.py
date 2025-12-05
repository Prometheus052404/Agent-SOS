"""
Task Tracker Module - Hybrid FSM for lab progress tracking.

Implements:
- Lab FSM definitions (Lock, Thread, FS)
- Adaptive confidence thresholds
- State transition validation with lambda functions
- Progress tracking with blocker detection
- Auto-detect active lab from file changes
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class LabType(Enum):
    """Types of labs supported."""
    LOCK = "lock"
    THREAD = "thread"
    FS = "fs"
    VM = "vm"
    NET = "net"
    UNKNOWN = "unknown"


@dataclass
class Blocker:
    """Represents a blocker preventing progress."""
    type: str  # panic, error, etc.
    message: str
    first_seen: str
    occurrence_count: int = 1


@dataclass
class TaskState:
    """Current state of a task/lab."""
    task_id: str
    current_state: str
    progress: float = 0.0
    session_start: str = ""
    substeps_completed: List[str] = field(default_factory=list)
    blockers: List[Blocker] = field(default_factory=list)
    llm_confidence_history: List[float] = field(default_factory=list)
    hints_given: List[str] = field(default_factory=list)


# Lab FSM Definitions
LAB_DEFINITIONS = {
    LabType.LOCK: {
        'states': [
            'NOT_STARTED',
            'IMPLEMENTING_ACQUIRE',
            'TESTING_ACQUIRE',
            'IMPLEMENTING_RELEASE',
            'TESTING_RELEASE',
            'IMPLEMENTING_SLEEP_LOCK',
            'TESTING_SLEEP_LOCK',
            'VALIDATION',
            'COMPLETED'
        ],
        'file_signatures': ['spinlock.c', 'sleeplock.c', 'proc.c'],
        'progress_map': {
            'NOT_STARTED': 0.0,
            'IMPLEMENTING_ACQUIRE': 0.1,
            'TESTING_ACQUIRE': 0.25,
            'IMPLEMENTING_RELEASE': 0.35,
            'TESTING_RELEASE': 0.5,
            'IMPLEMENTING_SLEEP_LOCK': 0.65,
            'TESTING_SLEEP_LOCK': 0.8,
            'VALIDATION': 0.9,
            'COMPLETED': 1.0
        }
    },
    LabType.THREAD: {
        'states': [
            'NOT_STARTED',
            'IMPLEMENTING_THREAD_CREATE',
            'IMPLEMENTING_THREAD_SWITCH',
            'TESTING_CONTEXT_SWITCH',
            'IMPLEMENTING_SCHEDULER',
            'VALIDATION',
            'COMPLETED'
        ],
        'file_signatures': ['proc.c', 'swtch.S'],
        'progress_map': {
            'NOT_STARTED': 0.0,
            'IMPLEMENTING_THREAD_CREATE': 0.15,
            'IMPLEMENTING_THREAD_SWITCH': 0.35,
            'TESTING_CONTEXT_SWITCH': 0.55,
            'IMPLEMENTING_SCHEDULER': 0.75,
            'VALIDATION': 0.9,
            'COMPLETED': 1.0
        }
    },
    LabType.FS: {
        'states': [
            'NOT_STARTED',
            'IMPLEMENTING_INODE_ALLOC',
            'IMPLEMENTING_BLOCK_CACHE',
            'IMPLEMENTING_LOGGING',
            'TESTING_CRASH_RECOVERY',
            'VALIDATION',
            'COMPLETED'
        ],
        'file_signatures': ['fs.c', 'log.c', 'bio.c'],
        'progress_map': {
            'NOT_STARTED': 0.0,
            'IMPLEMENTING_INODE_ALLOC': 0.15,
            'IMPLEMENTING_BLOCK_CACHE': 0.35,
            'IMPLEMENTING_LOGGING': 0.55,
            'TESTING_CRASH_RECOVERY': 0.75,
            'VALIDATION': 0.9,
            'COMPLETED': 1.0
        }
    }
}


class AdaptiveFSM:
    """
    FSM with adaptive confidence thresholds based on state type.
    
    Thresholds adjust by state:
    - IMPLEMENTING: Lower bar (0.7 * 0.8 = 0.56)
    - TESTING: Normal bar (0.7 * 1.0 = 0.70)
    - VALIDATION: Higher bar (0.7 * 1.2 = 0.84)
    """

    def __init__(
        self,
        base_threshold: float = 0.7,
        state_multipliers: Optional[Dict[str, float]] = None
    ):
        self.base_threshold = base_threshold
        self.state_multipliers = state_multipliers or {
            'IMPLEMENTING': 0.8,
            'TESTING': 1.0,
            'VALIDATION': 1.2,
            'NOT_STARTED': 0.5,
            'COMPLETED': 1.0
        }

    def get_threshold_for_state(self, state: str) -> float:
        """Get confidence threshold for a state."""
        for prefix, multiplier in self.state_multipliers.items():
            if prefix in state:
                return self.base_threshold * multiplier
        return self.base_threshold


class TaskTracker:
    """
    Tracks lab progress using hybrid FSM approach.
    """

    def __init__(
        self,
        state_file: Optional[str] = None,
        base_confidence_threshold: float = 0.7
    ):
        self.state_file = Path(state_file) if state_file else None
        self.adaptive_fsm = AdaptiveFSM(base_threshold=base_confidence_threshold)
        
        # Current state
        self.current_lab: Optional[LabType] = None
        self.task_state: Optional[TaskState] = None
        
        # Transition validators
        self.transition_validators: Dict[str, Callable] = {}
        self._setup_validators()
        
        # Load state if exists
        if self.state_file and self.state_file.exists():
            self._load_state()

    def _setup_validators(self):
        """Setup transition validators."""
        # Lock lab transitions
        self.transition_validators['IMPLEMENTING_ACQUIRE->TESTING_ACQUIRE'] = \
            lambda ctx: len(ctx.get('compiler_errors', [])) == 0
        
        self.transition_validators['TESTING_ACQUIRE->IMPLEMENTING_RELEASE'] = \
            lambda ctx: not any('acquire' in str(p) for p in ctx.get('panics', []))
        
        self.transition_validators['IMPLEMENTING_RELEASE->TESTING_RELEASE'] = \
            lambda ctx: len(ctx.get('compiler_errors', [])) == 0
        
        self.transition_validators['TESTING_RELEASE->IMPLEMENTING_SLEEP_LOCK'] = \
            lambda ctx: not any('lock' in str(p) for p in ctx.get('panics', []))

    def detect_lab(self, modified_files: List[str]) -> LabType:
        """Auto-detect which lab based on modified files."""
        file_names = [Path(f).name for f in modified_files]
        
        for lab_type, definition in LAB_DEFINITIONS.items():
            signatures = definition['file_signatures']
            if any(sig in file_names for sig in signatures):
                return lab_type
        
        return LabType.UNKNOWN

    def start_lab(self, lab_type: LabType):
        """Start tracking a new lab."""
        if lab_type not in LAB_DEFINITIONS:
            logger.warning(f"Unknown lab type: {lab_type}")
            return
        
        self.current_lab = lab_type
        definition = LAB_DEFINITIONS[lab_type]
        
        self.task_state = TaskState(
            task_id=f"lab_{lab_type.value}",
            current_state=definition['states'][0],
            progress=0.0,
            session_start=datetime.now().isoformat()
        )
        
        self._save_state()
        logger.info(f"Started lab: {lab_type.value}")

    def get_current_state(self) -> Optional[TaskState]:
        """Get current task state."""
        return self.task_state

    def get_progress(self) -> float:
        """Get current progress (0.0 - 1.0)."""
        if not self.task_state:
            return 0.0
        return self.task_state.progress

    def attempt_transition(
        self,
        context: Dict[str, Any],
        llm_confidence: Optional[float] = None
    ) -> Optional[str]:
        """
        Attempt to transition to next state.
        
        Args:
            context: Current context with errors, panics, etc.
            llm_confidence: Confidence score from LLM
            
        Returns:
            New state if transitioned, None otherwise
        """
        if not self.task_state or not self.current_lab:
            return None
        
        current_state = self.task_state.current_state
        definition = LAB_DEFINITIONS[self.current_lab]
        states = definition['states']
        
        # Find next state
        try:
            current_idx = states.index(current_state)
            if current_idx >= len(states) - 1:
                return None  # Already at final state
            next_state = states[current_idx + 1]
        except ValueError:
            return None
        
        # Check validator
        transition_key = f"{current_state}->{next_state}"
        validator = self.transition_validators.get(transition_key)
        
        if validator and not validator(context):
            logger.debug(f"Transition blocked by validator: {transition_key}")
            return None
        
        # Check confidence threshold
        if llm_confidence is not None:
            threshold = self.adaptive_fsm.get_threshold_for_state(current_state)
            if llm_confidence < threshold:
                logger.debug(
                    f"Transition blocked: confidence {llm_confidence:.2f} "
                    f"< threshold {threshold:.2f}"
                )
                return None
        
        # Perform transition
        self.task_state.current_state = next_state
        self.task_state.progress = definition['progress_map'].get(next_state, 0.0)
        
        if llm_confidence is not None:
            self.task_state.llm_confidence_history.append(llm_confidence)
        
        self._save_state()
        logger.info(f"Transitioned: {current_state} -> {next_state}")
        
        return next_state

    def add_blocker(
        self,
        blocker_type: str,
        message: str
    ):
        """Add or update a blocker."""
        if not self.task_state:
            return
        
        # Check if blocker already exists
        for blocker in self.task_state.blockers:
            if blocker.type == blocker_type and blocker.message == message:
                blocker.occurrence_count += 1
                self._save_state()
                return
        
        # Add new blocker
        self.task_state.blockers.append(Blocker(
            type=blocker_type,
            message=message,
            first_seen=datetime.now().isoformat()
        ))
        
        self._save_state()

    def clear_blockers(self, blocker_type: Optional[str] = None):
        """Clear blockers, optionally by type."""
        if not self.task_state:
            return
        
        if blocker_type:
            self.task_state.blockers = [
                b for b in self.task_state.blockers
                if b.type != blocker_type
            ]
        else:
            self.task_state.blockers = []
        
        self._save_state()

    def add_hint(self, hint: str):
        """Record a hint given to the student."""
        if self.task_state:
            self.task_state.hints_given.append(hint)
            self._save_state()

    def add_substep(self, substep: str):
        """Mark a substep as completed."""
        if self.task_state and substep not in self.task_state.substeps_completed:
            self.task_state.substeps_completed.append(substep)
            self._save_state()

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of current status."""
        if not self.task_state:
            return {
                'active': False,
                'message': 'No lab in progress'
            }
        
        return {
            'active': True,
            'lab': self.current_lab.value if self.current_lab else 'unknown',
            'state': self.task_state.current_state,
            'progress': f"{self.task_state.progress * 100:.0f}%",
            'blockers': len(self.task_state.blockers),
            'hints_given': len(self.task_state.hints_given),
            'confidence_avg': (
                sum(self.task_state.llm_confidence_history) /
                len(self.task_state.llm_confidence_history)
                if self.task_state.llm_confidence_history else 0.0
            )
        }

    def _save_state(self):
        """Save state to disk."""
        if not self.state_file or not self.task_state:
            return
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'lab': self.current_lab.value if self.current_lab else None,
            'task_id': self.task_state.task_id,
            'current_state': self.task_state.current_state,
            'progress': self.task_state.progress,
            'session_start': self.task_state.session_start,
            'substeps_completed': self.task_state.substeps_completed,
            'blockers': [
                {
                    'type': b.type,
                    'message': b.message,
                    'first_seen': b.first_seen,
                    'occurrence_count': b.occurrence_count
                }
                for b in self.task_state.blockers
            ],
            'llm_confidence_history': self.task_state.llm_confidence_history,
            'hints_given': self.task_state.hints_given
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_state(self):
        """Load state from disk."""
        if not self.state_file or not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            lab_value = data.get('lab')
            self.current_lab = LabType(lab_value) if lab_value else None
            
            self.task_state = TaskState(
                task_id=data.get('task_id', ''),
                current_state=data.get('current_state', 'NOT_STARTED'),
                progress=data.get('progress', 0.0),
                session_start=data.get('session_start', ''),
                substeps_completed=data.get('substeps_completed', []),
                blockers=[
                    Blocker(
                        type=b['type'],
                        message=b['message'],
                        first_seen=b['first_seen'],
                        occurrence_count=b.get('occurrence_count', 1)
                    )
                    for b in data.get('blockers', [])
                ],
                llm_confidence_history=data.get('llm_confidence_history', []),
                hints_given=data.get('hints_given', [])
            )
            
            logger.info(f"Loaded state: {self.task_state.current_state}")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")


def create_task_tracker(
    state_file: str = ".xv6_agent/task_state.json"
) -> TaskTracker:
    """Create a configured task tracker."""
    return TaskTracker(state_file=state_file)


if __name__ == "__main__":
    # Test the task tracker
    logging.basicConfig(level=logging.DEBUG)
    
    tracker = create_task_tracker()
    
    # Start lock lab
    tracker.start_lab(LabType.LOCK)
    
    # Check status
    print("\n=== Status ===")
    print(tracker.get_status_summary())
    
    # Simulate progress
    context = {'compiler_errors': [], 'panics': []}
    
    # Try to transition
    new_state = tracker.attempt_transition(context, llm_confidence=0.8)
    print(f"Transitioned to: {new_state}")
    
    print("\n=== Final Status ===")
    print(tracker.get_status_summary())


# Missing method implementations
def record_hint_to_tracker(tracker, hint: str):
    """Record a hint given to the student."""
    if hasattr(tracker, 'task_state') and tracker.task_state:
        tracker.task_state.hints_given.append(hint)

# Patch TaskTracker class
TaskTracker.record_hint = lambda self, hint: record_hint_to_tracker(self, hint)


class ErrorTracker:
    """Track recurring errors/mistakes."""
    
    def __init__(self):
        self.error_counts = {}
    
    def record_error(self, error_type: str, message: str):
        """Record an error occurrence."""
        key = f"{error_type}:{message[:50]}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def get_recurring_errors(self, min_count: int = 2):
        """Get errors that occurred multiple times."""
        return {k: v for k, v in self.error_counts.items() if v >= min_count}
