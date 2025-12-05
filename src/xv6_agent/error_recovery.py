"""
Error Recovery Module - Graceful degradation for Xv6 agent.

Implements:
- CPG fallback to grep
- LLM fallback to templates
- Health check daemon (60-second interval)
- Auto-restart failed components
- Degraded mode notification
"""

import time
import logging
import threading
import subprocess
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Status of a component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class HealthReport:
    """Health report for a component."""
    component: str
    status: ComponentStatus
    message: str
    fallback_active: bool = False
    last_error: Optional[str] = None


class GrepFallback:
    """Fallback search using grep when CPG is unavailable."""

    def __init__(self, search_dir: str = "."):
        self.search_dir = search_dir

    def search_function(self, function_name: str) -> List[Dict[str, Any]]:
        """Search for a function definition using grep."""
        try:
            # Search for function definition pattern
            pattern = rf'^(?:static\s+)?(?:\w+\s+)+{function_name}\s*\('
            
            result = subprocess.run(
                ['grep', '-rn', '-E', pattern, '--include=*.c', '--include=*.h', self.search_dir],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            results = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        results.append({
                            'file': parts[0],
                            'line': int(parts[1]),
                            'content': parts[2]
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Grep fallback failed: {e}")
            return []

    def search_references(self, symbol: str) -> List[Dict[str, Any]]:
        """Search for references to a symbol."""
        try:
            result = subprocess.run(
                ['grep', '-rn', symbol, '--include=*.c', '--include=*.h', self.search_dir],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            results = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        results.append({
                            'file': parts[0],
                            'line': int(parts[1]),
                            'content': parts[2]
                        })
            
            return results[:20]  # Limit results
            
        except Exception as e:
            logger.error(f"Grep reference search failed: {e}")
            return []


class TemplateFallback:
    """Fallback responses when LLM is unavailable."""

    def __init__(self):
        self.templates = {
            'panic': """
I can see you're experiencing a kernel panic. Without access to the 
AI assistant, here's general guidance:

1. Read the panic message carefully - it tells you what invariant was violated
2. Check the backtrace to see which function triggered it
3. Look at your recent code changes near that function
4. Review the relevant chapter in the xv6 book

Common panics and their meanings:
- "sched locks": Scheduler called with locks held
- "acquire": Lock already held or ordering issue
- "remap": Page table mapping error
""",
            'compile_error': """
Compile errors need to be fixed before the code can run.

Tips:
1. Read the error message - the compiler is usually specific
2. Check the line number mentioned
3. Look for typos, missing semicolons, or undeclared variables
4. Make sure all headers are included

If you're stuck, try commenting out the problematic code and building 
incrementally.
""",
            'general': """
I'm currently running in offline mode. Here's what I can help with:

1. Check your code for common issues
2. Re-read the relevant sections of the xv6 book
3. Look at similar implementations in the existing codebase
4. Try running the tests to see what specific checks are failing

When the AI assistant is available again, I can provide more targeted help.
"""
        }

    def get_response(self, context: Dict[str, Any]) -> str:
        """Get appropriate fallback response."""
        last_build = context.get('last_build', {})
        
        if last_build.get('panics'):
            return self.templates['panic']
        elif last_build.get('errors'):
            return self.templates['compile_error']
        else:
            return self.templates['general']


class ComponentHealth:
    """Tracks health of a single component."""

    def __init__(
        self,
        name: str,
        check_fn: Callable[[], bool],
        restart_fn: Optional[Callable[[], bool]] = None,
        fallback_fn: Optional[Callable[[], Any]] = None
    ):
        self.name = name
        self.check_fn = check_fn
        self.restart_fn = restart_fn
        self.fallback_fn = fallback_fn
        
        self.status = ComponentStatus.UNKNOWN
        self.failure_count = 0
        self.last_error: Optional[str] = None
        self.fallback_active = False

    def check(self) -> HealthReport:
        """Check component health."""
        try:
            if self.check_fn():
                self.status = ComponentStatus.HEALTHY
                self.failure_count = 0
                self.fallback_active = False
                return HealthReport(
                    component=self.name,
                    status=ComponentStatus.HEALTHY,
                    message="Component healthy"
                )
        except Exception as e:
            self.last_error = str(e)
        
        # Health check failed
        self.failure_count += 1
        
        if self.failure_count >= 3 and self.restart_fn:
            # Try to restart
            try:
                if self.restart_fn():
                    self.status = ComponentStatus.HEALTHY
                    self.failure_count = 0
                    return HealthReport(
                        component=self.name,
                        status=ComponentStatus.HEALTHY,
                        message="Component restarted successfully"
                    )
            except Exception as e:
                self.last_error = str(e)
        
        # Activate fallback if available
        if self.fallback_fn:
            self.fallback_active = True
            self.status = ComponentStatus.DEGRADED
            return HealthReport(
                component=self.name,
                status=ComponentStatus.DEGRADED,
                message="Running in fallback mode",
                fallback_active=True,
                last_error=self.last_error
            )
        
        self.status = ComponentStatus.FAILED
        return HealthReport(
            component=self.name,
            status=ComponentStatus.FAILED,
            message="Component failed",
            last_error=self.last_error
        )


class HealthDaemon(threading.Thread):
    """Background daemon for health monitoring."""

    def __init__(
        self,
        components: List[ComponentHealth],
        check_interval: int = 60,
        on_status_change: Optional[Callable[[HealthReport], None]] = None
    ):
        super().__init__(daemon=True)
        self.components = components
        self.check_interval = check_interval
        self.on_status_change = on_status_change
        self.running = False
        self.last_reports: Dict[str, HealthReport] = {}

    def run(self):
        """Run the health check loop."""
        self.running = True
        
        while self.running:
            for component in self.components:
                old_status = component.status
                report = component.check()
                
                # Notify on status change
                if old_status != report.status:
                    logger.info(f"Component {report.component}: {old_status.value} -> {report.status.value}")
                    
                    if self.on_status_change:
                        self.on_status_change(report)
                
                self.last_reports[component.name] = report
            
            time.sleep(self.check_interval)

    def stop(self):
        """Stop the daemon."""
        self.running = False

    def get_reports(self) -> Dict[str, HealthReport]:
        """Get all current health reports."""
        return self.last_reports


class ErrorRecoveryManager:
    """Manages error recovery and fallbacks for all components."""

    def __init__(self, agent):
        self.agent = agent
        
        # Fallbacks
        self.grep_fallback = GrepFallback(str(agent.workspace_dir))
        self.template_fallback = TemplateFallback()
        
        # Component health trackers
        self.components: List[ComponentHealth] = []
        self._setup_components()
        
        # Health daemon
        self.health_daemon: Optional[HealthDaemon] = None

    def _setup_components(self):
        """Setup health tracking for all components."""
        # CPG health
        self.components.append(ComponentHealth(
            name="cpg",
            check_fn=lambda: self.agent.cpg_builder.graph.number_of_nodes() > 0,
            restart_fn=lambda: self._restart_cpg(),
            fallback_fn=lambda: self.grep_fallback
        ))
        
        # LLM health
        self.components.append(ComponentHealth(
            name="llm",
            check_fn=lambda: self.agent.llm_client.is_available(),
            fallback_fn=lambda: self.template_fallback
        ))
        
        # Vector store health
        self.components.append(ComponentHealth(
            name="vector_store",
            check_fn=lambda: self.agent.vector_store.collection is not None
        ))

    def _restart_cpg(self) -> bool:
        """Attempt to restart CPG builder."""
        try:
            self.agent._cpg_builder = None  # Force re-initialization
            self.agent.cpg_builder.build_from_directory(str(self.agent.workspace_dir))
            return True
        except Exception as e:
            logger.error(f"CPG restart failed: {e}")
            return False

    def start_monitoring(self):
        """Start the health monitoring daemon."""
        if self.health_daemon is None or not self.health_daemon.is_alive():
            self.health_daemon = HealthDaemon(
                components=self.components,
                check_interval=60,
                on_status_change=self._on_status_change
            )
            self.health_daemon.start()
            logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop the health monitoring daemon."""
        if self.health_daemon:
            self.health_daemon.stop()
            logger.info("Health monitoring stopped")

    def _on_status_change(self, report: HealthReport):
        """Handle component status change."""
        if report.status == ComponentStatus.DEGRADED:
            logger.warning(
                f"Component {report.component} is degraded: {report.message}"
            )
        elif report.status == ComponentStatus.FAILED:
            logger.error(
                f"Component {report.component} has failed: {report.last_error}"
            )

    def get_fallback_for_cpg(self):
        """Get CPG fallback (grep search)."""
        cpg_component = next(
            (c for c in self.components if c.name == "cpg"),
            None
        )
        
        if cpg_component and cpg_component.fallback_active:
            return self.grep_fallback
        
        return None

    def get_fallback_for_llm(self):
        """Get LLM fallback (templates)."""
        llm_component = next(
            (c for c in self.components if c.name == "llm"),
            None
        )
        
        if llm_component and cpg_component.fallback_active:
            return self.template_fallback
        
        return None

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of all component health."""
        return {
            component.name: {
                'status': component.status.value,
                'failures': component.failure_count,
                'fallback': component.fallback_active,
                'error': component.last_error
            }
            for component in self.components
        }


if __name__ == "__main__":
    # Test fallbacks
    logging.basicConfig(level=logging.DEBUG)
    
    # Test grep fallback
    grep = GrepFallback(".")
    results = grep.search_function("main")
    print(f"\nGrep search results: {len(results)} matches")
    for r in results[:3]:
        print(f"  {r['file']}:{r['line']}")
    
    # Test template fallback
    templates = TemplateFallback()
    response = templates.get_response({'last_build': {'panics': [{'message': 'test'}]}})
    print(f"\nTemplate response preview:\n{response[:200]}...")
