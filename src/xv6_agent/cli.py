"""
CLI Module - Command-line interface for Xv6 agent.

Implements:
- agent init - Initialize workspace
- agent make [args] - Build wrapper with interception
- agent help "query" - Interactive help
- agent status - Show FSM state
- agent undo - Single-level rollback
- agent restore <id> - Deep rollback
- agent debug - Show last context assembly
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

try:
    import click
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    click = None

# Import agent modules
try:
    from file_sentinel import create_file_sentinel
    from shadow_workspace import ShadowWorkspace
    from diff_engine import DiffEngine
    from interceptor import AgentMakeWrapper
    from cpg_builder import CPGBuilder
    from vector_store import VectorStore
    from task_tracker import TaskTracker, LabType
    from session_context import SessionManager
    from context_assembler import ContextAssembler, ConsentManager
    from llm_client import LLMClient
    from pedagogical_validator import PedagogicalValidator, ConfidenceScorer
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Console for rich output
console = Console() if RICH_AVAILABLE else None

# Global state (lazy initialized)
_agent = None


def get_agent():
    """Get or create the agent instance."""
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


class Agent:
    """Main agent orchestration class."""

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.agent_dir = self.workspace_dir / ".xv6_agent"
        
        # Initialize components lazily
        self._shadow_workspace = None
        self._task_tracker = None
        self._session_manager = None
        self._cpg_builder = None
        self._vector_store = None
        self._context_assembler = None
        self._llm_client = None
        self._validator = None
        self._make_wrapper = None
        self._diff_engine = None
        self._consent_manager = None

    @property
    def shadow_workspace(self):
        if self._shadow_workspace is None:
            self._shadow_workspace = ShadowWorkspace(
                workspace_root=str(self.agent_dir),
                source_dir=str(self.workspace_dir)
            )
        return self._shadow_workspace

    @property
    def task_tracker(self):
        if self._task_tracker is None:
            self._task_tracker = TaskTracker(
                state_file=str(self.agent_dir / "task_state.json")
            )
        return self._task_tracker

    @property
    def session_manager(self):
        if self._session_manager is None:
            self._session_manager = SessionManager(
                context_file=str(self.agent_dir / "session_context.json")
            )
        return self._session_manager

    @property
    def cpg_builder(self):
        if self._cpg_builder is None:
            self._cpg_builder = CPGBuilder(
                cache_dir=str(self.agent_dir / "cpg_cache")
            )
        return self._cpg_builder

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = VectorStore(
                persist_dir=str(self.agent_dir / "chroma_db")
            )
        return self._vector_store

    @property
    def context_assembler(self):
        if self._context_assembler is None:
            self._context_assembler = ContextAssembler(max_tokens=3000)
        return self._context_assembler

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = LLMClient(provider="groq")
        return self._llm_client

    @property
    def validator(self):
        if self._validator is None:
            self._validator = PedagogicalValidator()
        return self._validator

    @property
    def make_wrapper(self):
        if self._make_wrapper is None:
            self._make_wrapper = AgentMakeWrapper(
                project_dir=str(self.workspace_dir),
                on_build_complete=self._on_build_complete
            )
        return self._make_wrapper

    @property
    def diff_engine(self):
        if self._diff_engine is None:
            self._diff_engine = DiffEngine()
        return self._diff_engine

    @property
    def consent_manager(self):
        if self._consent_manager is None:
            self._consent_manager = ConsentManager(
                consent_file=str(self.agent_dir / "consent.json")
            )
        return self._consent_manager

    def _on_build_complete(self, result):
        """Callback when build completes."""
        self.session_manager.update_build_result(
            success=result.success,
            errors=result.errors,
            panics=result.panics,
            warnings=result.warnings
        )
        
        # Update blockers
        if result.panics:
            for panic in result.panics:
                self.task_tracker.add_blocker('panic', panic.get('message', 'unknown'))
        elif result.errors:
            for error in result.errors:
                self.task_tracker.add_blocker('error', error.get('message', 'unknown'))
        else:
            self.task_tracker.clear_blockers()

    def initialize(self):
        """Initialize the agent workspace."""
        # Create directories
        dirs = [
            self.agent_dir,
            self.agent_dir / "current_ref" / "kernel",
            self.agent_dir / "snapshots" / "auto",
            self.agent_dir / "snapshots" / "manual",
            self.agent_dir / "chroma_db",
            self.agent_dir / "cpg_cache",
            self.agent_dir / "logs",
            self.agent_dir / "templates"
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        # Build initial CPG
        if console:
            console.print("[bold blue]Building Code Property Graph...[/]")
        
        self.cpg_builder.build_from_directory(str(self.workspace_dir))
        self.cpg_builder.save_cache()
        
        stats = self.cpg_builder.get_stats()
        if console:
            console.print(f"[green]✓[/] CPG built: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        
        # Create initial snapshot
        self.shadow_workspace.snapshot_before_patch("initial")
        
        if console:
            console.print("[green]✓[/] Agent initialized successfully!")

    def process_query(self, query: str) -> str:
        """Process a help query."""
        context = self.session_manager.get_context()
        
        # Get consent if needed
        if not self.consent_manager.consent_given:
            code_preview = f"Query: {query}\nState: {context.current_state}"
            if not self.consent_manager.request_consent(code_preview):
                return "Query cancelled - consent not given."
        
        # Get textbook chunks
        textbook_chunks = self.vector_store.query_for_context(
            query_text=query,
            task_id=context.task_id
        )
        
        # Assemble context
        assembled = self.context_assembler.assemble_context(
            user_query=query,
            session_state={
                'task_id': context.task_id,
                'current_state': context.current_state,
                'progress': context.progress,
                'last_build': {
                    'errors': context.last_build.errors if context.last_build else [],
                    'panics': context.last_build.panics if context.last_build else []
                } if context.last_build else {},
                'diff_engine': {
                    'files_changed': context.diff_engine.files_changed
                }
            },
            textbook_chunks=[textbook_chunks] if textbook_chunks else []
        )
        
        # Call LLM
        response = self.llm_client.generate(
            prompt=assembled['prompt'],
            context={'last_build': context.last_build.__dict__ if context.last_build else {}}
        )
        
        # Validate response
        validation = self.validator.validate(response.content)
        
        if not validation.passed:
            if console:
                console.print(f"[yellow]⚠ Response blocked: {validation.blocked_reason}[/]")
            response.content = self.validator.sanitize_response(response.content)
        
        # Score confidence
        scorer = ConfidenceScorer()
        confidence = scorer.score(
            response.content,
            {
                'task_id': context.task_id,
                'current_state': context.current_state,
                'last_build': context.last_build.__dict__ if context.last_build else {}
            }
        )
        
        # Log query
        self.session_manager.add_query(query, response.content, confidence.value)
        
        # Attempt FSM transition
        self.task_tracker.attempt_transition(
            context={
                'compiler_errors': context.last_build.errors if context.last_build else [],
                'panics': context.last_build.panics if context.last_build else []
            },
            llm_confidence=confidence.value
        )
        
        return response.content


# CLI commands using click if available
if click:
    @click.group()
    @click.option('--debug', is_flag=True, help='Enable debug logging')
    def cli(debug):
        """Xv6 Teaching Assistant Agent"""
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)

    @cli.command()
    def init():
        """Initialize the agent workspace."""
        agent = get_agent()
        
        console.print(Panel.fit(
            "[bold]Xv6 Agent Initialization[/]",
            subtitle="Setting up workspace"
        ))
        
        try:
            agent.initialize()
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            raise SystemExit(1)

    @cli.command()
    @click.argument('args', nargs=-1)
    def make(args):
        """Run make with output capture."""
        agent = get_agent()
        
        if not args:
            args = []
        
        console.print(f"[bold]Running: make {' '.join(args)}[/]\n")
        
        result = agent.make_wrapper.make(*args)
        
        if result.success:
            console.print("\n[bold green]✓ Build successful[/]")
        else:
            console.print("\n[bold red]✗ Build failed[/]")
            
            if result.errors:
                console.print("\n[bold]Errors:[/]")
                for error in result.errors[:5]:
                    console.print(f"  {error['file']}:{error['line']}: {error['message']}")
            
            if result.panics:
                console.print("\n[bold]Panics:[/]")
                for panic in result.panics:
                    console.print(f"  panic: {panic['message']}")

    @cli.command()
    @click.argument('query')
    def help(query):
        """Get pedagogical help for a question."""
        agent = get_agent()
        
        console.print(f"\n[bold]Query:[/] {query}\n")
        
        with console.status("[bold blue]Analyzing..."):
            response = agent.process_query(query)
        
        console.print(Panel(
            response,
            title="[bold green]Response[/]",
            border_style="green"
        ))

    @cli.command()
    def status():
        """Show current task status."""
        agent = get_agent()
        
        task_status = agent.task_tracker.get_status_summary()
        session_status = agent.session_manager.get_summary()
        
        table = Table(title="Agent Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Session ID", session_status['session_id'])
        table.add_row("Lab", task_status.get('lab', 'Not started'))
        table.add_row("State", task_status.get('state', 'N/A'))
        table.add_row("Progress", task_status.get('progress', '0%'))
        table.add_row("Blockers", str(task_status.get('blockers', 0)))
        table.add_row("Queries", str(session_status.get('queries_count', 0)))
        table.add_row("Avg Confidence", f"{session_status.get('avg_confidence', 0):.2f}")
        
        console.print(table)

    @cli.command()
    def undo():
        """Undo the last change."""
        agent = get_agent()
        
        snapshot_id = agent.shadow_workspace.undo_last_change()
        
        if snapshot_id:
            console.print(f"[green]✓ Restored snapshot: {snapshot_id}[/]")
        else:
            console.print("[yellow]No snapshots to undo[/]")

    @cli.command()
    @click.argument('snapshot_id')
    def restore(snapshot_id):
        """Restore a specific snapshot."""
        agent = get_agent()
        
        success = agent.shadow_workspace.restore_snapshot(snapshot_id)
        
        if success:
            console.print(f"[green]✓ Restored: {snapshot_id}[/]")
        else:
            console.print(f"[red]Failed to restore: {snapshot_id}[/]")

    @cli.command()
    def snapshots():
        """List available snapshots."""
        agent = get_agent()
        
        snapshots = agent.shadow_workspace.list_snapshots()
        
        if not snapshots:
            console.print("[yellow]No snapshots available[/]")
            return
        
        table = Table(title="Snapshots")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Created")
        table.add_column("Size")
        table.add_column("In Stack")
        
        for s in snapshots[:10]:
            table.add_row(
                s['id'][:30],
                "auto" if s['auto'] else "manual",
                s['created'][:19],
                f"{s['size_bytes'] / 1024:.1f}KB",
                "✓" if s['in_undo_stack'] else ""
            )
        
        console.print(table)

    @cli.command()
    def debug():
        """Show debug information."""
        agent = get_agent()
        
        console.print(Panel.fit("[bold]Debug Information[/]"))
        
        # CPG stats
        cpg_stats = agent.cpg_builder.get_stats()
        console.print("\n[bold]CPG Stats:[/]")
        console.print(f"  Nodes: {cpg_stats['total_nodes']}")
        console.print(f"  Edges: {cpg_stats['total_edges']}")
        console.print(f"  Files: {cpg_stats['files']}")
        
        # Vector store stats
        vs_stats = agent.vector_store.get_stats()
        console.print("\n[bold]Vector Store Stats:[/]")
        console.print(f"  Status: {vs_stats['status']}")
        console.print(f"  Chunks: {vs_stats.get('chunk_count', 0)}")
        
        # LLM status
        console.print("\n[bold]LLM Status:[/]")
        console.print(f"  Provider: {agent.llm_client.provider}")
        console.print(f"  Model: {agent.llm_client.model}")
        console.print(f"  Available: {agent.llm_client.is_available()}")

    @cli.command()
    @click.option('--apply', is_flag=True, help='Apply fix after approval')
    def fix(apply):
        """Analyze errors and propose fixes with consent."""
        agent = get_agent()
        console = Console()
        
        console.print("\n[bold cyan]🔍 Analyzing for fixes...[/bold cyan]\n")
        session = agent.session_manager.get_context()
        
        if not session.last_build:
            console.print("[yellow]No build errors found. Run ./agent make first.[/yellow]")
            return
        
        has_errors = session.last_build.errors and len(session.last_build.errors) > 0
        has_panics = session.last_build.panics and len(session.last_build.panics) > 0
        
        if not has_errors and not has_panics:
            console.print("[green]✓ No errors in last build![/green]")
            return
        
        errors_text = ""
        if has_errors:
            errors_text += "ERRORS:\n"
            for e in session.last_build.errors[:5]:
                errors_text += f"  {e.get('file','?')}:{e.get('line','?')}: {e.get('message','?')}\n"
        if has_panics:
            errors_text += "PANICS:\n"
            for p in session.last_build.panics[:3]:
                errors_text += f"  panic: {p.get('message','?')}\n"
        
        console.print(Panel(errors_text, title="Issues", border_style="red"))
        
        with console.status("[green]Analyzing..."):
            from llm_client import LLMClient
            client = LLMClient(provider="groq")
            if not client.is_available():
                console.print("[red]LLM not available[/red]")
                return
            
            resp = client.generate(
                prompt=f"Analyze and propose fix:\n{errors_text}\n\nFormat: FILE, LINE, ISSUE, FIX, RISK, EXPLANATION",
                system_prompt="You are an xv6 expert. Provide specific fixes."
            )
        
        if resp.success:
            console.print(Panel(resp.content, title="💡 Proposed Fix", border_style="green"))
            console.print("\n[Y] Apply  [n] Skip  [e] Explain more")
            choice = click.prompt("Choice", default="n")
            
            if choice.lower() == 'e':
                exp = client.generate(f"Explain why this fix works:\n{resp.content}", "Explain xv6 concepts.")
                console.print(Panel(exp.content, title="Explanation", border_style="blue"))
                choice = click.prompt("Apply now?", default="n")
            
            if choice.lower() == 'y':
                console.print("[dim]Creating snapshot...[/dim]")
                try:
                    sid = agent.shadow_workspace.snapshot_before_patch("pre_fix")
                    console.print(f"[green]✓ Snapshot: {sid}[/green]")
                except: pass
                console.print("[yellow]Apply changes manually, then run ./agent make[/yellow]")
        else:
            console.print(f"[red]Error: {resp.error}[/red]")

    @cli.command()
    def watch():
        """Start file watcher daemon."""
        console = Console()
        console.print("[cyan]🔍 Watching for changes...[/cyan]")
        from file_sentinel import DebouncedHandler
        from watchdog.observers import Observer
        import os, time
        
        def on_change(f): console.print(f"[dim]Changed: {f}[/dim]")
        
        handler = DebouncedHandler(on_change, 200, watch_extensions=['.c','.h'])
        obs = Observer()
        obs.schedule(handler, os.getcwd(), recursive=True)
        obs.start()
        console.print("[green]✓ Watching. Ctrl+C to stop[/green]")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            obs.stop()
        obs.join()

    def main():
        """Entry point."""
        cli()

else:
    def main():
        print("Rich/Click not installed")

if __name__ == "__main__":
    main()
