"""
File Sentinel Module - Watchdog-based file monitoring for Xv6 agent.

Implements:
- DebouncedHandler with 200ms debounce timer
- Filter patterns for temp files (.swp, .tmp, etc.)
- 30-second reconciliation scan fallback
- Health monitoring with auto-restart
"""

import os
import time
import logging
import threading
from pathlib import Path
from collections import defaultdict
from typing import Callable, Optional, Set, Dict, Any
from threading import Timer, Lock

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

logger = logging.getLogger(__name__)


class DebouncedHandler(FileSystemEventHandler):
    """
    Watchdog event handler with debouncing to prevent multiple triggers
    from editor save operations (vim creates 3+ events per save).
    """

    def __init__(
        self,
        on_change_callback: Callable[[str], None],
        debounce_ms: int = 200,
        ignore_patterns: Optional[list] = None,
        watch_extensions: Optional[list] = None
    ):
        super().__init__()
        self.on_change_callback = on_change_callback
        self.debounce_ms = debounce_ms
        self.ignore_patterns = ignore_patterns or ['.swp', '.tmp', '.swx', '~', '.git']
        self.watch_extensions = watch_extensions or ['.c', '.h']
        
        self.pending_events: Dict[str, Timer] = {}
        self.lock = Lock()
        self.processed_files: Set[str] = set()

    def should_ignore(self, path: str) -> bool:
        """Check if file should be ignored based on patterns."""
        path_lower = path.lower()
        
        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if pattern in path_lower or path.endswith(pattern):
                return True
        
        # Check if it's in .xv6_agent directory
        if '.xv6_agent' in path:
            return True
        
        return False

    def should_process(self, path: str) -> bool:
        """Check if file should be processed based on extensions."""
        if self.should_ignore(path):
            return False
        
        # Check extensions
        for ext in self.watch_extensions:
            if path.endswith(ext):
                return True
        
        return False

    def on_modified(self, event):
        """Handle file modification events with debouncing."""
        if event.is_directory:
            return
        
        self._handle_event(event.src_path)

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        self._handle_event(event.src_path)

    def _handle_event(self, filepath: str):
        """Process a file event with debouncing."""
        if not self.should_process(filepath):
            logger.debug(f"Ignoring file: {filepath}")
            return

        with self.lock:
            # Cancel existing timer for this file
            if filepath in self.pending_events:
                self.pending_events[filepath].cancel()
                logger.debug(f"Debounce reset for: {filepath}")

            # Create new timer
            timer = Timer(
                self.debounce_ms / 1000.0,
                self._process_change,
                args=[filepath]
            )
            self.pending_events[filepath] = timer
            timer.start()
            logger.debug(f"Debounce timer started for: {filepath}")

    def _process_change(self, filepath: str):
        """Process the file change after debounce period."""
        with self.lock:
            if filepath in self.pending_events:
                del self.pending_events[filepath]

        logger.info(f"File changed: {filepath}")
        
        try:
            self.on_change_callback(filepath)
        except Exception as e:
            logger.error(f"Error processing file change {filepath}: {e}")

    def shutdown(self):
        """Cancel all pending timers."""
        with self.lock:
            for timer in self.pending_events.values():
                timer.cancel()
            self.pending_events.clear()


class FileSentinel:
    """
    Main file monitoring class with health monitoring and reconciliation.
    
    Features:
    - Watchdog-based real-time monitoring
    - 30-second reconciliation scan fallback
    - Health monitoring with auto-restart
    """

    def __init__(
        self,
        watch_path: str,
        on_change_callback: Callable[[str], None],
        config: Optional[Dict[str, Any]] = None
    ):
        self.watch_path = Path(watch_path).resolve()
        self.on_change_callback = on_change_callback
        self.config = config or {}
        
        # Configuration
        self.debounce_ms = self.config.get('debounce_ms', 200)
        self.reconciliation_interval = self.config.get('reconciliation_interval_sec', 30)
        self.ignore_patterns = self.config.get('ignore_patterns', ['.swp', '.tmp', '.swx', '~'])
        self.watch_extensions = self.config.get('watch_extensions', ['.c', '.h'])
        
        # State
        self.file_mtimes: Dict[str, float] = {}
        self.observer: Optional[Observer] = None
        self.handler: Optional[DebouncedHandler] = None
        self.reconciliation_thread: Optional[threading.Thread] = None
        self.running = False
        self.health_check_failures = 0
        self.max_health_failures = 3

    def start(self):
        """Start the file sentinel."""
        logger.info(f"Starting File Sentinel for: {self.watch_path}")
        
        self.running = True
        
        # Initialize mtime cache
        self._scan_files()
        
        # Create handler
        self.handler = DebouncedHandler(
            on_change_callback=self._on_file_change,
            debounce_ms=self.debounce_ms,
            ignore_patterns=self.ignore_patterns,
            watch_extensions=self.watch_extensions
        )
        
        # Start watchdog observer
        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.watch_path), recursive=True)
        self.observer.start()
        
        # Start reconciliation thread
        self.reconciliation_thread = threading.Thread(
            target=self._reconciliation_loop,
            daemon=True
        )
        self.reconciliation_thread.start()
        
        logger.info("File Sentinel started successfully")

    def stop(self):
        """Stop the file sentinel."""
        logger.info("Stopping File Sentinel")
        self.running = False
        
        if self.handler:
            self.handler.shutdown()
        
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)
        
        logger.info("File Sentinel stopped")

    def _on_file_change(self, filepath: str):
        """Handle a validated file change."""
        # Update mtime cache
        try:
            mtime = os.path.getmtime(filepath)
            self.file_mtimes[filepath] = mtime
        except FileNotFoundError:
            # File was deleted
            if filepath in self.file_mtimes:
                del self.file_mtimes[filepath]
        
        # Call the callback
        self.on_change_callback(filepath)

    def _scan_files(self):
        """Scan all watched files and cache their mtimes."""
        for ext in self.watch_extensions:
            for filepath in self.watch_path.rglob(f'*{ext}'):
                filepath_str = str(filepath)
                if not any(p in filepath_str for p in self.ignore_patterns):
                    try:
                        self.file_mtimes[filepath_str] = os.path.getmtime(filepath_str)
                    except (FileNotFoundError, PermissionError):
                        pass

    def _reconciliation_loop(self):
        """
        Periodic reconciliation scan to catch missed events.
        Runs every 30 seconds by default.
        """
        while self.running:
            time.sleep(self.reconciliation_interval)
            
            if not self.running:
                break
            
            logger.debug("Running reconciliation scan")
            
            try:
                self._reconciliation_scan()
                self.health_check_failures = 0
            except Exception as e:
                logger.error(f"Reconciliation scan error: {e}")
                self.health_check_failures += 1
                
                if self.health_check_failures >= self.max_health_failures:
                    logger.warning("Multiple health check failures, restarting observer")
                    self._restart_observer()

    def _reconciliation_scan(self):
        """Check for files that changed but weren't detected by watchdog."""
        changed_files = []
        
        for ext in self.watch_extensions:
            for filepath in self.watch_path.rglob(f'*{ext}'):
                filepath_str = str(filepath)
                
                if any(p in filepath_str for p in self.ignore_patterns):
                    continue
                
                try:
                    current_mtime = os.path.getmtime(filepath_str)
                    cached_mtime = self.file_mtimes.get(filepath_str)
                    
                    if cached_mtime is None or current_mtime > cached_mtime:
                        changed_files.append(filepath_str)
                        self.file_mtimes[filepath_str] = current_mtime
                except (FileNotFoundError, PermissionError):
                    continue
        
        # Report missed changes
        for filepath in changed_files:
            logger.info(f"Reconciliation detected change: {filepath}")
            try:
                self.on_change_callback(filepath)
            except Exception as e:
                logger.error(f"Error processing reconciliation change {filepath}: {e}")

    def _restart_observer(self):
        """Restart the watchdog observer after failures."""
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=5)
            
            self.observer = Observer()
            self.observer.schedule(self.handler, str(self.watch_path), recursive=True)
            self.observer.start()
            
            self.health_check_failures = 0
            logger.info("Observer restarted successfully")
        except Exception as e:
            logger.error(f"Failed to restart observer: {e}")

    def is_healthy(self) -> bool:
        """Check if the file sentinel is healthy."""
        return (
            self.running and
            self.observer is not None and
            self.observer.is_alive() and
            self.health_check_failures < self.max_health_failures
        )


# Convenience function for quick setup
def create_file_sentinel(
    watch_path: str,
    on_change_callback: Callable[[str], None],
    config: Optional[Dict[str, Any]] = None
) -> FileSentinel:
    """Create and return a configured FileSentinel instance."""
    sentinel = FileSentinel(watch_path, on_change_callback, config)
    return sentinel


if __name__ == "__main__":
    # Test the file sentinel
    logging.basicConfig(level=logging.DEBUG)
    
    def on_change(filepath):
        print(f"[CALLBACK] File changed: {filepath}")
    
    sentinel = create_file_sentinel(".", on_change)
    sentinel.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sentinel.stop()
