"""
Command Interceptor Module - Non-blocking QEMU I/O for Xv6 agent.

Implements (per critical updates):
- Threading for stdout/stderr capture (NOT communicate())
- Real-time output streaming to user terminal
- Panic message parsing with regex
- Compiler error extraction

This is a CRITICAL component - using threading instead of communicate()
prevents deadlocks when QEMU panics.
"""

import os
import re
import sys
import time
import logging
import subprocess
import threading
from queue import Queue, Empty
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Result of a build/run operation."""
    success: bool
    return_code: int
    errors: List[Dict[str, Any]] = field(default_factory=list)
    panics: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    full_stdout: str = ""
    full_stderr: str = ""


class StreamReader(threading.Thread):
    """
    Non-blocking stream reader thread.
    Reads from a stream and puts output into a queue.
    """

    def __init__(
        self,
        stream,
        queue: Queue,
        prefix: str = "",
        echo: bool = True
    ):
        super().__init__(daemon=True)
        self.stream = stream
        self.queue = queue
        self.prefix = prefix
        self.echo = echo
        self.output_buffer = []

    def run(self):
        """Read from stream until EOF."""
        try:
            for line in iter(self.stream.readline, b''):
                if not line:
                    break
                
                try:
                    decoded = line.decode('utf-8', errors='replace')
                except AttributeError:
                    decoded = line
                
                self.output_buffer.append(decoded)
                self.queue.put((self.prefix, decoded))
                
                if self.echo:
                    print(decoded, end='', flush=True)
        except Exception as e:
            logger.error(f"Stream reader error: {e}")
        finally:
            self.stream.close()

    def get_output(self) -> str:
        """Get all captured output."""
        return ''.join(self.output_buffer)


class CommandInterceptor:
    """
    Intercepts and wraps commands for the Xv6 agent.
    Captures output while displaying it in real-time.
    """

    def __init__(self, on_result_callback: Optional[Callable[[BuildResult], None]] = None):
        self.on_result_callback = on_result_callback
        
        # Regex patterns
        self.error_pattern = re.compile(
            r'(\S+\.c):(\d+):(\d+):\s*error:\s*(.+)'
        )
        self.warning_pattern = re.compile(
            r'(\S+\.c):(\d+):(\d+):\s*warning:\s*(.+)'
        )
        self.panic_pattern = re.compile(
            r'panic:\s*(.+?)(?:\n|$)'
        )
        self.epc_pattern = re.compile(
            r'(?:epc|pc)\s*[:=]\s*(0x[0-9a-fA-F]+)'
        )
        self.backtrace_pattern = re.compile(
            r'(\S+)\+0x[0-9a-fA-F]+'
        )

    def run_make(
        self,
        args: List[str],
        cwd: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> BuildResult:
        """
        Run make with arguments and capture output.
        
        Args:
            args: Arguments to pass to make
            cwd: Working directory
            timeout: Timeout in seconds (None for no timeout)
            
        Returns:
            BuildResult with parsed output
        """
        cmd = ['make'] + args
        return self._run_command(cmd, cwd, timeout)

    def run_qemu(
        self,
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> BuildResult:
        """
        Run make qemu with non-blocking I/O.
        
        This is the critical path - we use threading to prevent deadlocks.
        """
        cmd = ['make', 'qemu'] + (args or [])
        return self._run_command(cmd, cwd, timeout, is_interactive=True)

    def _run_command(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
        is_interactive: bool = False
    ) -> BuildResult:
        """
        Run a command with non-blocking I/O.
        
        Uses threading pattern from blueprint critical updates to prevent
        blocking on QEMU panics.
        """
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Start process with unbuffered output
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                bufsize=0  # Unbuffered for real-time output
            )
            
            # Create output queues
            stdout_queue = Queue()
            stderr_queue = Queue()
            
            # Start reader threads
            stdout_reader = StreamReader(
                proc.stdout,
                stdout_queue,
                prefix="stdout",
                echo=True
            )
            stderr_reader = StreamReader(
                proc.stderr,
                stderr_queue,
                prefix="stderr",
                echo=True
            )
            
            stdout_reader.start()
            stderr_reader.start()
            
            # Wait for process with timeout
            try:
                return_code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Command timed out, terminating...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return_code = -1
            
            # Wait for readers to finish
            stdout_reader.join(timeout=2)
            stderr_reader.join(timeout=2)
            
            # Get captured output
            stdout_text = stdout_reader.get_output()
            stderr_text = stderr_reader.get_output()
            
            # Parse output
            result = self._parse_output(
                return_code,
                stdout_text,
                stderr_text
            )
            
            # Call callback if set
            if self.on_result_callback:
                self.on_result_callback(result)
            
            return result
            
        except FileNotFoundError:
            logger.error(f"Command not found: {cmd[0]}")
            return BuildResult(
                success=False,
                return_code=-1,
                full_stderr=f"Command not found: {cmd[0]}"
            )
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return BuildResult(
                success=False,
                return_code=-1,
                full_stderr=str(e)
            )

    def _parse_output(
        self,
        return_code: int,
        stdout: str,
        stderr: str
    ) -> BuildResult:
        """Parse command output to extract errors, warnings, and panics."""
        combined = stdout + stderr
        
        # Parse compiler errors
        errors = []
        for match in self.error_pattern.finditer(stderr):
            errors.append({
                'file': match.group(1),
                'line': int(match.group(2)),
                'column': int(match.group(3)),
                'message': match.group(4)
            })
        
        # Parse warnings
        warnings = []
        for match in self.warning_pattern.finditer(stderr):
            warnings.append({
                'file': match.group(1),
                'line': int(match.group(2)),
                'column': int(match.group(3)),
                'message': match.group(4)
            })
        
        # Parse panics
        panics = []
        panic_matches = self.panic_pattern.findall(combined)
        for panic_msg in panic_matches:
            panic_info = {'message': panic_msg.strip()}
            
            # Look for EPC near the panic
            epc_match = self.epc_pattern.search(combined)
            if epc_match:
                panic_info['epc'] = epc_match.group(1)
            
            # Look for backtrace
            backtrace = self.backtrace_pattern.findall(combined)
            if backtrace:
                panic_info['backtrace'] = backtrace[:10]  # Limit to 10 frames
            
            panics.append(panic_info)
        
        # Determine success
        success = return_code == 0 and len(errors) == 0 and len(panics) == 0
        
        return BuildResult(
            success=success,
            return_code=return_code,
            errors=errors,
            warnings=warnings,
            panics=panics,
            full_stdout=stdout,
            full_stderr=stderr
        )

    def run_with_timeout(
        self,
        cmd: List[str],
        timeout: int = 30,
        cwd: Optional[str] = None
    ) -> BuildResult:
        """Run a command with a timeout."""
        return self._run_command(cmd, cwd, timeout)


class AgentMakeWrapper:
    """
    Wrapper for the 'agent make' command.
    Provides a clean interface for building and running xv6.
    """

    def __init__(
        self,
        project_dir: str = ".",
        on_build_complete: Optional[Callable[[BuildResult], None]] = None
    ):
        self.project_dir = project_dir
        self.interceptor = CommandInterceptor(on_result_callback=on_build_complete)
        self.last_result: Optional[BuildResult] = None

    def make(self, *args) -> BuildResult:
        """Run make with arguments."""
        self.last_result = self.interceptor.run_make(
            list(args),
            cwd=self.project_dir
        )
        return self.last_result

    def make_clean(self) -> BuildResult:
        """Run make clean."""
        return self.make('clean')

    def make_qemu(self, timeout: Optional[int] = None) -> BuildResult:
        """Run make qemu."""
        self.last_result = self.interceptor.run_qemu(
            cwd=self.project_dir,
            timeout=timeout
        )
        return self.last_result

    def make_qemu_nox(self, timeout: Optional[int] = None) -> BuildResult:
        """Run make qemu-nox (no X display)."""
        self.last_result = self.interceptor.run_make(
            ['qemu-nox'],
            cwd=self.project_dir,
            timeout=timeout
        )
        return self.last_result

    def get_last_errors(self) -> List[Dict[str, Any]]:
        """Get errors from last build."""
        if self.last_result:
            return self.last_result.errors
        return []

    def get_last_panics(self) -> List[Dict[str, Any]]:
        """Get panics from last build."""
        if self.last_result:
            return self.last_result.panics
        return []

    def was_successful(self) -> bool:
        """Check if last build was successful."""
        return self.last_result is not None and self.last_result.success


def create_interceptor(
    on_result: Optional[Callable[[BuildResult], None]] = None
) -> CommandInterceptor:
    """Create a configured command interceptor."""
    return CommandInterceptor(on_result_callback=on_result)


if __name__ == "__main__":
    # Test the interceptor
    logging.basicConfig(level=logging.DEBUG)
    
    def on_result(result: BuildResult):
        print(f"\n=== Build Result ===")
        print(f"Success: {result.success}")
        print(f"Return code: {result.return_code}")
        print(f"Errors: {len(result.errors)}")
        print(f"Panics: {len(result.panics)}")
        
        for error in result.errors:
            print(f"  Error: {error['file']}:{error['line']}: {error['message']}")
        
        for panic in result.panics:
            print(f"  Panic: {panic['message']}")
    
    wrapper = AgentMakeWrapper(".", on_build_complete=on_result)
    
    # Test a simple make command
    print("Testing 'make' command...")
    result = wrapper.make()
