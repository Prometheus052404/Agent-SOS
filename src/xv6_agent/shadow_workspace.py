"""
Shadow Workspace Module - Snapshot-only rollback system for Xv6 agent.

Implements (per critical updates - NO git stash):
- SnapshotManager with 5-level LIFO undo stack
- Zip compression for kernel/*.c and *.h files
- Atomic file writes (temp-file-then-rename pattern)
- Optional Fernet encryption for shared environments
"""

import os
import json
import time
import shutil
import zipfile
import logging
import hashlib
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class SnapshotManager:
    """
    Manages file snapshots for rollback capability.
    Uses zip-based snapshots with a LIFO undo stack.
    """

    def __init__(
        self,
        workspace_root: str,
        source_dir: str = ".",
        max_undo_depth: int = 5,
        encryption_key: Optional[str] = None
    ):
        self.workspace_root = Path(workspace_root)
        self.source_dir = Path(source_dir)
        self.max_undo_depth = max_undo_depth
        self.encryption_key = encryption_key
        
        # Setup directories
        self.snapshots_dir = self.workspace_root / "snapshots"
        self.auto_dir = self.snapshots_dir / "auto"
        self.manual_dir = self.snapshots_dir / "manual"
        self.stack_file = self.workspace_root / "undo_stack.json"
        
        # Ensure directories exist
        self.auto_dir.mkdir(parents=True, exist_ok=True)
        self.manual_dir.mkdir(parents=True, exist_ok=True)
        
        # Load undo stack
        self.undo_stack = self._load_stack()
        
        # Setup encryption if provided
        self.cipher = None
        if encryption_key:
            try:
                from cryptography.fernet import Fernet
                key = base64.urlsafe_b64encode(
                    hashlib.sha256(encryption_key.encode()).digest()
                )
                self.cipher = Fernet(key)
            except ImportError:
                logger.warning("cryptography not installed, encryption disabled")

    def _load_stack(self) -> List[str]:
        """Load the undo stack from disk."""
        if self.stack_file.exists():
            try:
                with open(self.stack_file, 'r') as f:
                    data = json.load(f)
                    return data.get('stack', [])
            except (json.JSONDecodeError, KeyError):
                pass
        return []

    def _save_stack(self):
        """Save the undo stack to disk."""
        data = {
            'stack': self.undo_stack,
            'max_depth': self.max_undo_depth,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.stack_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_snapshot(
        self,
        reason: str,
        auto: bool = True,
        patterns: Optional[List[str]] = None
    ) -> str:
        """
        Create a zip snapshot of source files.
        
        Args:
            reason: Description of why snapshot was created
            auto: Whether this is an automatic snapshot
            patterns: File patterns to include (default: *.c, *.h)
        
        Returns:
            Snapshot ID
        """
        patterns = patterns or ['*.c', '*.h']
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Sanitize reason for filename
        safe_reason = "".join(c if c.isalnum() or c in '-_' else '_' for c in reason)
        snapshot_id = f"{timestamp}_{safe_reason}"
        
        target_dir = self.auto_dir if auto else self.manual_dir
        zip_path = target_dir / f"{snapshot_id}.zip"
        
        logger.info(f"Creating snapshot: {snapshot_id}")
        
        # Create zip archive
        files_added = 0
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for pattern in patterns:
                for filepath in self.source_dir.rglob(pattern):
                    # Skip agent directory
                    if '.xv6_agent' in str(filepath):
                        continue
                    
                    relative_path = filepath.relative_to(self.source_dir)
                    try:
                        zf.write(filepath, relative_path)
                        files_added += 1
                    except (FileNotFoundError, PermissionError) as e:
                        logger.warning(f"Couldn't add {filepath}: {e}")
        
        # Encrypt if enabled
        if self.cipher:
            self._encrypt_file(zip_path)
        
        # Add to undo stack (auto snapshots only)
        if auto:
            self.undo_stack.append(snapshot_id)
            
            # Maintain max depth
            while len(self.undo_stack) > self.max_undo_depth:
                old_id = self.undo_stack.pop(0)
                self._cleanup_snapshot(old_id)
            
            self._save_stack()
        
        logger.info(f"Snapshot created: {snapshot_id} ({files_added} files)")
        return snapshot_id

    def undo(self) -> Optional[str]:
        """
        Rollback to the most recent snapshot.
        
        Returns:
            Snapshot ID that was restored, or None if no snapshots
        """
        if not self.undo_stack:
            logger.warning("No snapshots to undo")
            return None
        
        snapshot_id = self.undo_stack.pop()
        zip_path = self.auto_dir / f"{snapshot_id}.zip"
        
        if not zip_path.exists():
            logger.error(f"Snapshot not found: {snapshot_id}")
            self._save_stack()
            return None
        
        logger.info(f"Restoring snapshot: {snapshot_id}")
        
        # Decrypt if needed
        if self.cipher:
            self._decrypt_file(zip_path)
        
        # Extract files
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.source_dir)
            
            logger.info(f"Restored snapshot: {snapshot_id}")
            self._save_stack()
            return snapshot_id
            
        except zipfile.BadZipFile as e:
            logger.error(f"Corrupted snapshot: {e}")
            return None

    def restore(self, snapshot_id: str) -> bool:
        """
        Restore a specific snapshot by ID.
        
        Args:
            snapshot_id: The snapshot to restore
            
        Returns:
            True if successful
        """
        # Check both auto and manual directories
        zip_path = self.auto_dir / f"{snapshot_id}.zip"
        if not zip_path.exists():
            zip_path = self.manual_dir / f"{snapshot_id}.zip"
        
        if not zip_path.exists():
            logger.error(f"Snapshot not found: {snapshot_id}")
            return False
        
        logger.info(f"Restoring snapshot: {snapshot_id}")
        
        # Decrypt if needed
        if self.cipher:
            self._decrypt_file(zip_path)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.source_dir)
            
            logger.info(f"Restored snapshot: {snapshot_id}")
            return True
            
        except zipfile.BadZipFile as e:
            logger.error(f"Corrupted snapshot: {e}")
            return False

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots."""
        snapshots = []
        
        for dir_path, auto in [(self.auto_dir, True), (self.manual_dir, False)]:
            for zip_path in dir_path.glob("*.zip"):
                stat = zip_path.stat()
                snapshots.append({
                    'id': zip_path.stem,
                    'auto': auto,
                    'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'size_bytes': stat.st_size,
                    'in_undo_stack': zip_path.stem in self.undo_stack
                })
        
        # Sort by creation time (newest first)
        snapshots.sort(key=lambda x: x['created'], reverse=True)
        return snapshots

    def _cleanup_snapshot(self, snapshot_id: str):
        """Remove an old snapshot."""
        zip_path = self.auto_dir / f"{snapshot_id}.zip"
        if zip_path.exists():
            try:
                zip_path.unlink()
                logger.debug(f"Cleaned up snapshot: {snapshot_id}")
            except OSError as e:
                logger.warning(f"Couldn't delete {snapshot_id}: {e}")

    def _encrypt_file(self, filepath: Path):
        """Encrypt a file in place."""
        if not self.cipher:
            return
        
        with open(filepath, 'rb') as f:
            plaintext = f.read()
        
        ciphertext = self.cipher.encrypt(plaintext)
        
        with open(filepath, 'wb') as f:
            f.write(ciphertext)

    def _decrypt_file(self, filepath: Path):
        """Decrypt a file in place."""
        if not self.cipher:
            return
        
        with open(filepath, 'rb') as f:
            ciphertext = f.read()
        
        try:
            plaintext = self.cipher.decrypt(ciphertext)
            
            with open(filepath, 'wb') as f:
                f.write(plaintext)
        except Exception:
            # File might not be encrypted
            pass


class AtomicWriter:
    """
    Atomic file writing using temp-file-then-rename pattern.
    Provides POSIX guarantee of atomic updates.
    """

    def __init__(self, temp_suffix: str = ".agent_tmp", use_fsync: bool = True):
        self.temp_suffix = temp_suffix
        self.use_fsync = use_fsync

    def write(self, filepath: str, content: str) -> bool:
        """
        Write content to file atomically.
        
        Args:
            filepath: Target file path
            content: Content to write
            
        Returns:
            True if successful
        """
        filepath = Path(filepath)
        temp_path = filepath.parent / f"{filepath.name}{self.temp_suffix}_{os.getpid()}"
        
        try:
            # Write to temp file
            with open(temp_path, 'w') as f:
                f.write(content)
                f.flush()
                
                if self.use_fsync:
                    os.fsync(f.fileno())
            
            # Atomic rename (POSIX guarantee)
            os.rename(temp_path, filepath)
            
            logger.debug(f"Atomic write completed: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Atomic write failed for {filepath}: {e}")
            
            # Cleanup temp file
            try:
                temp_path.unlink()
            except OSError:
                pass
            
            return False

    def write_bytes(self, filepath: str, content: bytes) -> bool:
        """Write binary content to file atomically."""
        filepath = Path(filepath)
        temp_path = filepath.parent / f"{filepath.name}{self.temp_suffix}_{os.getpid()}"
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(content)
                f.flush()
                
                if self.use_fsync:
                    os.fsync(f.fileno())
            
            os.rename(temp_path, filepath)
            return True
            
        except Exception as e:
            logger.error(f"Atomic write failed for {filepath}: {e}")
            
            try:
                temp_path.unlink()
            except OSError:
                pass
            
            return False


class ShadowWorkspace:
    """
    Combined shadow workspace manager.
    Handles snapshots, atomic writes, and file mirroring.
    """

    def __init__(
        self,
        workspace_root: str = ".xv6_agent",
        source_dir: str = ".",
        config: Optional[Dict[str, Any]] = None
    ):
        self.workspace_root = Path(workspace_root)
        self.source_dir = Path(source_dir)
        self.config = config or {}
        
        # Initialize components
        self.snapshot_manager = SnapshotManager(
            workspace_root=str(self.workspace_root),
            source_dir=str(self.source_dir),
            max_undo_depth=self.config.get('max_undo_depth', 5),
            encryption_key=self.config.get('encryption_key')
        )
        
        self.atomic_writer = AtomicWriter(
            temp_suffix=self.config.get('temp_suffix', '.agent_tmp'),
            use_fsync=self.config.get('use_fsync', True)
        )
        
        # Current reference directory for diffing
        self.current_ref = self.workspace_root / "current_ref"
        self.current_ref.mkdir(parents=True, exist_ok=True)

    def snapshot_before_patch(self, reason: str = "pre_patch") -> str:
        """Create a snapshot before applying a patch."""
        return self.snapshot_manager.create_snapshot(reason, auto=True)

    def undo_last_change(self) -> Optional[str]:
        """Undo the last change."""
        return self.snapshot_manager.undo()

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore a specific snapshot."""
        return self.snapshot_manager.restore(snapshot_id)

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available snapshots."""
        return self.snapshot_manager.list_snapshots()

    def write_file_atomic(self, filepath: str, content: str) -> bool:
        """Write a file atomically."""
        return self.atomic_writer.write(filepath, content)

    def update_current_ref(self, filepath: str):
        """Update the current reference copy of a file."""
        src = Path(filepath)
        if not src.exists():
            return
        
        # Compute relative path
        try:
            rel_path = src.relative_to(self.source_dir)
        except ValueError:
            rel_path = src.name
        
        dest = self.current_ref / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(src, dest)

    def get_reference_content(self, filepath: str) -> Optional[str]:
        """Get the reference content for a file."""
        src = Path(filepath)
        
        try:
            rel_path = src.relative_to(self.source_dir)
        except ValueError:
            rel_path = src.name
        
        ref_path = self.current_ref / rel_path
        
        if ref_path.exists():
            return ref_path.read_text()
        
        return None


if __name__ == "__main__":
    # Test the shadow workspace
    logging.basicConfig(level=logging.DEBUG)
    
    workspace = ShadowWorkspace()
    
    # Create a test snapshot
    snapshot_id = workspace.snapshot_before_patch("test_snapshot")
    print(f"Created snapshot: {snapshot_id}")
    
    # List snapshots
    snapshots = workspace.list_snapshots()
    print(f"Available snapshots: {len(snapshots)}")
    for s in snapshots:
        print(f"  - {s['id']} (auto={s['auto']})")
