"""
Security Module - Privacy and security controls for Xv6 agent.

Implements:
- Snapshot encryption (optional Fernet)
- User consent tracking and management
- Code preview before LLM API calls
- Minimize code context option
- Audit logging for sensitive operations
"""

import os
import json
import hashlib
import base64
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import cryptography
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False


@dataclass
class ConsentRecord:
    """Record of user consent."""
    timestamp: str
    consent_given: bool
    code_preview_shown: str
    scope: str  # 'session' or 'permanent'


class EncryptionManager:
    """Manages encryption for sensitive data."""

    def __init__(self, key: Optional[str] = None):
        self.key = key
        self.cipher = None
        
        if key and ENCRYPTION_AVAILABLE:
            # Derive key from passphrase
            derived_key = base64.urlsafe_b64encode(
                hashlib.sha256(key.encode()).digest()
            )
            self.cipher = Fernet(derived_key)

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data if encryption is available."""
        if self.cipher:
            return self.cipher.encrypt(data)
        return data

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data if encryption is available."""
        if self.cipher:
            try:
                return self.cipher.decrypt(data)
            except Exception as e:
                logger.warning(f"Decryption failed: {e}")
        return data

    def is_available(self) -> bool:
        """Check if encryption is available."""
        return self.cipher is not None


class ConsentManager:
    """Manages user consent for LLM API calls."""

    def __init__(self, consent_file: str = ".xv6_agent/consent.json"):
        self.consent_file = Path(consent_file)
        self.consent_records: List[ConsentRecord] = []
        self.current_consent = False
        self._load_consent()

    def _load_consent(self):
        """Load consent from file."""
        if self.consent_file.exists():
            try:
                with open(self.consent_file, 'r') as f:
                    data = json.load(f)
                    self.current_consent = data.get('consent', False)
                    
                    for record in data.get('records', []):
                        self.consent_records.append(ConsentRecord(
                            timestamp=record['timestamp'],
                            consent_given=record['consent_given'],
                            code_preview_shown=record.get('code_preview_shown', ''),
                            scope=record.get('scope', 'session')
                        ))
            except Exception as e:
                logger.warning(f"Failed to load consent: {e}")

    def _save_consent(self):
        """Save consent to file."""
        self.consent_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'consent': self.current_consent,
            'records': [
                {
                    'timestamp': r.timestamp,
                    'consent_given': r.consent_given,
                    'code_preview_shown': r.code_preview_shown,
                    'scope': r.scope
                }
                for r in self.consent_records[-10:]  # Keep last 10 records
            ]
        }
        
        with open(self.consent_file, 'w') as f:
            json.dump(data, f, indent=2)

    def request_consent(
        self,
        code_preview: str,
        scope: str = 'session'
    ) -> bool:
        """
        Request user consent before sending code to LLM.
        
        Args:
            code_preview: Preview of code to be sent
            scope: 'session' or 'permanent'
            
        Returns:
            True if consent given
        """
        if self.current_consent:
            return True
        
        print("\n" + "=" * 60)
        print("⚠️  ABOUT TO SEND CODE TO LLM API")
        print("=" * 60)
        print("\nThe following code snippet will be sent:")
        print("-" * 60)
        truncated = code_preview[:500]
        if len(code_preview) > 500:
            truncated += f"\n... [{len(code_preview) - 500} more characters]"
        print(truncated)
        print("-" * 60)
        
        try:
            response = input(
                f"\nProceed with API call ({scope} consent)? [Y/n]: "
            ).strip().lower()
            
            consent = response in ('', 'y', 'yes')
            
            # Record consent
            self.consent_records.append(ConsentRecord(
                timestamp=datetime.now().isoformat(),
                consent_given=consent,
                code_preview_shown=truncated[:200],
                scope=scope
            ))
            
            if consent and scope == 'permanent':
                self.current_consent = True
                print("✓ Permanent consent recorded. Future calls won't prompt.")
            elif consent:
                print("✓ Session consent recorded.")
            
            self._save_consent()
            return consent
            
        except (EOFError, KeyboardInterrupt):
            print("\n✗ Consent denied.")
            return False

    def has_consent(self) -> bool:
        """Check if consent is currently given."""
        return self.current_consent

    def revoke_consent(self):
        """Revoke all consent."""
        self.current_consent = False
        self._save_consent()
        logger.info("Consent revoked")


class CodeMinimizer:
    """Minimizes code sent to LLM for privacy."""

    def __init__(self, max_lines: int = 20, max_chars: int = 2000):
        self.max_lines = max_lines
        self.max_chars = max_chars

    def minimize(
        self,
        code: str,
        focus_lines: Optional[List[int]] = None
    ) -> str:
        """
        Minimize code while preserving relevant context.
        
        Args:
            code: Full code content
            focus_lines: Line numbers to focus on (1-indexed)
            
        Returns:
            Minimized code with context
        """
        lines = code.split('\n')
        
        if focus_lines:
            # Extract context around focus lines
            result_lines = []
            focus_set = set(focus_lines)
            
            # Add lines and context
            for i, line in enumerate(lines, 1):
                if any(abs(i - f) <= 3 for f in focus_set):
                    result_lines.append(f"{i:4d}: {line}")
            
            # Add ellipsis markers
            minimized = []
            last_num = 0
            for line in result_lines:
                current_num = int(line[:4])
                if current_num > last_num + 1 and last_num > 0:
                    minimized.append("     ...")
                minimized.append(line)
                last_num = current_num
            
            return '\n'.join(minimized[:self.max_lines])
        
        # No focus lines - just truncate
        if len(lines) > self.max_lines:
            half = self.max_lines // 2
            result = lines[:half] + ['...'] + lines[-half:]
            return '\n'.join(result)
        
        return code[:self.max_chars]


class AuditLogger:
    """Logs sensitive operations for audit purposes."""

    def __init__(self, log_file: str = ".xv6_agent/logs/audit.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        sensitive: bool = False
    ):
        """Log an audit event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'sensitive': sensitive,
            **details
        }
        
        # Redact sensitive info
        if sensitive and 'code' in event:
            event['code'] = f"[REDACTED - {len(event['code'])} chars]"
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")

    def log_api_call(
        self,
        provider: str,
        prompt_size: int,
        response_size: int
    ):
        """Log an LLM API call."""
        self.log_event('api_call', {
            'provider': provider,
            'prompt_size': prompt_size,
            'response_size': response_size
        })

    def log_consent(self, consent_given: bool, scope: str):
        """Log a consent decision."""
        self.log_event('consent', {
            'consent_given': consent_given,
            'scope': scope
        })


class SecurityManager:
    """Unified security management."""

    def __init__(
        self,
        workspace_dir: str = ".xv6_agent",
        encryption_key: Optional[str] = None
    ):
        self.workspace_dir = Path(workspace_dir)
        
        self.encryption = EncryptionManager(encryption_key)
        self.consent = ConsentManager(str(self.workspace_dir / "consent.json"))
        self.minimizer = CodeMinimizer()
        self.audit = AuditLogger(str(self.workspace_dir / "logs/audit.jsonl"))

    def check_api_consent(self, code_preview: str) -> bool:
        """Check consent before API call."""
        if self.consent.has_consent():
            return True
        
        return self.consent.request_consent(code_preview, scope='permanent')

    def prepare_code_for_api(
        self,
        code: str,
        focus_lines: Optional[List[int]] = None
    ) -> str:
        """Prepare code for sending to API (minimized)."""
        return self.minimizer.minimize(code, focus_lines)

    def encrypt_sensitive(self, data: bytes) -> bytes:
        """Encrypt sensitive data."""
        return self.encryption.encrypt(data)

    def decrypt_sensitive(self, data: bytes) -> bytes:
        """Decrypt sensitive data."""
        return self.encryption.decrypt(data)


def create_security_manager(
    encryption_key: Optional[str] = None
) -> SecurityManager:
    """Create a security manager instance."""
    return SecurityManager(encryption_key=encryption_key)


if __name__ == "__main__":
    # Test the security module
    logging.basicConfig(level=logging.DEBUG)
    
    manager = SecurityManager()
    
    # Test code minimizer
    code = "\n".join([f"line {i}" for i in range(1, 51)])
    minimized = manager.prepare_code_for_api(code, focus_lines=[25])
    print("=== Minimized Code ===")
    print(minimized)
    
    # Test encryption
    if manager.encryption.is_available():
        print("\n=== Encryption Test ===")
        data = b"sensitive data"
        encrypted = manager.encrypt_sensitive(data)
        decrypted = manager.decrypt_sensitive(encrypted)
        print(f"Original: {data}")
        print(f"Encrypted: {encrypted[:50]}...")
        print(f"Decrypted: {decrypted}")
    else:
        print("\nEncryption not available (cryptography not installed)")
    
    # Log an audit event
    manager.audit.log_event('test', {'message': 'Security module test'})
    print("\n✓ Security module working!")
