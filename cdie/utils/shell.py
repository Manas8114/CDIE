"""
CDIE v5 — Secure Shell Utilities
Provides safe wrappers for subprocess execution to prevent command injection.
"""

import logging
import platform
import subprocess

logger = logging.getLogger(__name__)


def safe_run(args: list[str], timeout: int = 30, capture_output: bool = True) -> subprocess.CompletedProcess[str] | None:
    """
    Safely execute a command with a list of arguments.
    Enforces shell=False to mitigate command injection.
    """
    try:
        # On Windows, some commands might still need shell=True if they are builtins,
        # but for security, we avoid it unless absolutely necessary.
        # We use the list-based API which is safer.
        result = subprocess.run(
            args,
            capture_output=capture_output,
            text=True,
            shell=False,  # Security Hardening: NO shell execution
            timeout=timeout,
        )
        return result
    except subprocess.TimeoutExpired:
        logger.error(f'Command timed out: {" ".join(args)}')
        return None
    except FileNotFoundError:
        logger.error(f'Command not found: {args[0]}')
        return None
    except Exception as e:
        logger.error(f'Unexpected error executing command {" ".join(args)}: {e}')
        return None


def detect_cpu_feature_linux(feature: str) -> bool:
    """Safely check /proc/cpuinfo for a specific feature on Linux."""
    if platform.system() != 'Linux':
        return False
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if 'flags' in line and feature.lower() in line.lower():
                    return True
        return False
    except Exception as e:
        logger.error(f'Error reading cpuinfo: {e}')
        return False


def detect_cpu_name_windows() -> str:
    """Safely get CPU name on Windows using wmic."""
    if platform.system() != 'Windows':
        return ''
    # Fixed command, no user input
    res = safe_run(['wmic', 'cpu', 'get', 'name'])
    if res and res.stdout:
        lines = res.stdout.strip().split('\n')
        if len(lines) > 1:
            return str(lines[1].strip())
    return ''
