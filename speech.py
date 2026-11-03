"""Non-blocking text-to-speech using the macOS `say` command (espeak on Linux)."""

import shutil
import subprocess


class Speaker:
    def __init__(self, enabled: bool = True):
        self._cmd = shutil.which("say") or shutil.which("espeak")
        self._enabled = enabled and self._cmd is not None
        self._proc: subprocess.Popen | None = None
        if enabled and self._cmd is None:
            print("No 'say' or 'espeak' found; speech disabled.")

    def say(self, text: str) -> None:
        """Speak text, interrupting anything still being spoken."""
        print(f"[speech] {text}")
        if not self._enabled:
            return
        self.stop()
        self._proc = subprocess.Popen([self._cmd, text])

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
