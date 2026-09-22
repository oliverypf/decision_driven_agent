"""Bootstrap the repository-local Decision-Driven Agent hook on Windows."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bind the project hook to this checkout's configuration even when Codex runs
# the hook with a different working directory.
CONFIG = ROOT / "jev.config.json"
if CONFIG.is_file():
    os.environ.setdefault("DECISION_AGENT_CONFIG", str(CONFIG))

from decision_agent.codex_hook import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
