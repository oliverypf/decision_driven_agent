"""Submit a decision-space packet from the source workspace."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
config = ROOT / "jev.config.json"
if config.is_file():
    os.environ.setdefault("DECISION_AGENT_CONFIG", str(config))

from decision_agent.direction import main

if __name__ == "__main__":
    main()
