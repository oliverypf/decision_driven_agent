"""Submit a decision-space packet from the source workspace."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from decision_agent.direction import main

if __name__ == "__main__":
    main()
