"""Run the pinned static baselines on the six Phase 1 pilot cases, C000-C005."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.shared.static_baseline import main

if __name__ == '__main__':
    main(default_phase='pilot')
