#!/usr/bin/env python
"""Launch the Streamlit dashboard.

Usage:
    python scripts/run_dashboard.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    dashboard_path = (
        Path(__file__).parent.parent / "risk_index" / "reporting" / "dashboard.py"
    )

    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)

    print("Starting Risk Regime Dashboard...")
    print("Press Ctrl+C to stop")
    print()

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
