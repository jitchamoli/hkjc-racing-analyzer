"""Backwards-compatible shim — import from dashboard.results_fetcher."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard.results_fetcher import *  # noqa: F401,F403

if __name__ == '__main__':
    import asyncio
    from dashboard.results_fetcher import main
    asyncio.run(main())
