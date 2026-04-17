"""Backwards-compatible shim — import from dashboard.generator."""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard.generator import main

if __name__ == '__main__':
    asyncio.run(main())
