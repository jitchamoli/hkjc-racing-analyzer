#!/usr/bin/env python3
"""CLI: Fetch results and score predictions.

Usage: python3 -m scripts.fetch_results '2026/04/16'
       python3 -m scripts.fetch_results '2026/04/16' --auto-learn
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.results_fetcher import main

if __name__ == '__main__':
    asyncio.run(main())
