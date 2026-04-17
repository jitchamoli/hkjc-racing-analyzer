#!/usr/bin/env python3
"""CLI: Scrape historical results.

Usage: python3 -m scripts.scrape
       python3 -m scripts.scrape --date 2026/04/09
       python3 -m scripts.scrape --rankings
       python3 -m scripts.scrape --days 90
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.scraper import main

if __name__ == '__main__':
    asyncio.run(main())
