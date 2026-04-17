#!/usr/bin/env python3
"""CLI: Generate predictions and dashboard.

Usage: python3 -m scripts.generate '2026/04/16'
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.generator import main

if __name__ == '__main__':
    asyncio.run(main())
