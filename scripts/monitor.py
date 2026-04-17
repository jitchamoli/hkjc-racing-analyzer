#!/usr/bin/env python3
"""CLI: Race day odds monitor with auto-refresh dashboard.

Usage: python3 -m scripts.monitor '2026/04/16'
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.race_day import main

if __name__ == '__main__':
    main()
