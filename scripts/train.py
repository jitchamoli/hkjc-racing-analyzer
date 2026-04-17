#!/usr/bin/env python3
"""CLI: Retrain ML model.

Usage: python3 -m scripts.train
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_ml_model import main

if __name__ == '__main__':
    main()
