#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Now import and run the main script
from model_comparison_study import main

if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.yaml"
    main(config_path) 