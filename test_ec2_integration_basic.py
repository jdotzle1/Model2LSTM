#!/usr/bin/env python3
"""
Basic Integration Test for EC2 Pipeline Components (No XGBoost Required)

This script tests the integration between:
1. Weighted labeling system
2. Feature engineering
3. Pipeline validation

Run this locally to ensure core components work before EC2 deployment.
"""

import pandas as pd
import numpy as np
impo