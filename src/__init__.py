"""
Vietnamese Relational Knowledge Distillation: Qwen3-4B sang 0.6B
"""

import os
import sys

# Đảm bảo đường dẫn root luôn có trong sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from . import compat  # noqa: F401
