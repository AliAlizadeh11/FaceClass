#!/usr/bin/env python3
"""
Unit tests for emotion analysis service.
"""

import numpy as np
from typing import List
from src.services.emotion_analysis import analyze_emotions


def test_analyze_emotions_empty_list() -> None:
    preds: List[str] = analyze_emotions([])
    assert preds == []


def test_analyze_emotions_single_image() -> None:
    # Create a simple bright image likely to map to a non-neutral label in heuristic
    img = np.ones((32, 32, 3), dtype=np.uint8) * 200
    preds = analyze_emotions([img])
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert isinstance(preds[0], str)


