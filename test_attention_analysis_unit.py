#!/usr/bin/env python3
"""
Unit tests for attention analysis service.
"""

from typing import Dict
from src.services.attention_analysis import analyze_attention


def test_analyze_attention_basic() -> None:
    result: Dict = analyze_attention({'yaw': 0, 'pitch': 0, 'roll': 0})
    assert isinstance(result, dict)
    assert result['is_attentive'] is True
    assert 'yaw' in result and 'pitch' in result and 'roll' in result


def test_analyze_attention_thresholds() -> None:
    # Exceed yaw threshold
    r1 = analyze_attention({'yaw': 30, 'pitch': 0, 'roll': 0})
    assert r1['is_attentive'] is False
    # Exceed pitch threshold
    r2 = analyze_attention({'yaw': 0, 'pitch': 25, 'roll': 0})
    assert r2['is_attentive'] is False


