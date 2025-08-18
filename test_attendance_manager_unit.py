#!/usr/bin/env python3
"""
Unit tests for attendance manager service.
"""

from src.services.attendance_manager import AttendanceManager


def test_add_and_summarize_presence() -> None:
    mgr = AttendanceManager()
    mgr.add_presence('s1', 'stu1', frames=30)
    mgr.add_presence('s1', 'stu1', frames=30)
    mgr.add_presence('s1', 'stu2', frames=60)
    summary = mgr.get_session_summary('s1', fps=30.0)
    assert summary['stu1'] == 2.0
    assert summary['stu2'] == 2.0


def test_get_all_returns_internal_map() -> None:
    mgr = AttendanceManager()
    mgr.add_presence('sX', 'stuA', frames=5)
    data = mgr.get_all()
    assert 'sX' in data
    assert data['sX']['stuA'] == 5


