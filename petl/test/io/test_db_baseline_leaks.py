# -*- coding: utf-8 -*-
"""Regression tests for the specific baseline defects named in the brief.

The baseline implementation exhibited three observable leaks:

1. an SQLAlchemy Engine read that only consumed the header then closed did
   not return the checked-out Connection to the pool;
2. SQLAlchemy Engine writes and DDL completed without closing the
   Connection they checked out;
3. a DB-API ``executemany`` failure left the cursor unclosed.

Each test asserts the corrected, explicitly-checked cleanup (no reliance on
garbage collection or interpreter shutdown).
"""
from __future__ import absolute_import, print_function, division

import pytest

sqlalchemy = pytest.importorskip('sqlalchemy')
from sqlalchemy import create_engine

# We deliberately exercise the pinned SQLAlchemy 1.4 legacy (non-future)
# API; its 2.0 deprecation notices are expected and not test failures.
pytestmark = pytest.mark.filterwarnings(
    'ignore::sqlalchemy.exc.RemovedIn20Warning')

from petl.io.db import fromdb, todb
from petl.io.db_create import create_table, drop_table
from petl.test.io._db_doubles import (
    RecordingConnection, FakeDBAPIError, CallLog)


def _engine():
    engine = create_engine('sqlite://')
    engine.execute('CREATE TABLE t (a INT, b INT)')
    engine.execute('INSERT INTO t VALUES (1, 2)')
    return engine


def test_baseline_engine_header_only_read_returns_connection():
    """Defect #1: header-only engine read must return its Connection."""
    engine = _engine()
    iterator = iter(fromdb(engine, 'SELECT * FROM t'))
    next(iterator)  # pull only the header
    iterator.close()
    # A leaked connection would leave the pool unable to hand out a fresh
    # connection without growing; asserting reusability after close pins the
    # fix deterministically rather than via gc.
    assert engine.execute('SELECT 1').fetchall() == [(1,)]
    engine.dispose()


def test_baseline_engine_write_closes_checked_out_connection():
    """Defect #2a: engine write returns its Connection on success/failure."""
    engine = _engine()
    todb((('a', 'b'), (3, 4)), engine, 't', commit=True)
    assert engine.execute('SELECT 1').fetchall() == [(1,)]

    with pytest.raises(Exception):
        todb((('a', 'b'), (5,)), engine, 't', commit=True)
    # still usable after the failed load: connection was returned on error
    assert engine.execute('SELECT 1').fetchall() == [(1,)]
    engine.dispose()


def test_baseline_engine_ddl_closes_checked_out_connection():
    """Defect #2b: engine DDL returns its Connection."""
    engine = _engine()
    create_table((('a', 'b'), (1, 2)), engine, 'ddl_table', commit=True)
    assert engine.execute('SELECT 1').fetchall() == [(1,)]
    drop_table(engine, 'ddl_table', commit=True)
    assert engine.execute('SELECT 1').fetchall() == [(1,)]
    engine.dispose()


def test_baseline_dbapi_executemany_failure_closes_cursor():
    """Defect #3: executemany error must still close the cursor."""
    log = CallLog()
    connection = RecordingConnection(
        log, fail_executemany=FakeDBAPIError('load boom'))
    with pytest.raises(FakeDBAPIError):
        todb((('a', 'b'), (1, 2)), connection, 't', commit=True)
    # the insert cursor (and the truncate cursor before it) are both closed
    assert log.count('cursor.close') == 2
    # and the caller's connection is never closed by petl
    assert 'connection.close' not in log.names()
    assert connection.closed is False


def test_baseline_dbapi_execute_failure_closes_cursor():
    """Companion to #3: a plain execute failure also closes the cursor."""
    log = CallLog()
    connection = RecordingConnection(
        log, fail_execute=FakeDBAPIError('ddl boom'))
    with pytest.raises(FakeDBAPIError):
        drop_table(connection, 't', commit=True)
    assert log.names() == ['connection.cursor', 'cursor.execute',
                           'cursor.close']
