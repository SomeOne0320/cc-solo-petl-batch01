# -*- coding: utf-8 -*-
"""SQLAlchemy 1.4 legacy-API ownership tests against real local SQLite.

Covers Engine, borrowed Connection and Session across success, pre-existing
outer transactions and mid-operation errors, including the requirement that a
caller's outer transaction and its pre-written marker rows are never committed
or rolled back by petl and remain usable afterwards.
"""
from __future__ import absolute_import, print_function, division

import pytest

sqlalchemy = pytest.importorskip('sqlalchemy')
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# We deliberately exercise the pinned SQLAlchemy 1.4 legacy (non-future)
# API; its 2.0 deprecation notices are expected and not test failures.
pytestmark = pytest.mark.filterwarnings(
    'ignore::sqlalchemy.exc.RemovedIn20Warning')

from petl.io.db import fromdb, todb, appenddb
from petl.io.db_create import create_table, drop_table
from petl.io.db_adapters import (
    resolve_target, make_adapter, ResourceStack,
    SOURCE_SQLALCHEMY_ENGINE, SOURCE_SQLALCHEMY_CONNECTION,
    SOURCE_SQLALCHEMY_SESSION, SQLAlchemyTransactionScope,
    NoTransactionScope)


TBL = (('foo', 'bar'), ('a', 1), ('b', 2))


@pytest.fixture()
def engine(tmp_path):
    path = str(tmp_path / 'test.db')
    eng = create_engine('sqlite:///%s' % path)
    eng.execute('CREATE TABLE src (foo TEXT, bar INT)')
    eng.execute('CREATE TABLE marker (a INT)')
    return eng


def _count(engine, table):
    return engine.execute('SELECT count(*) FROM %s' % table).fetchone()[0]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_resolve_sqlalchemy_objects(engine):
    Session = sessionmaker(bind=engine)
    assert resolve_target(engine).source == SOURCE_SQLALCHEMY_ENGINE
    connection = engine.connect()
    try:
        assert resolve_target(connection).source == \
            SOURCE_SQLALCHEMY_CONNECTION
    finally:
        connection.close()
    session = Session()
    try:
        assert resolve_target(session).source == SOURCE_SQLALCHEMY_SESSION
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def test_engine_write_commits_and_returns_connection(engine):
    todb(TBL, engine, 'src', commit=True)
    # engine still usable and data persisted
    assert engine.execute('SELECT * FROM src').fetchall() == \
        [('a', 1), ('b', 2)]


def test_engine_write_failure_rolls_back_and_returns_connection(engine):
    engine.execute("INSERT INTO marker VALUES (1)")
    bad = (('foo', 'bar'), ('a',), ('b', 2))
    with pytest.raises(Exception):
        todb(bad, engine, 'src', commit=True)
    # the engine remains usable: the petl-checked-out connection was returned
    assert engine.execute('SELECT 1').fetchall() == [(1,)]
    # src never received partial data
    assert _count(engine, 'src') == 0


def test_engine_commit_false_creates_no_transaction_scope(engine):
    target = resolve_target(engine)
    scope = make_adapter(target, ResourceStack()) \
        .make_transaction_scope(False)
    assert isinstance(scope, NoTransactionScope)
    appenddb((('foo', 'bar'), ('z', 9)), engine, 'src', commit=False)
    # connection returned and engine still usable
    assert engine.execute('SELECT count(*) FROM src').fetchone()[0] == 1


def test_engine_ddl_returns_connection(engine):
    create_table(TBL, engine, 'new_table', commit=True)
    drop_table(engine, 'new_table', commit=True)
    assert engine.execute('SELECT 1').fetchall() == [(1,)]


def test_engine_read_returns_connection_and_supports_interleaving(engine):
    engine.execute("INSERT INTO src VALUES ('a',1),('b',2),('c',3)")
    view = fromdb(engine, 'SELECT * FROM src')
    i1, i2 = iter(view), iter(view)
    assert next(i1) == ('foo', 'bar')
    assert next(i1) == ('a', 1)
    assert next(i2) == ('foo', 'bar')
    i1.close()
    assert next(i2) == ('a', 1)
    assert list(i2) == [('b', 2), ('c', 3)]
    # engine still fully usable after both iterators ended
    assert engine.execute('SELECT 1').fetchall() == [(1,)]


def test_engine_read_failure_returns_connection(engine):
    iterator = iter(fromdb(engine, 'SELECT * FROM no_such_table'))
    with pytest.raises(Exception):
        next(iterator)
    assert engine.execute('SELECT 1').fetchall() == [(1,)]


def test_engine_is_never_disposed(engine):
    todb(TBL, engine, 'src', commit=False)
    # disposing would invalidate the engine; a fresh execute proves it is alive
    assert not engine.pool.status().startswith('disposed')
    assert engine.execute('SELECT 1').fetchall() == [(1,)]


# ---------------------------------------------------------------------------
# Borrowed Connection
# ---------------------------------------------------------------------------

def test_borrowed_connection_outer_transaction_marker_preserved(engine):
    connection = engine.connect()
    outer = connection.begin()
    connection.execute("INSERT INTO marker VALUES (42)")
    try:
        todb(TBL, connection, 'src', commit=True)
        # petl used a savepoint and must not finalise the outer transaction
        assert outer.is_active
        assert connection.in_transaction()
        assert connection.execute(
            'SELECT * FROM marker').fetchall() == [(42,)]
    finally:
        outer.rollback()
        connection.close()
    # caller's rollback discards both marker and petl's savepoint work
    assert _count(engine, 'marker') == 0
    assert _count(engine, 'src') == 0


def test_borrowed_connection_failure_keeps_outer_alive(engine):
    connection = engine.connect()
    outer = connection.begin()
    connection.execute("INSERT INTO marker VALUES (42)")
    try:
        with pytest.raises(Exception):
            todb((('foo', 'bar'), ('a',)), connection, 'src', commit=True)
        assert outer.is_active
        # marker untouched and connection still usable
        assert connection.execute(
            'SELECT * FROM marker').fetchall() == [(42,)]
    finally:
        outer.rollback()
        connection.close()
    assert _count(engine, 'marker') == 0


def test_borrowed_connection_no_outer_uses_own_scope(engine):
    connection = engine.connect()
    try:
        assert connection.in_transaction() is False
        todb(TBL, connection, 'src', commit=True)
        assert connection.in_transaction() is False
    finally:
        connection.close()
    assert _count(engine, 'src') == 2


def test_borrowed_connection_commit_false_not_managed(engine):
    connection = engine.connect()
    try:
        target = resolve_target(connection)
        scope = make_adapter(target, ResourceStack()) \
            .make_transaction_scope(False)
        assert isinstance(scope, NoTransactionScope)
        appenddb((('foo', 'bar'), ('z', 9)), connection, 'src',
                 commit=False)
        assert not connection.closed
    finally:
        connection.close()


def test_borrowed_connection_is_not_closed_by_petl(engine):
    connection = engine.connect()
    todb(TBL, connection, 'src', commit=True)
    assert not connection.closed
    connection.close()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def test_fresh_session_ownership_detected_before_autobegin(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        assert session.in_transaction() is False
        target = resolve_target(session)
        adapter = make_adapter(target, ResourceStack())
        adapter._acquire_connection()  # this call triggers the autobegin
        # ownership was captured from the pre-acquisition session state
        assert adapter._session_had_transaction is False
        todb(TBL, session, 'src', commit=True)
        # petl completed its savepoint scope but never the Session itself
        assert session.in_transaction() is True
        assert session.in_nested_transaction() is False
    finally:
        session.close()


def test_fresh_session_caller_can_commit(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        todb(TBL, session, 'src', commit=True)
        assert session.in_transaction() is True
        session.commit()  # caller remains in control
        assert session.in_transaction() is False
    finally:
        session.close()


def test_session_outer_transaction_marker_preserved_on_failure(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        session.begin()
        session.execute("INSERT INTO marker VALUES (7)")
        with pytest.raises(Exception):
            todb((('foo', 'bar'), ('a',)), session, 'src', commit=True)
        # outer transaction still active and marker visible
        assert session.in_transaction() is True
        assert session.execute(
            'SELECT * FROM marker').fetchall() == [(7,)]
        # caller continues and finalises itself
        session.rollback()
    finally:
        session.close()
    assert _count(engine, 'marker') == 0


def test_session_outer_transaction_success_uses_savepoint(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        session.begin()
        session.execute("INSERT INTO marker VALUES (7)")
        target = resolve_target(session)
        # force acquisition so scope selection sees the existing txn
        adapter = make_adapter(target, ResourceStack())
        adapter._acquire_connection()
        assert adapter._session_had_transaction is True
        scope = adapter.make_transaction_scope(True)
        assert isinstance(scope, SQLAlchemyTransactionScope)
        assert scope._nested is True  # savepoint, never the outer root
        # and a real operation still only completes the savepoint
        todb(TBL, session, 'src', commit=True)
        assert session.in_transaction() is True
        session.rollback()
    finally:
        session.close()
    assert _count(engine, 'marker') == 0


def test_session_commit_false_leaves_transaction_to_caller(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        session.begin()
        session.execute("INSERT INTO marker VALUES (7)")
        appenddb((('foo', 'bar'), ('z', 9)), session, 'src', commit=False)
        assert session.in_transaction() is True
        session.rollback()
    finally:
        session.close()
    assert _count(engine, 'src') == 0


def test_session_is_not_closed_by_petl(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    todb(TBL, session, 'src', commit=True)
    # session still usable for further work
    session.execute('SELECT 1').fetchall()
    session.close()


# ---------------------------------------------------------------------------
# Real persistence through the full create/drop/load flow
# ---------------------------------------------------------------------------

def test_todb_create_drop_full_flow_real_sqlite(engine, tmp_path):
    table = (('foo', 'bar'), ('a', 1), ('b', 2))
    # create once so the subsequent drop=True has a table to remove
    todb(table, engine, 'generated', create=True, commit=True)
    assert engine.execute('SELECT * FROM generated').fetchall() == \
        [('a', 1), ('b', 2)]
    # drop + recreate + reload in one call
    todb(table, engine, 'generated', create=True, drop=True, commit=True)
    assert engine.execute('SELECT * FROM generated').fetchall() == \
        [('a', 1), ('b', 2)]
    drop_table(engine, 'generated', commit=True)
    rows = engine.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='generated'").fetchall()
    assert rows == []
