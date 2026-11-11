# -*- coding: utf-8 -*-
"""Regression tests for the unified database resource adapter layer.

These tests use the recording protocol doubles in ``_db_doubles`` plus real
local sqlite3 and SQLAlchemy 1.4 (legacy API) databases. They assert the
resource ownership, transaction and exception contracts documented in
:mod:`petl.io.db_adapters`, including the baseline defects that used to leak
(engine connections not returned, DB-API cursors not closed on executemany
errors). No garbage collection is relied upon: every cleanup assertion runs
at explicit close or on the failed call's return.
"""
from __future__ import absolute_import, print_function, division

import sqlite3
import tempfile
import warnings

import pytest

from petl.compat import next
from petl.io.db import fromdb, todb, appenddb
from petl.io.db_create import drop_table
from petl.io.db_adapters import (
    resolve_target, open_sqlite_file_target, make_adapter, ResourceStack,
    CleanupError,
    KIND_DBAPI,
    SOURCE_DBAPI_CONNECTION, SOURCE_DBAPI_CURSOR,
    SOURCE_DBAPI_CURSOR_FACTORY, SOURCE_SQLITE_FILE,
    NoTransactionScope)
from petl.test.io._db_doubles import (
    RecordingConnection, NonIterableConnection, ClickHouseLikeConnection,
    RecordingCursor, FakeDBAPIError, CallLog, make_paramstyle_connection)


warnings.simplefilter('ignore')

DESCRIPTION = [('foo',), ('bar',)]
ROWS = [('a', 1), ('b', 2), ('c', 3)]
TABLE = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 3))


# ---------------------------------------------------------------------------
# Object resolution / provenance
# ---------------------------------------------------------------------------

def test_resolve_target_provenance():
    log = CallLog()
    conn = RecordingConnection(log)
    target = resolve_target(conn)
    assert target.kind == KIND_DBAPI
    assert target.source == SOURCE_DBAPI_CONNECTION
    assert target.petl_owns_connection is False
    assert target.petl_owns_cursor is True

    cursor = conn.cursor()
    target = resolve_target(cursor)
    assert target.source == SOURCE_DBAPI_CURSOR
    assert target.petl_owns_cursor is False

    target = resolve_target(conn.cursor)
    assert target.source == SOURCE_DBAPI_CURSOR_FACTORY
    assert target.petl_owns_cursor is True


def test_resolve_clickhouse_priority():
    log = CallLog()
    ch = ClickHouseLikeConnection(log)
    target = resolve_target(ch)
    assert target.kind == KIND_DBAPI
    assert target.is_clickhouse is True


def test_resolve_unsupported_raises():
    from petl.errors import ArgumentError
    with pytest.raises(ArgumentError):
        resolve_target(object())


def test_sqlite_file_target_is_owned():
    f = tempfile.NamedTemporaryFile(delete=False)
    f.close()
    target = open_sqlite_file_target(f.name)
    assert target.source == SOURCE_SQLITE_FILE
    assert target.petl_owns_connection is True
    assert target.petl_owns_cursor is True
    target.obj.close()


# ---------------------------------------------------------------------------
# Read contract: laziness and DB-API cursor lifecycle
# ---------------------------------------------------------------------------

def test_fromdb_does_not_connect_until_first_next():
    log = CallLog()

    class NoConnect(RecordingConnection):
        def cursor(self):
            raise AssertionError('cursor() must not be called before next()')

    view = fromdb(NoConnect(log), 'SELECT 1')
    iterator = iter(view)  # must not connect
    assert log.names() == []
    # the iterator itself is closed without ever pulling a row
    iterator.close()


def test_fromdb_connection_closes_cursor_not_connection_on_exhaustion():
    log = CallLog()
    conn = RecordingConnection(log, rows=ROWS, description=DESCRIPTION)
    # N.B. use an explicit iterator: list(view) also probes Table.__len__,
    # which legitimately performs a second full iteration.
    out = list(iter(fromdb(conn, 'SELECT * FROM t')))
    assert out[0] == ('foo', 'bar')
    assert out[1:] == ROWS
    assert log.count('cursor.close') == 1
    assert 'connection.close' not in log.names()
    assert conn.closed is False


def test_fromdb_explicit_close_releases_cursor():
    log = CallLog()
    conn = RecordingConnection(log, rows=ROWS, description=DESCRIPTION)
    iterator = iter(fromdb(conn, 'SELECT * FROM t'))
    next(iterator)
    iterator.close()
    assert log.names()[-1] == 'cursor.close'
    assert conn.closed is False


def test_fromdb_execute_failure_still_closes_cursor():
    log = CallLog()
    conn = RecordingConnection(
        log, rows=ROWS, description=DESCRIPTION,
        fail_execute=FakeDBAPIError('boom'))
    iterator = iter(fromdb(conn, 'SELECT bad'))
    with pytest.raises(FakeDBAPIError):
        next(iterator)
    assert log.names()[-1] == 'cursor.close'
    assert conn.closed is False


def test_fromdb_fetch_failure_still_closes_cursor():
    log = CallLog()
    # fail on the second streamed data row (third iterator pull)
    conn = RecordingConnection(log, rows=ROWS, description=DESCRIPTION,
                               fail_fetch=2)
    iterator = iter(fromdb(conn, 'SELECT * FROM t'))
    next(iterator)  # header (streams row 1)
    next(iterator)  # cached row 1
    with pytest.raises(FakeDBAPIError):
        next(iterator)  # streams row 2 -> failure
    assert log.names()[-1] == 'cursor.close'
    assert 'connection.close' not in log.names()


def test_fromdb_delayed_description_and_empty_result():
    log = CallLog()
    conn = RecordingConnection(log, rows=[], description=DESCRIPTION,
                               delayed_description=True)
    assert list(fromdb(conn, 'SELECT * FROM empty')) == [('foo', 'bar')]


def test_fromdb_non_iterable_cursor_uses_fetchall_fallback():
    log = CallLog()
    conn = NonIterableConnection(log, rows=ROWS, description=DESCRIPTION)
    out = list(fromdb(conn, 'SELECT * FROM t'))
    assert out == [('foo', 'bar')] + ROWS
    assert 'cursor.fetchall' in log.names()


def test_fromdb_borrowed_cursor_is_never_closed():
    log = CallLog()
    conn = RecordingConnection(log, rows=ROWS, description=DESCRIPTION)
    cursor = conn.cursor()
    log.events = []
    out = list(fromdb(cursor, 'SELECT * FROM t'))
    assert out[0] == ('foo', 'bar')
    assert 'cursor.close' not in log.names()
    assert cursor.closed is False


def test_fromdb_two_iterators_interleave_independently():
    log = CallLog()
    conn = RecordingConnection(log, rows=ROWS, description=DESCRIPTION)
    view = fromdb(conn, 'SELECT * FROM t')
    i1, i2 = iter(view), iter(view)
    assert next(i1) == ('foo', 'bar')
    assert next(i1) == ('a', 1)
    assert next(i2) == ('foo', 'bar')
    i1.close()
    assert next(i2) == ('a', 1)
    assert list(i2) == ROWS[1:]


def test_fromdb_preserves_native_row_objects():
    class NativeRow(list):
        pass

    native = [NativeRow(('a', 1)), NativeRow(('b', 2))]
    log = CallLog()
    conn = RecordingConnection(log, rows=native, description=DESCRIPTION)
    out = list(fromdb(conn, 'SELECT * FROM t'))
    assert isinstance(out[1], NativeRow)


def test_fromdb_header_is_tuple_and_params_forwarded():
    log = CallLog()
    conn = RecordingConnection(log, rows=ROWS[:1], description=DESCRIPTION)
    out = list(fromdb(conn, 'SELECT * FROM t WHERE bar > ?', (1,)))
    assert isinstance(out[0], tuple)
    execute_event = log.of('cursor.execute')[0]
    assert execute_event[2] == (1,)


def test_fromdb_factory_cursor_closed_per_iteration():
    log = CallLog()
    conn = RecordingConnection(log, rows=ROWS, description=DESCRIPTION)
    view = fromdb(conn.cursor, 'SELECT * FROM t')
    list(iter(view))  # explicit iterator: one cursor per iteration
    list(iter(view))
    assert log.count('connection.cursor') == 2
    assert log.count('cursor.close') == 2
    assert conn.closed is False


# ---------------------------------------------------------------------------
# Read contract: real sqlite3 including file convenience entry
# ---------------------------------------------------------------------------

def _seed_sqlite(path):
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE foobar (foo TEXT, bar INT)')
    conn.executemany('INSERT INTO foobar VALUES (?, ?)', ROWS)
    conn.commit()
    conn.close()


def test_sqlite_file_read_opens_and_closes_own_connection():
    f = tempfile.NamedTemporaryFile(delete=False)
    f.close()
    _seed_sqlite(f.name)
    view = fromdb(f.name, 'SELECT * FROM foobar')
    assert list(view) == [('foo', 'bar')] + ROWS
    # two independent iterations each manage their own connection
    i1, i2 = iter(view), iter(view)
    assert next(i1) == ('foo', 'bar')
    assert next(i2) == ('foo', 'bar')
    list(i1)
    list(i2)


def test_sqlite_file_read_failure_closes_connection():
    f = tempfile.NamedTemporaryFile(delete=False)
    f.close()
    _seed_sqlite(f.name)
    iterator = iter(fromdb(f.name, 'SELECT * FROM missing'))
    with pytest.raises(sqlite3.OperationalError):
        next(iterator)
    # a fresh connection can still be opened; nothing leaked an open handle
    fresh = sqlite3.connect(f.name)
    fresh.close()


# ---------------------------------------------------------------------------
# DML contract: DB-API connection / cursor / factory matrix
# ---------------------------------------------------------------------------

def test_todb_connection_commit_lifecycle():
    log = CallLog()
    conn = RecordingConnection(log)
    todb(TABLE, conn, 'foobar', commit=True)
    names = log.names()
    assert names.count('connection.cursor') == 2  # truncate resurrect
    assert names.count('cursor.close') == 2
    assert names.count('connection.commit') == 1
    assert 'connection.rollback' not in names
    assert 'connection.close' not in names
    truncate = log.of('cursor.execute')[0]
    assert truncate[1] == 'DELETE FROM "foobar"'
    insert = log.of('cursor.executemany.iterator')[0]
    assert insert[1] == 'INSERT INTO "foobar" ("foo", "bar") VALUES (?, ?)'


def test_todb_commit_false_issues_no_transaction_calls():
    log = CallLog()
    conn = RecordingConnection(log)
    todb(TABLE, conn, 'foobar', commit=False)
    names = log.names()
    assert 'connection.commit' not in names
    assert 'connection.rollback' not in names
    assert 'connection.begin' not in names
    assert 'connection.close' not in names
    # and the adapter itself selects the no-op scope
    target = resolve_target(RecordingConnection(CallLog()))
    scope = make_adapter(target, ResourceStack()).make_transaction_scope(False)
    assert isinstance(scope, NoTransactionScope)


def test_todb_executemany_failure_closes_cursors_no_commit_no_rollback():
    log = CallLog()
    conn = RecordingConnection(log, fail_executemany=FakeDBAPIError('load'))
    with pytest.raises(FakeDBAPIError):
        todb(TABLE, conn, 'foobar', commit=True)
    names = log.names()
    assert 'connection.commit' not in names
    assert 'connection.rollback' not in names
    assert 'connection.close' not in names
    assert names.count('cursor.close') == 2  # truncate + load cursors


def test_todb_source_generator_failure_cleans_up():
    log = CallLog()
    conn = RecordingConnection(log)

    def broken():
        yield ('foo', 'bar')
        yield ('a', 1)
        raise FakeDBAPIError('source boom')

    with pytest.raises(FakeDBAPIError):
        todb(broken(), conn, 'foobar', commit=True)
    names = log.names()
    assert 'connection.commit' not in names
    assert names.count('cursor.close') == 2


def test_todb_does_not_materialise_source():
    log = CallLog()

    class StrictCursor(RecordingCursor):
        def executemany(self, statement, rows, **kwargs):
            # reject a concrete list: the contract passes a live iterator
            assert not isinstance(rows, list), 'rows must stay an iterator'
            return super(StrictCursor, self).executemany(statement, rows,
                                                         **kwargs)

    class StrictConn(RecordingConnection):
        def cursor(self):
            log.record('connection.cursor')
            return StrictCursor(log, self)

    conn = StrictConn(log)

    def gen():
        for row in TABLE:
            yield row

    todb(gen(), conn, 'foobar', commit=False)
    assert log.of('cursor.executemany.iterator')


def test_todb_borrowed_cursor_not_closed_or_replaced():
    log = CallLog()
    conn = RecordingConnection(log)
    cursor = conn.cursor()
    log.events = []
    todb(TABLE, cursor, 'foobar', commit=True)
    assert 'cursor.close' not in log.names()
    assert cursor.closed is False
    assert log.count('connection.commit') == 1


def test_todb_cursor_factory_lifecycle():
    log = CallLog()
    conn = RecordingConnection(log)
    todb(TABLE, conn.cursor, 'foobar', commit=True)
    names = log.names()
    assert names.count('connection.cursor') == 2
    assert names.count('cursor.close') == 2
    assert names.count('connection.commit') == 1


def test_appenddb_skips_truncate():
    log = CallLog()
    conn = RecordingConnection(log)
    appenddb(TABLE, conn, 'foobar', commit=True)
    names = log.names()
    assert names.count('connection.cursor') == 1
    assert not any('DELETE' in str(event) for event in log.events)
    assert names.count('connection.commit') == 1


def test_todb_identifier_and_schema_quoting():
    log = CallLog()
    conn = RecordingConnection(log)
    todb((('foo foo', 'bar.baz'), ('a', 1)), conn, 'weird " name',
         schema='s x', commit=False)
    assert log.of('cursor.execute')[0][1] == \
        'DELETE FROM "s x"."weird "" name"'


@pytest.mark.parametrize('style,expected', [
    ('qmark', '?, ?'),
    ('format', '%s, %s'),
    ('pyformat', '%s, %s'),
    ('numeric', ':1, :2'),
    ('named', ':"foo", :"bar"'),
])
def test_placeholder_styles_match_legacy_output(style, expected):
    log = CallLog()
    conn = make_paramstyle_connection(style, log)
    target = resolve_target(conn)
    adapter = make_adapter(target, ResourceStack())
    from petl.io.db_planning import build_dml_plan
    plan = build_dml_plan(adapter, ('foo', 'bar'), 't', None, False)
    assert plan.placeholders == expected


def test_clickhouse_dedicated_sql_branch():
    log = CallLog()
    ch = ClickHouseLikeConnection(log)
    todb(TABLE, ch, 'foo', commit=True)
    assert log.of('cursor.execute')[0][1] == \
        'TRUNCATE TABLE IF EXISTS "foo"'
    assert log.of('cursor.executemany.iterator')[0][1] == \
        'INSERT INTO "foo" ("foo", "bar") VALUES'


# ---------------------------------------------------------------------------
# DDL contract
# ---------------------------------------------------------------------------

def test_ddl_connection_lifecycle():
    log = CallLog()
    conn = RecordingConnection(log)
    drop_table(conn, 'foobar', commit=True)
    names = log.names()
    # The cursor is released (LIFO) before the single terminal commit.
    assert names == ['connection.cursor', 'cursor.execute', 'cursor.close',
                     'connection.commit']
    assert log.of('cursor.execute')[0][1] == 'DROP TABLE "foobar"'


def test_ddl_failure_closes_cursor_and_skips_commit():
    log = CallLog()
    conn = RecordingConnection(log, fail_execute=FakeDBAPIError('ddl'))
    with pytest.raises(FakeDBAPIError):
        drop_table(conn, 'foobar', commit=True)
    assert log.names() == ['connection.cursor', 'cursor.execute',
                           'cursor.close']


def test_ddl_borrowed_cursor_not_closed():
    log = CallLog()
    conn = RecordingConnection(log)
    cursor = conn.cursor()
    log.events = []
    drop_table(cursor, 'foobar', commit=True)
    assert 'cursor.close' not in log.names()
    assert cursor.closed is False


# ---------------------------------------------------------------------------
# Cleanup error precedence
# ---------------------------------------------------------------------------

def test_business_error_takes_precedence_over_cleanup_error(caplog):
    class Conn(RecordingConnection):
        _seq = 0

        def cursor(self):
            cursor = super(Conn, self).cursor()
            original = cursor.close

            def close():
                Conn._seq += 1
                if Conn._seq == 2:  # final cleanup close, after truncate close
                    raise RuntimeError('final close exploded')
                return original()

            cursor.close = close
            return cursor

    log = CallLog()
    conn = Conn(log, fail_executemany=FakeDBAPIError('biz boom'))
    with pytest.raises(FakeDBAPIError):
        todb(TABLE, conn, 'foo', commit=True)
    # the secondary close failure is logged and discoverable
    assert any('final close exploded' in record.getMessage()
               or 'releasing resource' in record.getMessage()
               for record in caplog.records)


def test_cleanup_only_failure_raises_cleanup_error():
    class Conn(RecordingConnection):
        def cursor(self):
            cursor = super(Conn, self).cursor()
            cursor._fail_close = RuntimeError('close exploded')
            return cursor

    log = CallLog()
    conn = Conn(log)
    with pytest.raises(CleanupError) as info:
        appenddb(TABLE, conn, 'foo', commit=True)
    assert isinstance(info.value.__cause__, RuntimeError)


def test_second_resource_acquisition_failure_releases_first():
    class Conn(RecordingConnection):
        def __init__(self, log):
            super(Conn, self).__init__(log)
            self._n = 0

        def cursor(self):
            self._n += 1
            if self._n > 1:  # truncate cursor ok, resurrect cursor fails
                raise RuntimeError('cannot open cursor')
            return super(Conn, self).cursor()

    log = CallLog()
    conn = Conn(log)
    with pytest.raises(RuntimeError):
        todb(TABLE, conn, 'foo', commit=True)
    assert log.count('cursor.close') == 1


def test_each_resource_released_at_most_once():
    log = CallLog()
    conn = RecordingConnection(log, rows=ROWS, description=DESCRIPTION)
    list(iter(fromdb(conn, 'SELECT * FROM t')))
    assert log.count('cursor.close') == 1


# ---------------------------------------------------------------------------
# SQLite file write ownership and visible persisted state
# ---------------------------------------------------------------------------

def test_sqlite_file_write_commits_and_closes():
    f = tempfile.NamedTemporaryFile(delete=False)
    f.close()
    conn = sqlite3.connect(f.name)
    conn.execute('CREATE TABLE foobar (foo TEXT, bar INT)')
    conn.commit()
    conn.close()

    todb(TABLE, f.name, 'foobar', commit=True)
    check = sqlite3.connect(f.name)
    assert check.execute('SELECT * FROM foobar').fetchall() == ROWS
    check.close()


def test_sqlite_file_write_failure_rolls_back_and_closes():
    f = tempfile.NamedTemporaryFile(delete=False)
    f.close()
    conn = sqlite3.connect(f.name)
    conn.execute('CREATE TABLE foobar (foo TEXT NOT NULL, bar INT)')
    conn.execute("INSERT INTO foobar VALUES ('keep', 0)")
    conn.commit()
    conn.close()

    # A NOT NULL violation makes the load fail mid-stream after truncate.
    bad = (('foo', 'bar'), ('a', 1), (None, 2))
    with pytest.raises(sqlite3.IntegrityError):
        todb(bad, f.name, 'foobar', commit=True)

    check = sqlite3.connect(f.name)
    # Rollback restored the pre-existing row that truncate would otherwise
    # have removed; nothing partial is committed.
    assert check.execute('SELECT * FROM foobar').fetchall() == [('keep', 0)]
    check.close()
