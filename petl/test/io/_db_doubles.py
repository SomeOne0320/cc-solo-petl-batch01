# -*- coding: utf-8 -*-
"""Recording DB-API and SQLAlchemy protocol doubles for adapter tests.

These are *doubles*, not external-server validation: they record every
resource/transaction call and can inject failures at any lifecycle point.
Real persistence and transaction behaviour is exercised against local
sqlite3 / SQLAlchemy in the companion test modules.
"""
from __future__ import absolute_import, print_function, division

from petl.compat import next as _next


class FakeDBAPIError(Exception):
    """Sentinel error raised by the doubles to simulate driver failures."""


class CallLog(object):
    def __init__(self):
        self.events = []

    def record(self, name, *args):
        self.events.append((name,) + args)

    def names(self):
        return [event[0] for event in self.events]

    def count(self, name):
        return self.names().count(name)

    def of(self, name):
        return [event for event in self.events if event[0] == name]


class _RowStream(object):
    """Lazy row iterator that can fail on a configured 1-based pull."""

    def __init__(self, rows, fail_at=None):
        self._rows = iter(rows)
        self._fail_at = fail_at
        self._n = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._n += 1
        if self._fail_at is not None and self._n == self._fail_at:
            raise FakeDBAPIError('fetch failure on row %d' % self._n)
        return next(self._rows)

    next = __next__


class RecordingCursor(object):
    """A DB-API cursor double with independently injectable failures."""

    def __init__(self, log, connection, rows=None, description=None,
                 iterable=True, fail_execute=None, fail_executemany=None,
                 fail_fetch=None, fail_close=None, delayed_description=False,
                 record_fetchall=False):
        self.log = log
        self.connection = connection
        self._rows = list(rows or [])
        self._description = description
        self._iterable = iterable
        self._fail_execute = fail_execute
        self._fail_executemany = fail_executemany
        self._fail_fetch = fail_fetch
        self._fail_close = fail_close
        self._delayed = delayed_description
        self._record_fetchall = record_fetchall
        self.description = None if delayed_description else description
        self.closed = False
        self.execute_calls = []

    def execute(self, query, params=None, **kwargs):
        self.log.record('cursor.execute', query, params, kwargs)
        self.execute_calls.append((query, params, kwargs))
        if self._fail_execute is not None:
            raise self._fail_execute
        self._stream = _RowStream(self._rows, self._fail_fetch)
        if not self._delayed:
            self.description = self._description

    def executemany(self, statement, rows, **kwargs):
        self.log.record('cursor.executemany.iterator', statement, kwargs)

        def spy():
            for row in rows:
                self.log.record('cursor.executemany.row', row)
                if self._fail_executemany is not None:
                    raise self._fail_executemany
                yield row

        # Consume lazily; the double drives the iterator but never builds a
        # parallel materialised list of the whole input.
        for _ in spy():
            pass

    def __iter__(self):
        if not self._iterable:
            raise TypeError('cursor is not iterable')
        return self

    def __next__(self):
        if self._delayed and self.description is None:
            self.description = self._description
        row = next(self._stream)
        self.log.record('cursor.next-row')
        return row

    next = __next__

    def fetchall(self):
        if self._record_fetchall:
            self.log.record('cursor.fetchall')
        if self._delayed and self.description is None:
            self.description = self._description
        if self._fail_fetch is not None:
            raise FakeDBAPIError('fetch failure')
        return list(self._rows)

    def fetchone(self):
        try:
            return _next(self)
        except StopIteration:
            return None

    def fetchmany(self, size=1):
        out = []
        for _ in range(size):
            try:
                out.append(_next(self))
            except StopIteration:
                break
        return out

    def close(self):
        self.log.record('cursor.close')
        if self._fail_close is not None:
            raise self._fail_close
        self.closed = True


class RecordingConnection(object):
    """A DB-API connection double; override ``paramstyle`` per test."""

    paramstyle = 'qmark'

    def __init__(self, log=None, rows=None, description=None, **cursor_kw):
        self.log = log if log is not None else CallLog()
        self._rows = rows
        self._description = description
        self._cursor_kw = cursor_kw
        self.closed = False
        self.commit_count = 0
        self.rollback_count = 0
        self.begin_count = 0

    def cursor(self):
        self.log.record('connection.cursor')
        return RecordingCursor(self.log, self, rows=self._rows,
                               description=self._description,
                               **self._cursor_kw)

    def commit(self):
        self.log.record('connection.commit')
        self.commit_count += 1

    def rollback(self):
        self.log.record('connection.rollback')
        self.rollback_count += 1

    def begin(self):
        self.log.record('connection.begin')
        self.begin_count += 1

    def close(self):
        self.log.record('connection.close')
        self.closed = True


class NonIterableConnection(RecordingConnection):
    def cursor(self):
        self.log.record('connection.cursor')
        kw = dict(self._cursor_kw)
        kw['iterable'] = False
        kw['record_fetchall'] = True
        return RecordingCursor(self.log, self, rows=self._rows,
                               description=self._description, **kw)


class ClickHouseLikeConnection(RecordingConnection):
    """Connection whose type name matches the ClickHouse detection rule."""

    paramstyle = 'qmark'


import sys as _sys
import types as _types

_clickhouse_module = _types.ModuleType('clickhouse_driver.connection')
_clickhouse_module.paramstyle = 'qmark'
_sys.modules.setdefault('clickhouse_driver', _types.ModuleType('clickhouse_driver'))
_sys.modules['clickhouse_driver.connection'] = _clickhouse_module
ClickHouseLikeConnection.__module__ = 'clickhouse_driver.connection'
ClickHouseLikeConnection.__qualname__ = \
    'clickhouse_driver.connection.Connection'


def make_paramstyle_connection(style, log, **cursor_kw):
    """Build a connection double whose defining module reports paramstyle."""
    import sys
    import types

    module_name = 'petl_test_paramstyle_%s' % style
    module = types.ModuleType(module_name)
    module.paramstyle = style
    sys.modules[module_name] = module

    class _Conn(RecordingConnection):
        paramstyle = style

    _Conn.__module__ = module_name
    return _Conn(log, **cursor_kw)
