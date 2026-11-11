# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


# standard library dependencies
import logging
from petl.compat import string_types


# internal dependencies
from petl.util.base import Table
from petl.io.db_adapters import resolve_target, open_sqlite_file_target, \
    ResourceStack, make_adapter, CleanupError, KIND_DBAPI, \
    SOURCE_SQLITE_FILE
from petl.io.db_create import drop_table, create_table
from petl.io.db_runner import run_dml


logger = logging.getLogger(__name__)
debug = logger.debug
warning = logger.warning


def fromdb(dbo, query, *args, **kwargs):
    """Provides access to data from any DB-API 2.0 connection via a given query.
    E.g., using :mod:`sqlite3`::

        >>> import petl as etl
        >>> import sqlite3
        >>> connection = sqlite3.connect('example.db')
        >>> table = etl.fromdb(connection, 'SELECT * FROM example')

    E.g., using :mod:`psycopg2` (assuming you've installed it first)::

        >>> import petl as etl
        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> table = etl.fromdb(connection, 'SELECT * FROM example')

    E.g., using :mod:`pymysql` (assuming you've installed it first)::

        >>> import petl as etl
        >>> import pymysql
        >>> connection = pymysql.connect(password='moonpie', database='thangs')
        >>> table = etl.fromdb(connection, 'SELECT * FROM example')

    The `dbo` argument may also be a function that creates a cursor. N.B., each
    call to the function should return a new cursor. E.g.::

        >>> import petl as etl
        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> mkcursor = lambda: connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        >>> table = etl.fromdb(mkcursor, 'SELECT * FROM example')

    The parameter `dbo` may also be an SQLAlchemy engine, session or
    connection object.

    The parameter `dbo` may also be a string, in which case it is interpreted
    as the name of a file containing an :mod:`sqlite3` database.

    Note that the default behaviour of most database servers and clients is for
    the entire result set for each query to be sent from the server to the
    client. If your query returns a large result set this can result in
    significant memory usage at the client. Some databases support server-side
    cursors which provide a means for client libraries to fetch result sets
    incrementally, reducing memory usage at the client.

    To use a server-side cursor with a PostgreSQL database, e.g.::

        >>> import petl as etl
        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> table = etl.fromdb(lambda: connection.cursor(name='arbitrary'),
        ...                    'SELECT * FROM example')

    For more information on server-side cursors see the following links:

        * http://initd.org/psycopg/docs/usage.html#server-side-cursors
        * http://mysql-python.sourceforge.net/MySQLdb.html#using-and-extending

    **Resource ownership**

    Constructing the view and calling :func:`iter` on it never connects,
    opens a cursor or executes SQL; resources are only acquired on the first
    row pull. Each iteration acquires its own resources:

    * a DB-API connection or cursor factory: petl closes the cursors it
      creates but never the caller's connection;
    * a directly borrowed DB-API cursor: petl neither closes nor replaces it,
      and two iterations cannot interleave safely because they share the
      single caller-managed cursor;
    * an sqlite file name: each iteration opens and closes its own
      connection, so iterations are independent;
    * an SQLAlchemy Engine: each iteration checks out and returns its own
      Connection (the engine is never disposed);
    * a borrowed SQLAlchemy Connection/Session: petl never closes or commits
      it; the query ``Result`` petl creates is always closed first.

    Reads never manage transactions. The same rules apply on normal
    exhaustion, explicit ``iterator.close()`` and any execute/fetch error.

    """

    # convenience for working with sqlite3
    if isinstance(dbo, string_types):
        return DbView(None, query, args, kwargs, sqlite_file=dbo)

    return DbView(dbo, query, args, kwargs)


class DbView(Table):

    def __init__(self, dbo, query, args, kwargs, sqlite_file=None):
        self.dbo = dbo
        self.query = query
        self.kwargs = kwargs
        # store args as a tuple so repeated iterations forward them verbatim
        self.args = tuple(args)
        self.sqlite_file = sqlite_file

    def __iter__(self):
        # Resolution happens per iteration (not at construction) so merely
        # building the view never touches the database.
        if self.sqlite_file is not None:
            target = open_sqlite_file_target(self.sqlite_file)
        else:
            target = resolve_target(self.dbo)
        return _iter_target(target, self.query, self.args, self.kwargs)


def _iter_target(target, query, args, kwargs):
    """Single streaming read implementation shared by every object type.

    Generator semantics give us the lazy contract for free: nothing before
    the first ``yield`` runs until the caller pulls the first row, and the
    ``finally`` block runs identically on exhaustion, ``close()`` and raised
    exceptions.
    """
    resources = ResourceStack()
    # For an sqlite file the connection itself is petl-owned and must be
    # closed after (and only after) its cursors/results.
    if target.source == SOURCE_SQLITE_FILE:
        resources.register('sqlite file connection', target.obj,
                           _close_dbapi_connection)
    adapter = make_adapter(target, resources)
    error = None
    try:
        result = adapter.open_query(query, args, dict(kwargs))
        if target.kind == KIND_DBAPI:
            header, first_row, row_iter = result.header()
            yield tuple(header)
            if first_row is not None:
                yield first_row  # native driver row, no tuple/dict coercion
            for row in row_iter:
                yield row
        else:
            header, _ignored, row_iter = result.header()
            yield tuple(header)
            for row in row_iter:
                yield row
    except GeneratorExit:
        # Triggered by explicit iterator.close() while suspended.
        raise
    except Exception as exc:
        error = exc
        raise
    finally:
        cleanup_error = resources.release_all(prior_error=error)
        if error is None and isinstance(cleanup_error, CleanupError):
            # Only a cleanup failure occurred: surface it rather than ending
            # the iteration silently.
            raise cleanup_error


def _close_dbapi_connection(connection):
    connection.close()


def todb(table, dbo, tablename, schema=None, commit=True,
         create=False, drop=False, constraints=True, metadata=None,
         dialect=None, sample=1000):
    """
    Load data into an existing database table via a DB-API 2.0
    connection or cursor. Note that the database table will be truncated,
    i.e., all existing rows will be deleted prior to inserting the new data.
    E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar'],
        ...          ['a', 1],
        ...          ['b', 2],
        ...          ['c', 2]]
        >>> # using sqlite3
        ... import sqlite3
        >>> connection = sqlite3.connect('example.db')
        >>> # assuming table "foobar" already exists in the database
        ... etl.todb(table, connection, 'foobar')
        >>> # using psycopg2
        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> # assuming table "foobar" already exists in the database
        ... etl.todb(table, connection, 'foobar')
        >>> # using pymysql
        >>> connection = pymysql.connect(password='moonpie', database='thangs')
        >>> # tell MySQL to use standard quote character
        ... connection.cursor().execute('SET SQL_MODE=ANSI_QUOTES')
        >>> # load data, assuming table "foobar" already exists in the database
        ... etl.todb(table, connection, 'foobar')

    N.B., for MySQL the statement ``SET SQL_MODE=ANSI_QUOTES`` is required to
    ensure MySQL uses SQL-92 standard quote characters.

    A cursor can also be provided instead of a connection, e.g.::

        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> cursor = connection.cursor()
        >>> etl.todb(table, cursor, 'foobar')

    The parameter `dbo` may also be an SQLAlchemy engine, session or
    connection object.

    The parameter `dbo` may also be a string, in which case it is interpreted
    as the name of a file containing an :mod:`sqlite3` database.

    If ``create=True`` this function will attempt to automatically create a
    database table before loading the data. This functionality requires
    `SQLAlchemy <http://www.sqlalchemy.org/>`_ to be installed.

    **Keyword arguments:**

    table : table container
        Table data to load
    dbo : database object
        DB-API 2.0 connection, callable returning a DB-API 2.0 cursor, or
        SQLAlchemy connection, engine or session
    tablename : string
        Name of the table in the database
    schema : string
        Name of the database schema to find the table in
    commit : bool
        If True commit the changes
    create : bool
        If True attempt to create the table before loading, inferring types
        from a sample of the data (requires SQLAlchemy)
    drop : bool
        If True attempt to drop the table before recreating (only relevant if
        create=True)
    constraints : bool
        If True use length and nullable constraints (only relevant if
        create=True)
    metadata : sqlalchemy.MetaData
        Custom table metadata (only relevant if create=True)
    dialect : string
        One of {'access', 'sybase', 'sqlite', 'informix', 'firebird', 'mysql',
        'oracle', 'maxdb', 'postgresql', 'mssql'} (only relevant if
        create=True)
    sample : int
        Number of rows to sample when inferring types etc. Set to 0 to use the
        whole table (only relevant if create=True)

    **Resource and transaction ownership**

    Every supported object type goes through the shared database adapter
    layer (see :mod:`petl.io.db_adapters`). Borrowed DB-API connections,
    cursors and cursor factories are never closed by petl; cursors petl
    creates are always closed, including after an execution or source-row
    error. With ``commit=True`` a successful load is committed exactly once;
    on failure petl does not commit and does not roll back a borrowed
    connection's transaction - the caller keeps control. sqlite file names
    open a petl-owned connection that is always closed, with a rollback on
    failure when ``commit=True``. SQLAlchemy Engines only return the
    connections petl checked out (never ``dispose``); borrowed Connections and
    Sessions keep their outer transactions, with petl using a SAVEPOINT scope
    when an outer transaction already exists. The DDL steps triggered by
    ``create``/``drop`` follow the same per-step contract; the steps are not
    wrapped in one cross-statement atomic transaction.

    .. note::

        This function is in principle compatible with any DB-API 2.0
        compliant database driver. However, at the time of writing some DB-API
        2.0 implementations, including cx_Oracle and MySQL's
        Connector/Python, are not compatible with this function, because they
        only accept a list argument to the cursor.executemany() function
        called internally by :mod:`petl`. This can be worked around by
        proxying the cursor objects, e.g.::

            >>> import cx_Oracle
            >>> connection = cx_Oracle.Connection(...)
            >>> class CursorProxy(object):
            ...     def __init__(self, cursor):
            ...         self._cursor = cursor
            ...     def executemany(self, statement, parameters, **kwargs):
            ...         # convert parameters to a list
            ...         parameters = list(parameters)
            ...         # pass through to proxied cursor
            ...         return self._cursor.executemany(statement, parameters, **kwargs)
            ...     def __getattr__(self, item):
            ...         return getattr(self._cursor, item)
            ...
            >>> def get_cursor():
            ...     return CursorProxy(connection.cursor())
            ...
            >>> import petl as etl
            >>> etl.todb(tbl, get_cursor, ...)

        Note however that this does imply loading the entire table into
        memory as a list prior to inserting into the database.

    """

    target = _resolve_write_target(dbo)
    try:
        if create:
            if drop:
                drop_table(_TargetWrapper(target), tablename, schema=schema,
                           commit=commit)
            create_table(table, _TargetWrapper(target), tablename,
                         schema=schema, commit=commit,
                         constraints=constraints, metadata=metadata,
                         dialect=dialect, sample=sample)
        run_dml(target, table, tablename, schema=schema, commit=commit,
                truncate=True)
    finally:
        _release_write_target(dbo, target)


Table.todb = todb


def appenddb(table, dbo, tablename, schema=None, commit=True):
    """
    Load data into an existing database table via a DB-API 2.0
    connection or cursor. As :func:`petl.io.db.todb` except that the database
    table will be appended, i.e., the new data will be inserted into the
    table, and any existing rows will remain.

    """

    target = _resolve_write_target(dbo)
    try:
        run_dml(target, table, tablename, schema=schema, commit=commit,
                truncate=False)
    finally:
        _release_write_target(dbo, target)


Table.appenddb = appenddb


def _resolve_write_target(dbo):
    """Resolve a write/DDL dbo, opening sqlite file connections as owned."""
    if isinstance(dbo, string_types):
        return open_sqlite_file_target(dbo)
    if isinstance(dbo, _TargetWrapper):
        return dbo.target
    return resolve_target(dbo)


def _release_write_target(dbo, target):
    """Close petl-owned sqlite file connections; nothing else."""
    if isinstance(dbo, string_types) and target.source == SOURCE_SQLITE_FILE:
        target.obj.close()


class _TargetWrapper(object):
    """Internal carrier so todb can pass a resolved target into DDL helpers.

    The public create_table/drop_table signatures keep accepting raw dbo
    objects; internally a todb(create=True/...) call hands them the already
    resolved target to avoid re-resolving (and re-opening sqlite files) per
    step.
    """

    def __init__(self, target):
        self.target = target


# ---------------------------------------------------------------------------
# Backwards-compatible private helpers and constants
# ---------------------------------------------------------------------------
# These SQL strings used to live in this module; the canonical definitions
# now belong to the shared planning module. Re-export them so any code that
# imported them from petl.io.db keeps working.
from petl.io import db_planning as _db_planning

SQL_TRUNCATE_QUERY = _db_planning.SQL_TRUNCATE_QUERY
SQL_INSERT_QUERY = _db_planning.SQL_INSERT_QUERY


def _todb(table, dbo, tablename, schema=None, commit=True, truncate=False):
    """Legacy internal entry point retained for compatibility."""
    target = _resolve_write_target(dbo)
    try:
        run_dml(target, table, tablename, schema=schema, commit=commit,
                truncate=truncate)
    finally:
        _release_write_target(dbo, target)
