# -*- coding: utf-8 -*-
"""Unified database resource adapter layer.

This module is the single place where ``petl`` decides *what kind* of database
object it has been handed, *which* resources it has to create itself, *who*
owns the surrounding transaction and *how* acquired resources are released.
The read path (:mod:`petl.io.db`), the DML path (``todb``/``appenddb``) and
the DDL path (:mod:`petl.io.db_create`) all resolve their database object
through :func:`resolve_target` and drive it through the same
:class:`DbAdapter` interface and :class:`ResourceStack` lifecycle machinery.

Design contract
---------------

* Object resolution (duck typing, including the ClickHouse priority) happens
  once, in :func:`resolve_target`, and returns a :class:`DbTarget` describing
  the object's provenance and ownership.
* SQL generation (identifier quoting, placeholder strings, INSERT/DELETE
  plans) lives in :mod:`petl.io.db_planning` and never touches ownership.
* Adapters acquire resources lazily, register every successfully acquired
  resource with a :class:`ResourceStack` and release exactly those resources,
  exactly once, in reverse acquisition order.
* Borrowed resources (caller connections, caller cursors, borrowed
  SQLAlchemy connections/sessions) are never closed or disposed by petl;
  resources petl creates itself always are.
* Read iterations never manage transactions. Write/DDL transaction scopes
  are expressed through :class:`TransactionScope` so the same begin/commit/
  rollback rules apply to DML and DDL.

The module imports SQLAlchemy lazily, so plain DB-API / sqlite3 usage keeps
working when SQLAlchemy is not installed.
"""
from __future__ import absolute_import, print_function, division

import logging

from petl.errors import ArgumentError
from petl.io.db_utils import _hasmethod, _hasmethods, _hasprop, \
    _is_clikchouse_dbapi_connection


logger = logging.getLogger(__name__)
debug = logger.debug
warning = logger.warning


# ---------------------------------------------------------------------------
# Object resolution
# ---------------------------------------------------------------------------

# Provenance identifiers. ``source`` records where the object came from;
# ``kind`` records which adapter protocol it speaks.
KIND_DBAPI = 'dbapi'
KIND_SQLALCHEMY = 'sqlalchemy'

SOURCE_SQLITE_FILE = 'sqlite_file'
SOURCE_DBAPI_CONNECTION = 'dbapi_connection'
SOURCE_DBAPI_CURSOR = 'dbapi_cursor'
SOURCE_DBAPI_CURSOR_FACTORY = 'dbapi_cursor_factory'
SOURCE_SQLALCHEMY_ENGINE = 'sqlalchemy_engine'
SOURCE_SQLALCHEMY_CONNECTION = 'sqlalchemy_connection'
SOURCE_SQLALCHEMY_SESSION = 'sqlalchemy_session'


class DbTarget(object):
    """Resolved description of a polymorphic ``dbo`` argument.

    Attributes
    ----------
    obj : object
        The original database object (or the connection opened for a sqlite
        file convenience entry).
    kind : str
        One of the ``KIND_*`` constants - the executable protocol spoken by
        the object.
    source : str
        One of the ``SOURCE_*`` constants - the object's provenance, which
        determines the ownership rules.
    petl_owns_connection : bool
        True when petl opened (and therefore must close) the underlying
        connection itself.
    petl_owns_cursor : bool
        True when cursors used against this target are created by petl and
        therefore must be closed by petl.
    is_clickhouse : bool
        True for ClickHouse DB-API connections, which keep their dedicated
        TRUNCATE / INSERT SQL branch.
    """

    __slots__ = ('obj', 'kind', 'source', 'petl_owns_connection',
                 'petl_owns_cursor', 'is_clickhouse')

    def __init__(self, obj, kind, source, petl_owns_connection,
                 petl_owns_cursor, is_clickhouse=False):
        self.obj = obj
        self.kind = kind
        self.source = source
        self.petl_owns_connection = petl_owns_connection
        self.petl_owns_cursor = petl_owns_cursor
        self.is_clickhouse = is_clickhouse

    def __repr__(self):
        return ('DbTarget(kind=%r, source=%r, owns_connection=%r, '
                'owns_cursor=%r, clickhouse=%r)'
                % (self.kind, self.source, self.petl_owns_connection,
                   self.petl_owns_cursor, self.is_clickhouse))


def _is_sqlalchemy_engine(dbo):
    return (_hasmethods(dbo, 'execute', 'connect', 'raw_connection')
            and _hasprop(dbo, 'driver'))


def _is_sqlalchemy_session(dbo):
    return _hasmethods(dbo, 'execute', 'connection', 'get_bind')


def _is_sqlalchemy_connection(dbo):
    # N.B., not completely selective on its own; apply after ruling out
    # DB-API cursors and engines/sessions.
    return _hasmethod(dbo, 'execute') and _hasprop(dbo, 'connection')


def _is_dbapi_connection(dbo):
    return _hasmethod(dbo, 'cursor')


def _is_dbapi_cursor(dbo):
    return _hasmethods(dbo, 'execute', 'executemany', 'fetchone',
                       'fetchmany', 'fetchall')


def resolve_target(dbo):
    """Resolve a polymorphic database object into a :class:`DbTarget`.

    The detection order is shared by the read, DML and DDL paths. The
    ClickHouse connection check intentionally runs first, preserving the
    historical priority where a ClickHouse connection must take its dedicated
    SQL branch rather than the generic DB-API one.
    """

    # ClickHouse connections are DB-API connections with dedicated SQL.
    if _is_clikchouse_dbapi_connection(dbo):
        debug('assuming %r is clickhouse DB-API 2.0 connection', dbo)
        return DbTarget(dbo, KIND_DBAPI, SOURCE_DBAPI_CONNECTION,
                        petl_owns_connection=False, petl_owns_cursor=True,
                        is_clickhouse=True)

    if _is_dbapi_connection(dbo):
        debug('assuming %r is standard DB-API 2.0 connection', dbo)
        return DbTarget(dbo, KIND_DBAPI, SOURCE_DBAPI_CONNECTION,
                        petl_owns_connection=False, petl_owns_cursor=True)

    if _is_dbapi_cursor(dbo):
        debug('assuming %r is standard DB-API 2.0 cursor', dbo)
        warning('using a DB-API cursor with petl database helpers is not '
                'recommended and may lead to unexpected results, a DB-API '
                'connection is better')
        return DbTarget(dbo, KIND_DBAPI, SOURCE_DBAPI_CURSOR,
                        petl_owns_connection=False, petl_owns_cursor=False)

    if _is_sqlalchemy_engine(dbo):
        debug('assuming %r is an sqlalchemy Engine', dbo)
        return DbTarget(dbo, KIND_SQLALCHEMY, SOURCE_SQLALCHEMY_ENGINE,
                        petl_owns_connection=True, petl_owns_cursor=False)

    if _is_sqlalchemy_session(dbo):
        debug('assuming %r is an sqlalchemy Session', dbo)
        return DbTarget(dbo, KIND_SQLALCHEMY, SOURCE_SQLALCHEMY_SESSION,
                        petl_owns_connection=False, petl_owns_cursor=False)

    if _is_sqlalchemy_connection(dbo):
        debug('assuming %r is an sqlalchemy Connection', dbo)
        return DbTarget(dbo, KIND_SQLALCHEMY,
                        SOURCE_SQLALCHEMY_CONNECTION,
                        petl_owns_connection=False, petl_owns_cursor=False)

    if callable(dbo):
        debug('assuming %r is a function returning DB-API 2.0 cursors', dbo)
        return DbTarget(dbo, KIND_DBAPI, SOURCE_DBAPI_CURSOR_FACTORY,
                        petl_owns_connection=False, petl_owns_cursor=True)

    raise ArgumentError('unsupported database object type: %r' % dbo)


def open_sqlite_file_target(filename):
    """Open the sqlite3 connection backing a file-name convenience entry.

    The returned target records that petl owns the connection, so every
    iteration / write / DDL operation will close it again. File names are
    only supported by fromdb/todb/appenddb, never by create_table/drop_table.
    """
    import sqlite3
    connection = sqlite3.connect(filename)
    return DbTarget(connection, KIND_DBAPI, SOURCE_SQLITE_FILE,
                    petl_owns_connection=True, petl_owns_cursor=True)


# ---------------------------------------------------------------------------
# Resource lifecycle
# ---------------------------------------------------------------------------

class CleanupError(Exception):
    """Raised when a cleanup step fails without an earlier business error.

    Cleanup failures that happen *after* a business exception has already
    been raised are attached to that exception via :meth:`ResourceStack`
    logging / exception chaining instead, so the original failure remains the
    exception that propagates.
    """


class _Resource(object):
    """A single acquired resource together with its release callable."""

    __slots__ = ('name', 'resource', 'release', 'released')

    def __init__(self, name, resource, release):
        self.name = name
        self.resource = resource
        self.release = release
        self.released = False


class ResourceStack(object):
    """LIFO registry of resources acquired during one operation.

    Every resource is registered *after* it has been acquired successfully,
    so a failure while acquiring resource N still releases resources
    ``0..N-1``. Each resource is released at most once. Business exceptions
    always take precedence over cleanup exceptions: secondary cleanup errors
    are logged and chained onto the propagating exception, while a cleanup
    failure on an otherwise successful operation raises :class:`CleanupError`.
    """

    def __init__(self):
        self._resources = []

    def register(self, name, resource, release):
        """Record an already-acquired resource and its release callable."""
        entry = _Resource(name, resource, release)
        self._resources.append(entry)
        return resource

    def detach(self, resource):
        """Remove the entry for ``resource`` without releasing it.

        Used when ownership of an already-registered resource transfers to
        another wrapper (e.g. a DB-API cursor whose closing is now driven by
        the query result). Guarantees the physical resource keeps exactly one
        release path.
        """
        for pos in range(len(self._resources) - 1, -1, -1):
            entry = self._resources[pos]
            if entry.resource is resource:
                del self._resources[pos]
                return entry

    def release_for(self, resource):
        """Release and remove the entry registered for ``resource``.

        Used by the truncate-then-resurrect cursor rule. No-op if the
        resource is not currently held (e.g. a borrowed cursor that was never
        registered because petl does not own it).
        """
        for pos in range(len(self._resources) - 1, -1, -1):
            entry = self._resources[pos]
            if entry.resource is resource and not entry.released:
                entry.released = True
                entry.release(entry.resource)
                del self._resources[pos]
                return
            if entry.resource is resource and entry.released:
                del self._resources[pos]
                return

    def release_all(self, prior_error=None):
        """Release every still-held resource in reverse acquisition order.

        Returns the exception that should propagate: ``prior_error`` when one
        was already in flight (cleanup errors are logged and attached as
        suppressed context), otherwise ``None`` on success or a
        :class:`CleanupError` describing the first cleanup failure.
        """
        cleanup_errors = []
        while self._resources:
            entry = self._resources.pop()
            if entry.released:
                continue
            entry.released = True
            try:
                debug('releasing resource %r: %r', entry.name, entry.resource)
                entry.release(entry.resource)
            except Exception as exc:
                logger.exception(
                    'error while releasing resource %r (%r)',
                    entry.name, entry.resource)
                cleanup_errors.append((entry.name, exc))

        if prior_error is not None:
            # The business exception must remain the propagated one. Attach
            # secondary cleanup errors as discoverable context.
            for name, exc in cleanup_errors:
                prior_error.__suppress_context__ = False
                logger.error('cleanup error for %r suppressed in favour of '
                             'business exception', name, exc_info=exc)
            return prior_error

        if cleanup_errors:
            name, exc = cleanup_errors[0]
            err = CleanupError('resource cleanup failed for %r' % name)
            err.__cause__ = exc
            for other_name, other_exc in cleanup_errors[1:]:
                logger.error('additional cleanup error for %r', other_name,
                             exc_info=other_exc)
            return err
        return None


# ---------------------------------------------------------------------------
# Transaction scopes
# ---------------------------------------------------------------------------

class TransactionScope(object):
    """Abstract ownership-aware transaction boundary for one write/DDL op.

    Concrete scopes implement :meth:`begin`, :meth:`commit` and
    :meth:`rollback`; :meth:`finish` guarantees the right terminal call given
    whether the business operation succeeded. Read paths never construct a
    scope.

    ``commit_requested`` mirrors the public ``commit`` flag. When it is false
    petl makes no active transaction calls at all.
    """

    def __init__(self, commit_requested):
        self.commit_requested = commit_requested
        self._begun = False
        # Finalization order differs by protocol:
        # * DB-API: cursors are released first, then the connection commit
        #   (the legacy order; the connection itself is borrower-owned or
        #   closed separately for sqlite files).
        # * SQLAlchemy: the transaction must be completed while the
        #   Connection is still open, only afterwards is a self-acquired
        #   Connection returned to the pool.
        self.scope_before_resources = False

    @property
    def begun(self):
        return self._begun

    def begin(self):
        if not self.commit_requested:
            debug('commit=False: petl does not manage the transaction')
            return
        self._begun = True
        self._do_begin()

    def commit(self):
        if not self._begun:
            return
        self._do_commit()
        self._begun = False

    def rollback(self):
        if not self._begun:
            return
        try:
            self._do_rollback()
        finally:
            self._begun = False

    def finish(self, success):
        if success:
            self.commit()
        else:
            self.rollback()

    # --- protocol ---------------------------------------------------------
    def _do_begin(self):
        raise NotImplementedError

    def _do_commit(self):
        raise NotImplementedError

    def _do_rollback(self):
        raise NotImplementedError


class NoTransactionScope(TransactionScope):
    """Scope that never issues transaction calls (commit=False)."""

    def _do_begin(self):
        pass

    def _do_commit(self):
        pass

    def _do_rollback(self):
        pass


class DBAPITransactionScope(TransactionScope):
    """Owns commit/rollback on a DB-API connection.

    Used for borrowed DB-API connections/cursors/factories and for sqlite
    file connections. petl never calls ``begin()`` on the driver itself:
    drivers/SQLAlchemy autobegin implicitly, and the contract only requires a
    single terminal commit (success) or rollback (failure, only for
    petl-owned sqlite file connections).
    """

    def __init__(self, commit_requested, connection, rollback_on_failure):
        super(DBAPITransactionScope, self).__init__(commit_requested)
        self._connection = connection
        # Borrowed caller connections: on failure petl must not roll back the
        # caller's transaction. sqlite file connections are petl-owned, so a
        # failed operation rolls back before closing.
        self._rollback_on_failure = rollback_on_failure

    def _do_begin(self):
        # No explicit begin: DB-API drivers start transactions implicitly.
        debug('relying on implicit DB-API transaction on %r', self._connection)

    def _do_commit(self):
        debug('commit DB-API transaction on %r', self._connection)
        self._connection.commit()

    def _do_rollback(self):
        if self._rollback_on_failure:
            debug('rollback DB-API transaction on %r', self._connection)
            self._connection.rollback()
        else:
            debug('leaving failed transaction for caller to handle on %r',
                  self._connection)


class SQLAlchemyTransactionScope(TransactionScope):
    """Transaction boundary for SQLAlchemy Connection/Session writes.

    Ownership rules (see module docstring):

    * ``commit=False``: no transaction calls whatsoever.
    * A transaction already exists when the operation starts: petl opens a
      nested transaction (SAVEPOINT) and only ever completes that scope, so
      the outer transaction is neither committed nor invalidated.
    * No existing transaction:
        - Engine/self-acquired Connection or borrowed Connection: petl begins
          and completes its own root transaction.
        - Session: petl must establish ownership *before* calling
          ``session.connection()`` (which triggers autobegin); the
          petl-induced outer transaction is bracketed with a nested scope so
          that rollback cannot leave the caller-facing Session on an inactive
          root transaction. The Session itself is never committed or rolled
          back by petl.
    """

    def __init__(self, commit_requested, connection, nested):
        super(SQLAlchemyTransactionScope, self).__init__(commit_requested)
        self._connection = connection
        self._nested = nested
        self._transaction = None
        # Complete the SQLAlchemy transaction while the Connection is open,
        # then release resources (returning an engine Connection last).
        self.scope_before_resources = True

    def _do_begin(self):
        if self._nested:
            debug('begin nested SQLAlchemy transaction (savepoint) on %r',
                  self._connection)
            self._transaction = self._connection.begin_nested()
        else:
            debug('begin SQLAlchemy transaction on %r', self._connection)
            self._transaction = self._connection.begin()

    def _do_commit(self):
        debug('commit SQLAlchemy transaction scope %r', self._transaction)
        self._transaction.commit()

    def _do_rollback(self):
        debug('rollback SQLAlchemy transaction scope %r', self._transaction)
        self._transaction.rollback()


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class DbAdapter(object):
    """Common adapter interface used by read, DML and DDL operations.

    An adapter is bound to a resolved :class:`DbTarget` and a
    :class:`ResourceStack`. It exposes the smallest executable surface the
    three paths need and centralises every ownership decision.
    """

    def __init__(self, target, resources):
        self.target = target
        self.resources = resources

    # -- read --------------------------------------------------------------
    def open_query(self, query, args, kwargs):
        """Execute ``query`` and return a result context.

        The returned object implements ``header()``, ``__iter__`` (streaming
        native rows) and ``close()`` and is also registered with the
        resource stack.
        """
        raise NotImplementedError

    # -- write / DDL -------------------------------------------------------
    def execute_statement(self, statement, parameters=None, execute_kwargs=None):
        raise NotImplementedError

    def executemany(self, statement, rows):
        raise NotImplementedError

    def make_transaction_scope(self, commit_requested):
        raise NotImplementedError


class DBAPIAdapter(DbAdapter):
    """Adapter for DB-API connections, cursors and cursor factories."""

    def __init__(self, target, resources):
        super(DBAPIAdapter, self).__init__(target, resources)
        self._connection = None
        self._cursor = None

    # -- resource acquisition ---------------------------------------------
    def _connection_for_placeholders(self):
        """Return the DB-API connection used for paramstyle discovery."""
        source = self.target.source
        if source == SOURCE_DBAPI_CONNECTION or source == SOURCE_SQLITE_FILE:
            return self.target.obj
        # Cursor and factory targets expose the connection through the
        # optional DB-API ``cursor.connection`` attribute.
        return self._ensure_cursor().connection

    def get_paramstyle_connection(self):
        return self._connection_for_placeholders()

    def _ensure_cursor(self):
        if self._cursor is not None:
            return self._cursor
        source = self.target.source
        if source == SOURCE_DBAPI_CURSOR:
            # Borrowed cursor: petl did not create it and must not close it.
            self._cursor = self.target.obj
        elif source == SOURCE_DBAPI_CURSOR_FACTORY:
            cursor = self.target.obj()
            self.resources.register('dbapi cursor (factory)', cursor,
                                    _close_cursor)
            self._cursor = cursor
        else:
            # Connection (borrowed or petl-owned sqlite file).
            cursor = self.target.obj.cursor()
            self.resources.register('dbapi cursor', cursor, _close_cursor)
            self._cursor = cursor
        return self._cursor

    def new_cursor(self):
        """Acquire a fresh cursor, closing the previous petl-owned one.

        This implements the truncate-then-resurrect rule. Borrowed cursors
        are never closed or replaced. The replacement cursor is registered
        with the same resource stack, so a later load failure still closes
        it.

        A failure while closing the old cursor is treated like any other
        business-path error: it propagates to the runner (which still drives
        the transaction/resource finalization), rather than being swallowed
        here.
        """
        if self._cursor is None:
            return self._ensure_cursor()
        if not self.target.petl_owns_cursor:
            # Directly borrowed cursor: keep reusing it.
            return self._cursor
        old = self._cursor
        # Close and remove the pre-truncate cursor through the stack. If its
        # close() fails the exception propagates from here; the runner's
        # finalization still releases every remaining held resource.
        self.resources.release_for(old)
        self._cursor = None
        # ...then acquire a registered replacement.
        return self._ensure_cursor()

    # -- read --------------------------------------------------------------
    def open_query(self, query, args, kwargs):
        cursor = self._ensure_cursor()
        owns_cursor = self.target.petl_owns_cursor
        if owns_cursor:
            # The cursor's lifecycle from this point on is owned by the
            # query result: detach the plain cursor entry so the cursor has
            # exactly one release path (the result's stack entry below).
            self.resources.detach(cursor)
        result = DBAPIQueryResult(cursor, query, args, kwargs,
                                  owns_cursor=owns_cursor,
                                  resources=self.resources)
        # Register before executing so an execute failure still releases the
        # cursor (and any other held resources) via the stack.
        self.resources.register('dbapi query result', result, _close_result)
        result.execute()
        return result

    # -- write / DDL -------------------------------------------------------
    def execute_statement(self, statement, parameters=None, execute_kwargs=None):
        """Run a single statement, forwarding parameters/kwargs verbatim."""
        cursor = self._ensure_cursor()
        execute_kwargs = execute_kwargs or {}
        if parameters is None:
            cursor.execute(statement, **execute_kwargs)
        else:
            cursor.execute(statement, parameters, **execute_kwargs)

    def executemany(self, statement, rows):
        cursor = self._ensure_cursor()
        # Rows may be a generator: pass the iterator straight through without
        # materialising it, preserving the streaming contract.
        cursor.executemany(statement, rows)

    def make_transaction_scope(self, commit_requested):
        if not commit_requested:
            # Explicit object so tests can assert that no begin/commit/
            # rollback call is ever issued when commit=False.
            return NoTransactionScope(False)
        connection = self._connection_for_placeholders()
        rollback_on_failure = self.target.source == SOURCE_SQLITE_FILE
        return DBAPITransactionScope(commit_requested, connection,
                                     rollback_on_failure)


def _close_cursor(cursor):
    cursor.close()


def _close_result(result):
    result.close()


class DBAPIQueryResult(object):
    """Streaming wrapper around an executed DB-API cursor.

    The header is only fetched on the first row pull so drivers that populate
    ``cursor.description`` lazily keep working. Iterable cursors are consumed
    lazily; the legacy ``fetchall`` fallback is retained for non-iterable
    cursors and documented as loading the whole result set into memory.
    """

    def __init__(self, cursor, query, args, kwargs, owns_cursor, resources):
        self.cursor = cursor
        self.query = query
        self.args = args
        self.kwargs = kwargs
        self.owns_cursor = owns_cursor
        self._resources = resources
        self._started = False
        self._executed = False
        self._iterator = None
        self._fallback_rows = None
        self._closed = False

    def execute(self):
        # Separated from construction so the result (and therefore the
        # cursor) can be registered with the resource stack before the
        # driver call, keeping execute failures inside the cleanup contract.
        if self._executed:
            return
        self._executed = True
        debug('execute query %r', self.query)
        if self.args:
            self.cursor.execute(self.query, *self.args, **self.kwargs)
        else:
            self.cursor.execute(self.query, **self.kwargs)

    def _start(self):
        if self._started:
            return
        self._started = True
        cursor = self.cursor
        try:
            self._iterator = iter(cursor)
        except TypeError:
            # Legacy driver whose cursor is not iterable. Memory boundary:
            # fetchall() materialises the entire result set client-side.
            warning('DB-API cursor %r is not iterable; falling back to '
                    'fetchall(), which loads the whole result set into '
                    'memory', cursor)
            self._fallback_rows = iter(cursor.fetchall())
            self._iterator = self._fallback_rows

    def header(self):
        self._start()
        # Fetch one row first to force population of cursor.description for
        # drivers with postponed description (e.g. server-side cursors).
        try:
            first_row = next(self._iterator)
        except StopIteration:
            first_row = None
        hdr = tuple(d[0] for d in self.cursor.description)
        return hdr, first_row, self._iterator

    def __iter__(self):
        # Used by callers that want raw streaming without the header.
        self._start()
        return self._iterator

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.owns_cursor:
            self.cursor.close()


class SQLAlchemyAdapter(DbAdapter):
    """Adapter for SQLAlchemy engines, connections and sessions (1.4 legacy).

    The engine/session acquisition happens lazily on first use so merely
    constructing a ``fromdb`` view never connects.
    """

    def __init__(self, target, resources):
        super(SQLAlchemyAdapter, self).__init__(target, resources)
        self._connection = None
        self._session = None
        self._session_had_transaction = None

    # -- acquisition -------------------------------------------------------
    def _acquire_connection(self):
        if self._connection is not None:
            return self._connection
        source = self.target.source
        obj = self.target.obj
        if source == SOURCE_SQLALCHEMY_ENGINE:
            connection = obj.connect()
            self.resources.register('sqlalchemy connection (engine)',
                                    connection, _close_sa_connection)
            self._connection = connection
        elif source == SOURCE_SQLALCHEMY_SESSION:
            # Ownership must be established before connection() triggers
            # autobegin on a fresh session.
            self._session_had_transaction = obj.in_transaction()
            connection = obj.connection()
            self._session = obj
            self._connection = connection
            # Borrowed session: do not register any closing action.
        else:
            # Borrowed connection: petl neither closes nor disposes it.
            self._connection = obj
        return self._connection

    def get_paramstyle_connection(self):
        connection = self._acquire_connection()
        # Reach the raw DB-API connection exactly as the historical code did.
        return connection.connection.connection

    # -- read --------------------------------------------------------------
    def open_query(self, query, args, kwargs):
        connection = self._acquire_connection()
        result = SQLAlchemyQueryResult(connection, query, args, kwargs,
                                       resources=self.resources)
        # Register before executing: a failed execute still closes the result
        # (no-op safe) and, crucially, releases any petl-acquired connection.
        # LIFO ordering guarantees the Result is closed before the
        # self-acquired Connection is returned to the pool.
        self.resources.register('sqlalchemy result', result, _close_result)
        result.execute()
        return result

    # -- write / DDL -------------------------------------------------------
    def execute_statement(self, statement, parameters=None, execute_kwargs=None):
        connection = self._acquire_connection()
        execute_kwargs = execute_kwargs or {}
        if parameters is None:
            result = connection.execute(statement, **execute_kwargs)
        else:
            result = connection.execute(statement, parameters,
                                        **execute_kwargs)
        self.resources.register('sqlalchemy write result', result,
                                _close_sa_result)

    def executemany(self, statement, rows):
        # The historical implementation issues one execute per row; keep the
        # same SQL/parameter behaviour while streaming the source iterator.
        # Register each Result immediately so a failure on row N still closes
        # the Results from rows 0..N-1.
        connection = self._acquire_connection()
        for row in rows:
            result = connection.execute(statement, row)
            self.resources.register('sqlalchemy write result', result,
                                    _close_sa_result)

    def make_transaction_scope(self, commit_requested):
        connection = self._acquire_connection()
        if not commit_requested:
            # No transaction object is begun, committed or rolled back; the
            # connection is still released (returned to the pool) by the
            # resource stack.
            return NoTransactionScope(False)
        source = self.target.source
        if source == SOURCE_SQLALCHEMY_SESSION:
            # Whether or not the Session had a transaction before petl's
            # connection() call, always use a nested (savepoint) scope: if an
            # outer transaction exists only the savepoint is completed, and if
            # petl itself triggered autobegin the savepoint lets a rollback
            # succeed without leaving the Session's root transaction
            # inactive. The Session is never committed/rolled back by petl.
            return SQLAlchemyTransactionScope(
                commit_requested, connection, nested=True)
        if source == SOURCE_SQLALCHEMY_CONNECTION:
            nested = bool(connection.in_transaction())
            return SQLAlchemyTransactionScope(commit_requested, connection,
                                              nested=nested)
        # Engine: petl acquired this connection, so it owns a root scope.
        return SQLAlchemyTransactionScope(commit_requested, connection,
                                          nested=False)


def _close_sa_connection(connection):
    # Returns the connection to the pool; never disposes the engine.
    connection.close()


def _close_sa_result(result):
    if not getattr(result, 'closed', False):
        result.close()


class SQLAlchemyQueryResult(object):
    """Streaming wrapper around a SQLAlchemy query ``Result``."""

    def __init__(self, connection, query, args, kwargs, resources):
        self._connection = connection
        self._resources = resources
        self.query = query
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self._closed = False

    def execute(self):
        # Deliberately separate from construction so the adapter can put the
        # result on the resource stack before the driver call executes.
        debug('execute sqlalchemy query %r', self.query)
        if self.args:
            self.result = self._connection.execute(self.query, *self.args,
                                                   **self.kwargs)
        else:
            self.result = self._connection.execute(self.query, **self.kwargs)

    def header(self):
        hdr = tuple(self.result.keys())
        return hdr, None, iter(self.result)

    def __iter__(self):
        return iter(self.result)

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Close the Result before releasing any petl-acquired connection, as
        # required by the ownership contract.
        if self.result is not None:
            self.result.close()


def _adapter_factory(target):
    if target.kind == KIND_DBAPI:
        return DBAPIAdapter
    return SQLAlchemyAdapter


def make_adapter(target, resources=None):
    """Build the adapter for a resolved target over a resource stack."""
    if resources is None:
        resources = ResourceStack()
    return _adapter_factory(target)(target, resources)
