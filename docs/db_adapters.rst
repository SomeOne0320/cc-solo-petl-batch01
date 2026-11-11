.. _db_resource_adapters:

Database resource adapter layer
================================

.. currentmodule:: petl.io.db_adapters

The database read (:func:`petl.io.db.fromdb`), DML
(:func:`petl.io.db.todb` / :func:`petl.io.db.appenddb`) and DDL
(:func:`petl.io.db_create.create_table` /
:func:`petl.io.db_create.drop_table`) paths all run through one internal
adapter layer instead of each keeping its own object-type dispatch and
resource handling. This page documents the ownership and lifecycle rules that
layer enforces.

Modules
-------

* :mod:`petl.io.db_adapters` - object resolution, the resource stack,
  transaction scopes and the DB-API / SQLAlchemy adapters.
* :mod:`petl.io.db_planning` - identifier quoting, paramstyle placeholders and
  the shared DELETE/INSERT (and ClickHouse) SQL plans. It performs no
  execution and owns no resources.
* :mod:`petl.io.db_runner` - the single DML/DDL execution driver that combines
  a plan, an adapter, a transaction scope and the resource stack.

``resolve_target(dbo)`` returns a :class:`DbTarget` recording the object's
provenance (``source``), its executable protocol (``kind``) and whether petl
owns the underlying connection and/or cursors. The ClickHouse connection check
runs first, preserving its dedicated SQL branch. The same resolution is used
by reads, DML and DDL.

Mapping from the legacy paths
-----------------------------

The responsibilities that used to be duplicated across the per-object-type
branches in ``petl/io/db.py`` and ``petl/io/db_create.py`` now have a single
home:

========================================== =====================================
Legacy responsibility                      New location
========================================== =====================================
``_is_dbapi_*`` / ``_is_sqlalchemy_*``      :func:`resolve_target` producing a
dispatch (separate copies in both modules)  :class:`DbTarget`, shared by all
                                            paths
``_iter_dbapi_*`` / ``_iter_sqlalchemy_*``  :class:`DBAPIAdapter` /
read generators                             :class:`SQLAlchemyAdapter`
                                            ``open_query`` and the
                                            ``_iter_target`` generator in
                                            :mod:`petl.io.db`
``_todb_dbapi_*`` / ``_todb_sqlalchemy_*``  :func:`petl.io.db_runner.run_dml`
write branches                              driving one adapter + one
                                            :class:`~petl.io.db_planning.DMLPlan`
``_execute_dbapi_*`` /                      :func:`petl.io.db_runner.run_ddl`
``_execute_sqlalchemy_*`` DDL branches
duplicated DELETE/INSERT string building    :mod:`petl.io.db_planning`
(including the ClickHouse variant)          (``build_dml_plan``)
ad-hoc ``try/finally`` cursor and           :class:`ResourceStack`,
connection handling                         :class:`TransactionScope` and the
                                            shared ``_finalize`` driver
========================================== =====================================


Object sources and ownership
----------------------------

============================= =============== =============== =======================
Object                        Connection owner Cursor owner    Transaction owner
============================= =============== =============== =======================
DB-API connection             caller          petl-created    caller / petl on commit
DB-API cursor (borrowed)      caller          caller          caller
cursor factory (callable)     caller          petl-created    caller / petl on commit
sqlite file name              petl            petl            petl
SQLAlchemy Engine             petl (checked   n/a             petl scope
                              out Connection)
SQLAlchemy Connection         caller          n/a             savepoint if outer txn
SQLAlchemy Session            caller          n/a             savepoint; Session left
                                                             to the caller
============================= =============== =============== =======================

Borrowed connections, cursors, Connections and Sessions are never closed or
disposed by petl. Resources petl acquires itself are always released, on
success and on failure, each exactly once.

Read lifecycle
--------------

Constructing a :func:`~petl.io.db.fromdb` view and calling :func:`iter` on it
never connect, open a cursor or execute SQL. Resources are acquired only on
the first row pull. Each iteration manages its own resources, so two
iterators over a connection, cursor factory, sqlite file or Engine can be
interleaved and closed independently. A directly borrowed DB-API cursor is the
exception: it is a single caller-managed cursor and cannot support concurrent
iterations.

* Cursors petl creates (DB-API connection / factory / sqlite file) are closed
  when the iterator is exhausted, explicitly closed, or fails; the caller's
  connection is left open.
* A borrowed DB-API cursor is never closed by petl.
* For an Engine each iteration checks out and returns its own Connection; the
  Engine is never disposed. The query ``Result`` is closed before the
  Connection is returned.
* Borrowed Connections/Sessions are never closed, committed or rolled back.
* Reads never manage transactions.
* Iterable cursors are streamed row by row. Cursors that are not iterable use
  the legacy ``fetchall()`` fallback, which materialises the whole result set
  in client memory.
* The header is emitted as a ``tuple``; data rows are the driver's native row
  objects and are not coerced into tuples or dicts. Drivers that populate
  ``cursor.description`` lazily are supported, and an empty result still
  yields the header.

Write and DDL transaction matrix
---------------------------------

"Active" calls below are explicit ``begin``/``commit``/``rollback`` calls made
by petl. Driver and SQLAlchemy behaviour such as autobegin, autoflush, legacy
autocommit and the implicit rollback when a pooled connection is returned are
not active petl calls and are separate from this matrix.

DB-API connection, borrowed cursor or cursor factory
    With ``commit=False`` petl issues no begin/commit/rollback calls. With
    ``commit=True`` a successful DML load or a standalone DDL statement is
    committed once after all work succeeds. On an execution or source-row
    error petl does not commit and does not roll back the caller's
    transaction; the transaction state is left for the caller to resolve.
    The caller's connection and any directly borrowed cursor are never
    closed. A factory that returns new cursors over the same underlying
    connection follows the same contract.

sqlite file name
    The connection is created and always closed by petl. With ``commit=True``
    a success commits and a failure rolls back before closing. With
    ``commit=False`` petl makes no active transaction calls and only releases
    its own resources; persistence under a driver-level autocommit mode is not
    guaranteed.

SQLAlchemy Engine
    Only the Connection petl checked out is closed (returned to the pool); the
    Engine is never disposed. With ``commit=True`` petl begins an operation
    transaction that commits on success and rolls back on failure. With
    ``commit=False`` no transaction is begun, committed or rolled back, while
    the Connection is still returned.

Borrowed SQLAlchemy Connection
    Whether a transaction already exists is checked before the operation.
    With ``commit=False`` the caller fully owns the transaction. With
    ``commit=True`` and no existing transaction petl begins and completes its
    own root transaction. If a transaction already exists petl opens a nested
    transaction (SAVEPOINT) and only completes that scope, so the outer
    transaction is neither committed nor invalidated.

Borrowed SQLAlchemy Session
    Ownership is established *before* calling ``session.connection()``, which
    itself triggers autobegin on a fresh Session. With ``commit=False`` petl
    makes no transaction calls. With ``commit=True`` petl always brackets its
    work in a nested transaction: when the caller already had a transaction
    only the savepoint is completed, and when petl triggered the autobegin the
    savepoint still allows a clean rollback without leaving the Session's root
    transaction inactive. The Session itself is never committed or rolled back
    by petl, and its ORM flush/expire policy is unchanged.

``todb(create=True, drop=True)`` keeps the original order - drop, create, then
load - with each step following the matrix above. There is no cross-statement
atomicity guarantee, and not every database supports transactional DDL.

Errors and cleanup
------------------

Every resource is registered with a :class:`ResourceStack` only after it has
been acquired, so a failure while acquiring resource *N* still releases
resources ``0..N-1``, and each resource is released at most once in LIFO
order. For an overwriting load the DELETE step closes and recreates a
petl-owned cursor before the INSERT; a directly borrowed cursor is neither
closed nor replaced.

Source rows are streamed, never fully materialised, so a source generator
that raises after the header or mid-load still drives the same cleanup. If a
business error has already been raised, a later rollback/close failure does
not replace it: the original exception propagates and the secondary error is
logged and attached as exception context. If cleanup alone fails on an
otherwise successful operation, a
:class:`~petl.io.db_adapters.CleanupError` is raised rather than swallowing the
failure. Cleanup never relies on garbage collection, weak references or
interpreter shutdown.
