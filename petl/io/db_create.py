# -*- coding: utf-8 -*-
"""
Module providing some convenience functions for working with SQL databases.
SQLAlchemy is required, try ``apt-get install python-sqlalchemy``
or ``pip install SQLAlchemy``.

Acknowledgments: much of the code of this module is based on the ``csvsql``
utility in the `csvkit <https://github.com/onyxfish/csvkit>`_ package.

The DDL execution path (``create_table``/``drop_table``) shares the object
resolution, resource adapter and transaction/ownership machinery in
:mod:`petl.io.db_adapters` and :mod:`petl.io.db_runner` with the read and DML
paths in :mod:`petl.io.db`; this module only keeps the type inference and
statement generation that is specific to CREATE TABLE.
"""


import datetime
import logging
from petl.compat import long, text_type


from petl.util.materialise import columns
from petl.transform.basics import head
from petl.io.db_adapters import DbTarget, resolve_target
from petl.io.db_planning import build_drop_statement
from petl.io.db_runner import run_ddl


logger = logging.getLogger(__name__)
debug = logger.debug


DIALECTS = {
    'access': 'access.base',
    'firebird': 'firebird.kinterbasdb',
    'informix': 'informix.informixdb',
    'maxdb': 'maxdb.sapdb',
    'mssql': 'mssql.pyodbc',
    'mysql': 'mysql.mysqlconnector',
    'oracle': 'oracle.cx_oracle',
    'postgresql': 'postgresql.psycopg2',
    'sqlite': 'sqlite.pysqlite',
    'sybase': 'sybase.pyodbc'
}


SQL_INTEGER_MAX = 2147483647
SQL_INTEGER_MIN = -2147483647
NULL_COLUMN_MAX_LENGTH = 32


def make_sqlalchemy_column(col, colname, constraints=True):
    """
    Infer an appropriate SQLAlchemy column type based on a sequence of values.

    Keyword arguments:

    col : sequence
        A sequence of values to use to infer type, length etc.
    colname : string
        Name of column
    constraints : bool
        If True use length and nullable constraints

    """

    import sqlalchemy

    col_not_none = [v for v in col if v is not None]
    sql_column_kwargs = {}
    sql_type_kwargs = {}

    if len(col_not_none) == 0:
        sql_column_type = sqlalchemy.String
        if constraints:
            sql_type_kwargs['length'] = NULL_COLUMN_MAX_LENGTH

    elif all(isinstance(v, bool) for v in col_not_none):
        sql_column_type = sqlalchemy.Boolean

    elif all(isinstance(v, int) for v in col_not_none):
        if max(col_not_none) > SQL_INTEGER_MAX \
                or min(col_not_none) < SQL_INTEGER_MIN:
            sql_column_type = sqlalchemy.BigInteger
        else:
            sql_column_type = sqlalchemy.Integer

    elif all(isinstance(v, long) for v in col_not_none):
        sql_column_type = sqlalchemy.BigInteger

    elif all(isinstance(v, (int, long)) for v in col_not_none):
        sql_column_type = sqlalchemy.BigInteger

    elif all(isinstance(v, (int, long, float)) for v in col_not_none):
        sql_column_type = sqlalchemy.Float

    elif all(isinstance(v, datetime.datetime) for v in col_not_none):
        sql_column_type = sqlalchemy.DateTime

    elif all(isinstance(v, datetime.date) for v in col_not_none):
        sql_column_type = sqlalchemy.Date

    elif all(isinstance(v, datetime.time) for v in col_not_none):
        sql_column_type = sqlalchemy.Time

    elif all(isinstance(v, (dict, list)) for v in col_not_none):
        sql_column_type = sqlalchemy.JSON

    else:
        sql_column_type = sqlalchemy.String
        if constraints:
            sql_type_kwargs['length'] = max([len(text_type(v)) for v in col])

    if constraints:
        sql_column_kwargs['nullable'] = len(col_not_none) < len(col)

    return sqlalchemy.Column(colname, sql_column_type(**sql_type_kwargs),
                             **sql_column_kwargs)


def make_sqlalchemy_table(table, tablename, schema=None, constraints=True,
                          metadata=None):
    """
    Create an SQLAlchemy table definition based on data in `table`.

    Keyword arguments:

    table : table container
        Table data to use to infer types etc.
    tablename : text
        Name of the table
    schema : text
        Name of the database schema to create the table in
    constraints : bool
        If True use length and nullable constraints
    metadata : sqlalchemy.MetaData
        Custom table metadata

    """

    import sqlalchemy

    if not metadata:
        metadata = sqlalchemy.MetaData()

    sql_table = sqlalchemy.Table(tablename, metadata, schema=schema)
    cols = columns(table)
    flds = list(cols.keys())
    for f in flds:
        sql_column = make_sqlalchemy_column(cols[f], f,
                                            constraints=constraints)
        sql_table.append_column(sql_column)

    return sql_table


def make_create_table_statement(table, tablename, schema=None,
                                constraints=True, metadata=None, dialect=None):
    """
    Generate a CREATE TABLE statement based on data in `table`.

    Keyword arguments:

    table : table container
        Table data to use to infer types etc.
    tablename : text
        Name of the table
    schema : text
        Name of the database schema to create the table in
    constraints : bool
        If True use length and nullable constraints
    metadata : sqlalchemy.MetaData
        Custom table metadata
    dialect : text
        One of {'access', 'sybase', 'sqlite', 'informix', 'firebird', 'mysql',
        'oracle', 'maxdb', 'postgresql', 'mssql'}

    """

    import sqlalchemy
    sql_table = make_sqlalchemy_table(table, tablename, schema=schema,
                                      constraints=constraints,
                                      metadata=metadata)

    if dialect:
        module = __import__('sqlalchemy.dialects.%s' % DIALECTS[dialect],
                            fromlist=['dialect'])
        sql_dialect = module.dialect()
    else:
        sql_dialect = None

    return text_type(sqlalchemy.schema.CreateTable(sql_table)
                     .compile(dialect=sql_dialect)).strip()


def _as_target(dbo):
    """Accept either a raw database object or an already resolved target."""
    if isinstance(dbo, DbTarget):
        return dbo
    # Internal carrier produced by petl.io.db for todb(create=...).
    wrapped = getattr(dbo, 'target', None)
    if isinstance(wrapped, DbTarget):
        return wrapped
    return resolve_target(dbo)


def create_table(table, dbo, tablename, schema=None, commit=True,
                 constraints=True, metadata=None, dialect=None, sample=1000):
    """
    Create a database table based on a sample of data in the given `table`.

    Keyword arguments:

    table : table container
        Table data to load
    dbo : database object
        DB-API 2.0 connection, callable returning a DB-API 2.0 cursor, or
        SQLAlchemy connection, engine or session
    tablename : text
        Name of the table
    schema : text
        Name of the database schema to create the table in
    commit : bool
        If True commit the changes
    constraints : bool
        If True use length and nullable constraints
    metadata : sqlalchemy.MetaData
        Custom table metadata
    dialect : text
        One of {'access', 'sybase', 'sqlite', 'informix', 'firebird', 'mysql',
        'oracle', 'maxdb', 'postgresql', 'mssql'}
    sample : int
        Number of rows to sample when inferring types etc., set to 0 to use
        the whole table

    Execution goes through the shared adapter layer: cursors petl creates are
    closed on success and failure, SQLAlchemy connections checked out from an
    Engine are returned, and the transaction follows the same ownership rules
    as :func:`petl.io.db.todb`.

    """

    if sample > 0:
        table = head(table, sample)
    sql = make_create_table_statement(table, tablename, schema=schema,
                                      constraints=constraints,
                                      metadata=metadata, dialect=dialect)
    target = _as_target(dbo)
    _execute(sql, target, commit=commit)


def drop_table(dbo, tablename, schema=None, commit=True):
    """
    Drop a database table.

    Keyword arguments:

    dbo : database object
        DB-API 2.0 connection, callable returning a DB-API 2.0 cursor, or
        SQLAlchemy connection, engine or session
    tablename : text
        Name of the table
    schema : text
        Name of the database schema the table is in
    commit : bool
        If True commit the changes

    """

    target = _as_target(dbo)
    sql = build_drop_statement(tablename, schema)
    _execute(sql, target, commit)


def _execute(sql, target, commit):
    """Single DDL execution entry point shared by create and drop.

    ``target`` is always a resolved :class:`~petl.io.db_adapters.DbTarget` by
    the time it reaches here; the polymorphic object dispatch lives in
    :func:`petl.io.db_adapters.resolve_target`.
    """
    debug(sql)
    run_ddl(target, sql, commit)
