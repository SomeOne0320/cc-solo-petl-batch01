# -*- coding: utf-8 -*-
"""SQL generation shared by the read, DML and DDL database paths.

This module is deliberately free of any resource or transaction handling: it
turns table/schema/column names and a resolved adapter into the concrete SQL
strings and parameter placeholders used by the drivers. Keeping generation
separate from ownership means the same DELETE/INSERT plan is used for every
DB-API connection, cursor and cursor factory, while the ClickHouse branch
keeps its dedicated statements.
"""
from __future__ import absolute_import, print_function, division

import logging

from petl.compat import text_type
from petl.io.db_utils import _quote, _placeholders


logger = logging.getLogger(__name__)
debug = logger.debug


SQL_TRUNCATE_QUERY = 'DELETE FROM %s'
SQL_INSERT_QUERY = 'INSERT INTO %s (%s) VALUES (%s)'
SQL_CLICKHOUSE_TRUNCATE_QUERY = 'TRUNCATE TABLE IF EXISTS %s'
SQL_CLICKHOUSE_INSERT_QUERY = 'INSERT INTO %s (%s) VALUES'
SQL_DROP_TABLE_QUERY = u'DROP TABLE %s'


def quote_identifier(tablename, schema=None):
    """Quote a schema-qualified table identifier using the SQL-92 rules."""
    quoted = _quote(tablename)
    if schema is not None:
        quoted = _quote(schema) + '.' + quoted
    debug('quoted table identifier: %r', quoted)
    return quoted


def quote_columns(header):
    """Quote column names taken from a table header (native values kept)."""
    flds = [text_type(f) for f in header]
    return [_quote(n) for n in flds]


class DMLPlan(object):
    """A fully resolved truncate/insert plan for one write operation.

    Attributes
    ----------
    table_ref : str
        Quoted, schema-qualified table identifier.
    columns : list of str
        Quoted column identifiers.
    placeholders : str
        Driver paramstyle placeholder list for one row.
    truncate_statement : str or None
        Statement run before inserts when truncating; ``None`` for appends.
    insert_statement : str
        The statement handed to ``executemany`` (or per-row execute under
        SQLAlchemy).
    """

    __slots__ = ('table_ref', 'columns', 'placeholders',
                 'truncate_statement', 'insert_statement')

    def __init__(self, table_ref, columns, placeholders, truncate_statement,
                 insert_statement):
        self.table_ref = table_ref
        self.columns = columns
        self.placeholders = placeholders
        self.truncate_statement = truncate_statement
        self.insert_statement = insert_statement


def build_dml_plan(adapter, header, tablename, schema, truncate):
    """Build the shared ordinary (and ClickHouse) INSERT/DELETE plan.

    ``adapter`` is only used for paramstyle discovery via its resolved
    DB-API connection; it performs no execution here.
    """
    table_ref = quote_identifier(tablename, schema)
    columns = quote_columns(header)
    paramstyle_connection = adapter.get_paramstyle_connection()
    placeholders = _placeholders(paramstyle_connection, columns)
    debug('placeholders: %r', placeholders)

    if adapter.target.is_clickhouse:
        truncate_statement = (
            SQL_CLICKHOUSE_TRUNCATE_QUERY % table_ref if truncate else None)
        insert_statement = SQL_CLICKHOUSE_INSERT_QUERY % (
            table_ref, ', '.join(columns))
    else:
        truncate_statement = (
            SQL_TRUNCATE_QUERY % table_ref if truncate else None)
        insert_statement = SQL_INSERT_QUERY % (
            table_ref, ', '.join(columns), placeholders)

    debug('insert statement: %r', insert_statement)
    return DMLPlan(table_ref, columns, placeholders, truncate_statement,
                   insert_statement)


def build_drop_statement(tablename, schema):
    return SQL_DROP_TABLE_QUERY % quote_identifier(tablename, schema)
