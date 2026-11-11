# -*- coding: utf-8 -*-
"""Shared execution machinery for database writes and DDL.

Both :func:`petl.io.db.todb`/:func:`petl.io.db.appenddb` (DML) and
:func:`petl.io.db_create.create_table`/:func:`petl.io.db_create.drop_table`
(DDL) execute through the helpers here, so the resource ordering,
transaction ownership and error/cleanup precedence rules live in exactly one
place and are genuinely shared rather than copied per object type.
"""
from __future__ import absolute_import, print_function, division

import logging

from petl.io.db_adapters import ResourceStack, make_adapter, CleanupError
from petl.io.db_planning import build_dml_plan


logger = logging.getLogger(__name__)
debug = logger.debug


def _finalize(scope, resources, error):
    """Run every terminal step and resolve the propagated exception.

    Ordering mirrors the legacy implementation: all resources are released
    (LIFO) first, then the transaction scope is completed. Both stages are
    themselves exception-guarded so a rollback or close failure cannot
    prevent the remaining steps from running. Precedence:

    * if a business exception exists, it is always the propagated
      exception; any rollback/cleanup exception is logged and attached as
      exception context;
    * if the business path succeeded but a terminal step failed, the first
      terminal failure is raised (wrapped in :class:`CleanupError` when it
      comes from resource release).
    """
    terminal_errors = []

    def run_scope():
        try:
            scope.finish(success=error is None)
        except Exception as exc:
            logger.exception('error while completing transaction scope')
            terminal_errors.append(('transaction', exc))

    def run_resources():
        # The stack keeps releasing even after a failed release and folds its
        # own errors into the precedence logic.
        outcome = resources.release_all(prior_error=error)
        if isinstance(outcome, CleanupError):
            terminal_errors.append(('cleanup', outcome))

    # Protocol-specific ordering (see TransactionScope.scope_before_resources).
    if getattr(scope, 'scope_before_resources', False):
        run_scope()
        run_resources()
    else:
        run_resources()
        run_scope()

    if error is not None:
        # Business exception always wins. Make terminal failures discoverable
        # via the standard exception context chain.
        if terminal_errors:
            name, terminal_exc = terminal_errors[0]
            logger.error('terminal %s failure suppressed in favour of business '
                         'exception', name, exc_info=terminal_exc)
            if error.__context__ is None:
                error.__context__ = terminal_exc
        return error

    if terminal_errors:
        name, terminal_exc = terminal_errors[0]
        if isinstance(terminal_exc, CleanupError):
            return terminal_exc
        wrapped = CleanupError('terminal %s step failed' % name)
        wrapped.__cause__ = terminal_exc
        for other_name, other_exc in terminal_errors[1:]:
            logger.error('additional terminal failure for %r', other_name,
                         exc_info=other_exc)
        return wrapped
    return None


def run_ddl(target, statement, commit):
    """Execute one DDL statement under the unified ownership rules."""
    resources = ResourceStack()
    adapter = make_adapter(target, resources)
    scope = adapter.make_transaction_scope(commit)
    error = None
    try:
        scope.begin()
        debug('execute DDL: %r', statement)
        adapter.execute_statement(statement)
    except Exception as exc:
        error = exc
    propagated = _finalize(scope, resources, error)
    if propagated is not None:
        raise propagated


def run_dml(target, table, tablename, schema, commit, truncate):
    """Execute one truncate/append load under the unified ownership rules.

    The source table is consumed as a streaming iterator: the header is read
    first to build the plan, then the remaining rows are handed straight to
    ``executemany`` (DB-API) or per-row ``execute`` (SQLAlchemy) without
    materialising the whole input. A failure while generating source rows
    still drives the same transaction/resource cleanup as an execution
    failure.
    """
    resources = ResourceStack()
    adapter = make_adapter(target, resources)
    scope = adapter.make_transaction_scope(commit)
    error = None
    try:
        # Resolve the header before acquiring transactional resources so a
        # broken source table fails without opening anything.
        source_it = iter(table)
        header = next(source_it)

        plan = build_dml_plan(adapter, header, tablename, schema, truncate)

        scope.begin()

        if plan.truncate_statement is not None:
            debug('truncate via %r', plan.truncate_statement)
            adapter.execute_statement(plan.truncate_statement)
            # Resurrect the cursor after truncate for self-owned DB-API
            # cursors (historical close-and-recreate rule). Borrowed cursors
            # are left untouched.
            if target.kind == 'dbapi' and target.petl_owns_cursor:
                adapter.new_cursor()

        debug('insert rows via %r', plan.insert_statement)
        adapter.executemany(plan.insert_statement, source_it)
    except Exception as exc:
        error = exc
    propagated = _finalize(scope, resources, error)
    if propagated is not None:
        raise propagated


__all__ = ['run_ddl', 'run_dml']
