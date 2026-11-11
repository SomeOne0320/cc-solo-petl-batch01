from __future__ import absolute_import, division, print_function

import datetime
import operator

from petl.comparison import Comparable, comparable_itemgetter
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.transform.basics import stack
from petl.transform.sorts import sort
from petl.util.base import Table, asindices, header, rowgetter


DIRECTIONS = ('backward', 'forward', 'nearest')


def asofjoin(left, right, key, by=None, direction='backward', tolerance=None,
             allow_exact_matches=True, presorted=False, missing=None,
             lprefix=None, rprefix=None, buffersize=None, tempdir=None,
             cache=True):
    """
    Perform an as-of join on the given tables. Each row from the `left`
    table is paired with the single most relevant row from the `right` table
    whose key value is close in time to the left row's key value. This is
    useful, for example, for aligning events with the most recent state of
    the same device. E.g.::

        >>> import petl as etl
        >>> left = [('device', 'ts', 'event'),
        ...         ('A', 5, 'x'),
        ...         ('A', 10, 'y'),
        ...         ('B', 7, 'z')]
        >>> right = [('device', 'ts', 'state'),
        ...          ('A', 3, 'old'),
        ...          ('A', 8, 'new'),
        ...          ('B', 9, 'future')]
        >>> table = etl.asofjoin(left, right, key='ts', by='device')
        >>> table
        +--------+----+-------+-------+
        | device | ts | event | state |
        +========+====+=======+=======+
        | 'A'    |  5 | 'x'   | 'old' |
        +--------+----+-------+-------+
        | 'A'    | 10 | 'y'   | 'new' |
        +--------+----+-------+-------+
        | 'B'    |  7 | 'z'   | None  |
        +--------+----+-------+-------+

    The `key` argument is a single field selector (field name or zero-based
    field index) selecting the field shared by both tables that holds the
    ordering values (e.g. timestamps). Positional selectors are interpreted
    against each table's own header, so the tables may have columns in
    different orders.

    The optional `by` argument is a field selector, or a list/tuple of field
    selectors, again shared by both tables, identifying partition (e.g.
    device) fields; only right rows in the same partition as a left row are
    candidate matches. The `key` and `by` fields must not overlap. Pairing
    differently named left and right fields is not supported.

    The result is a left join: every left data row produces exactly one
    output row. All left columns are kept and the right columns other than
    the `key` and `by` columns are appended. Where no candidate right row is
    found, the appended cells are filled with `missing`. The `lprefix` and
    `rprefix` arguments, if given, are only applied to the output header
    field names, in the same style as :func:`petl.transform.joins.join`.

    The `direction` argument controls which right row is selected relative
    to the left key value:

    * ``'backward'`` (the default): the most recent right row whose key is
      not greater than the left key.
    * ``'forward'``: the next right row whose key is not less than the left
      key.
    * ``'nearest'``: the right row with the smallest absolute key distance;
      ties between a backward and a forward row are resolved in favour of
      the backward side.

    If several right rows share the same key value, the last of them in the
    stable input order is used on the backward side and the first of them on
    the forward side; an exact hit for ``'nearest'`` is treated as a
    backward hit.

    If `allow_exact_matches` is False, right rows whose key equals the left
    key are excluded before the selection is made.

    If `tolerance` is not None, it is a non-negative maximum distance
    (inclusive at the boundary) between the left key and the selected
    candidate's key; a candidate further away is rejected and the row is
    emitted with `missing` values instead. A tolerance of zero therefore
    only permits exact matches. Numeric keys use numeric distances, while
    `datetime.datetime` keys use `datetime.timedelta` distances.

    Key values must support ordering within a table (e.g. all numeric or all
    datetimes). Mixing values of mutually incomparable types, or using
    `None` or ``NaN`` key values, is not supported and gives undefined
    behaviour.

    If `presorted` is False (the default), both tables are stably sorted
    ascending by the `by` and `key` fields, reusing the `buffersize`,
    `tempdir` and `cache` behaviour of :func:`petl.transform.sorts.sort`,
    and the output left rows are emitted in that sorted order. If
    `presorted` is True, both tables are assumed to already be sorted
    ascending by the same fields and the input order of equal keys is
    preserved; in that mode the join is computed with a single lazy merge
    pass that never materialises a whole table or a whole partition, so a
    large partition with a long history only keeps the candidate needed for
    selection plus a bounded look-ahead in memory (aside from the rows
    sharing one identical key value, which may be read as a run). The
    `presorted` declaration is not checked.

    Uneven rows are squared up to the header width with `missing` before
    joining, in the same manner as the equi-joins; over-long rows follow the
    existing :func:`petl.transform.basics.stack` trimming behaviour.

    Constructing the view does not consume the source tables, and requesting
    the output header only reads the two input headers without scanning any
    data. Views over re-iterable inputs are themselves re-iterable and do not
    reuse candidate state from a previous iteration.

    .. versionadded:: 1.7.18

    """

    return AsofJoinView(
        left, right, key=key, by=by, direction=direction,
        tolerance=tolerance, allow_exact_matches=allow_exact_matches,
        presorted=presorted, missing=missing, lprefix=lprefix,
        rprefix=rprefix, buffersize=buffersize, tempdir=tempdir, cache=cache,
    )


Table.asofjoin = asofjoin


class AsofJoinView(Table):

    def __init__(self, left, right, key, by=None, direction='backward',
                 tolerance=None, allow_exact_matches=True, presorted=False,
                 missing=None, lprefix=None, rprefix=None,
                 buffersize=None, tempdir=None, cache=True):

        # arguments independent of the table headers are validated eagerly,
        # so misuse does not silently fall back to different behaviour
        if direction not in DIRECTIONS:
            raise ArgumentError(
                'direction must be one of %r, got %r'
                % (DIRECTIONS, direction))
        if not isinstance(allow_exact_matches, bool):
            raise ArgumentError(
                'allow_exact_matches must be a boolean, got %r'
                % (allow_exact_matches,))
        if tolerance is not None:
            zero = (datetime.timedelta(0)
                    if isinstance(tolerance, datetime.timedelta) else 0)
            try:
                negative = tolerance < zero
            except TypeError:
                raise ArgumentError(
                    'tolerance must be a non-negative number or timedelta, '
                    'got %r' % (tolerance,))
            if negative:
                raise ArgumentError(
                    'tolerance must be non-negative, got %r' % (tolerance,))
        if key is None or isinstance(key, (list, tuple)):
            raise ArgumentError(
                'key must be a single field selector (field name or index), '
                'compound keys are not supported, got %r' % (key,))

        self.key = key
        self.by = by
        self.direction = direction
        self.tolerance = tolerance
        self.allow_exact_matches = allow_exact_matches
        self.presorted = presorted
        self.missing = missing
        self.lprefix = lprefix
        self.rprefix = rprefix

        # square up uneven rows exactly like the equi-joins; like the other
        # views, constructing these does not consume the source tables
        self.left = stack(left, missing=missing)
        self.right = stack(right, missing=missing)

        # sorting is wrapped lazily so that constructing the view neither
        # scans data nor even reads the input headers; positional selectors
        # are resolved against each table's own header on first iteration
        if not presorted:
            self.left = _LazyAsOfSort(
                self.left, key, by, buffersize=buffersize,
                tempdir=tempdir, cache=cache)
            self.right = _LazyAsOfSort(
                self.right, key, by, buffersize=buffersize,
                tempdir=tempdir, cache=cache)

    def __iter__(self):
        return iter_asofjoin(
            self.left, self.right, self.key, by=self.by,
            direction=self.direction, tolerance=self.tolerance,
            allow_exact_matches=self.allow_exact_matches,
            missing=self.missing, lprefix=self.lprefix,
            rprefix=self.rprefix,
        )


class _LazyAsOfSort(Table):
    """Resolve the (by..., key) selectors against this table's own header
    and delegate to :func:`petl.transform.sorts.sort`, all on iteration."""

    def __init__(self, source, key, by, buffersize=None, tempdir=None,
                 cache=True):
        self.source = source
        self.key = key
        self.by = by
        self.buffersize = buffersize
        self.tempdir = tempdir
        self.cache = cache

    def __iter__(self):
        hdr = header(self.source)
        spec = _normalize_by(self.by) + (self.key,)
        indices = asindices(hdr, spec)
        sorted_view = sort(self.source, indices,
                           buffersize=self.buffersize, tempdir=self.tempdir,
                           cache=self.cache)
        return iter(sorted_view)


def _normalize_by(by):
    if by is None:
        return ()
    if isinstance(by, (list, tuple)):
        return tuple(by)
    return (by,)


def iter_asofjoin(left, right, key, by=None, direction='backward',
                  tolerance=None, allow_exact_matches=True, missing=None,
                  lprefix=None, rprefix=None):

    lit = iter(left)
    rit = iter(right)

    lhdr = next(lit)
    rhdr = next(rit)

    by_spec = _normalize_by(by)

    # field resolution is deferred to first iteration, following petl's
    # lazy convention; unknown fields and overlapping selections fail here
    lkind = asindices(lhdr, key)[0]
    rkind = asindices(rhdr, key)[0]
    lbind = asindices(lhdr, by_spec)
    rbind = asindices(rhdr, by_spec)

    if lkind in lbind or rkind in rbind:
        raise ArgumentError('key and by fields must not overlap')

    # without `by` every row belongs to one constant partition
    empty_group = Comparable(())
    lgetby = (comparable_itemgetter(*lbind)
              if lbind else lambda row: empty_group)
    rgetby = (comparable_itemgetter(*rbind)
              if rbind else lambda row: empty_group)
    lgetkey = comparable_itemgetter(lkind)
    rgetkey = comparable_itemgetter(rkind)
    # raw key values are needed for arithmetic (distances and tolerance)
    lgetraw = operator.itemgetter(lkind)
    rgetraw = operator.itemgetter(rkind)

    # right value columns: everything except the key and by fields
    rvind = [i for i in range(len(rhdr))
             if i != rkind and i not in rbind]
    rgetv = rowgetter(*rvind)

    # output header: all left fields, then the remaining right fields, with
    # prefixes applied in the same style as the equi-joins
    if lprefix is None:
        outhdr = list(lhdr)
    else:
        outhdr = [text_type(lprefix) + text_type(f) for f in lhdr]
    if rprefix is None:
        outhdr.extend(rgetv(rhdr))
    else:
        outhdr.extend([text_type(rprefix) + text_type(f)
                       for f in rgetv(rhdr)])
    yield tuple(outhdr)

    cursor = _AsOfRightCursor(rit, rgetby, rgetkey, rgetraw)
    n_rvalues = len(rvind)

    for lrow in lit:
        lgrp = lgetby(lrow)
        lkey = lgetkey(lrow)
        lraw = lgetraw(lrow)

        # advance the single streaming cursor to the left key value
        cursor.align(lgrp, lkey)

        candidate = None
        if direction == 'backward':
            if allow_exact_matches:
                candidate = cursor.exact_or_prev(lgrp, lkey)
            else:
                candidate = cursor.prev(lgrp, lkey)
        elif direction == 'forward':
            if allow_exact_matches:
                candidate = cursor.exact_or_next(lgrp, lkey)
            else:
                candidate = cursor.next_strict(lgrp, lkey)
        else:  # nearest
            back = (cursor.exact_or_prev(lgrp, lkey)
                    if allow_exact_matches
                    else cursor.prev(lgrp, lkey))
            fwd = (cursor.exact_or_next(lgrp, lkey)
                   if allow_exact_matches
                   else cursor.next_strict(lgrp, lkey))
            if back is None:
                candidate = fwd
            elif fwd is None:
                candidate = back
            else:
                back_dist = lraw - rgetraw(back)
                fwd_dist = rgetraw(fwd) - lraw
                # equidistant ties resolve to the backward side
                if back_dist <= fwd_dist:
                    candidate = back
                else:
                    candidate = fwd

        if candidate is not None and tolerance is not None:
            distance = abs(lraw - rgetraw(candidate))
            if distance > tolerance:
                candidate = None

        outrow = list(lrow)
        if candidate is None:
            outrow.extend([missing] * n_rvalues)
        else:
            outrow.extend(rgetv(candidate))
        yield tuple(outrow)


class _AsOfRightCursor(object):
    """Single-pass, group-aware cursor over a right table sorted ascending
    by (by..., key). Only the strictly-before candidate for the most recent
    position, the run of rows at the current equal key value, and one
    look-ahead row are retained; whole groups and whole tables are never
    materialised."""

    def __init__(self, rit, getby, getkey, getraw):
        self._it = rit
        self._getby = getby
        self._getkey = getkey
        self._getraw = getraw
        self._peek = None          # next un-consumed right row, or None at EOF
        self._eof = False
        self._current_group = None  # left partition the cursor is positioned on
        self._held_group = None
        self._held_key = None
        self._held = None          # last row with key < current position, or
        self._run_group = None
        self._run_key = None
        self._run = None           # stable-ordered rows at the equal key run

    def _consume(self):
        if self._eof:
            return None
        try:
            row = next(self._it)
        except StopIteration:
            self._eof = True
            self._peek = None
            return None
        self._peek = row
        return row

    def align(self, group, key):
        """Position the cursor so that `_held` is the last in-group row with
        key strictly less than `key`, `_run` holds the full in-group run of
        rows with key equal to `key`, and `_peek` is the first row beyond
        (or the first row of a later group / EOF)."""
        # entering a different left partition invalidates every candidate
        # retained from the previous one; the right stream itself is not
        # rewound because both inputs are sorted by (by..., key)
        if self._current_group is None or self._current_group != group:
            self._held_group = self._held_key = self._held = None
            self._run_group = self._run_key = self._run = None
            self._current_group = group
        # settle state left over from an earlier position before consuming
        # any new rows; this also runs when the right table is exhausted
        if self._run is not None and self._run_key < key:
            # the equal-key run is now strictly before the new position:
            # its last row becomes the backward candidate
            self._held_group = self._run_group
            self._held_key = self._run_key
            self._held = self._run[-1]
            self._run = None
        while True:
            if self._peek is None and not self._eof:
                if self._consume() is None:
                    return
            if self._peek is None:
                return
            rgrp = self._getby(self._peek)
            rkey = self._getkey(self._peek)
            if rgrp < group:
                # whole groups strictly before the current left group are
                # discarded without buffering anything
                self._peek = None
                continue
            if rgrp > group:
                return
            # same group
            if rkey < key:
                self._held_group = rgrp
                self._held_key = rkey
                self._held = self._peek
                self._peek = None
                continue
            if rkey == key:
                if (self._run is None or self._run_group != group
                        or self._run_key != rkey):
                    self._run_group = rgrp
                    self._run_key = rkey
                    self._run = []
                self._run.append(self._peek)
                self._peek = None
                continue
            # rkey > key
            return

    def exact_or_prev(self, group, key):
        if self._run is not None and self._run_group == group \
                and self._run_key == key:
            # last row of the equal-key run on the backward side
            return self._run[-1]
        return self.prev(group, key)

    def prev(self, group, key):
        if self._held is not None and self._held_group == group \
                and self._held_key < key:
            return self._held
        return None

    def exact_or_next(self, group, key):
        if self._run is not None and self._run_group == group \
                and self._run_key == key:
            # first row of the equal-key run on the forward side
            return self._run[0]
        return self.next_strict(group, key)

    def next_strict(self, group, key):
        if self._peek is not None and self._getby(self._peek) == group \
                and self._getkey(self._peek) > key:
            return self._peek
        return None
