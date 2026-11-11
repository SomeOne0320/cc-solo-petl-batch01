from __future__ import absolute_import, division, print_function

from datetime import datetime, timedelta

import pytest

import petl as etl
from petl.errors import ArgumentError, FieldSelectionError
from petl.test.helpers import eq_, ieq


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _example_tables():
    left = (('device', 'ts', 'event'),
            ('A', 5, 'x'),
            ('A', 10, 'y'),
            ('B', 7, 'z'))
    right = (('device', 'ts', 'state'),
             ('A', 3, 'old'),
             ('A', 8, 'new'),
             ('B', 9, 'future'))
    return left, right


class CountingTable(etl.Table):
    """Counts the data rows pulled from the underlying rows so laziness and
    bounded look-ahead can be asserted."""

    def __init__(self, rows):
        self.rows = rows
        self.count = 0

    def __iter__(self):
        self.count = 0
        it = iter(self.rows)
        yield next(it)
        for row in it:
            self.count += 1
            yield row


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------

def test_asofjoin_example():
    left, right = _example_tables()
    table = etl.asofjoin(left, right, key='ts', by='device')
    expect = (('device', 'ts', 'event', 'state'),
              ('A', 5, 'x', 'old'),
              ('A', 10, 'y', 'new'),
              ('B', 7, 'z', None))
    ieq(expect, table)
    ieq(expect, table)


def test_asofjoin_fluent_and_export():
    left, right = _example_tables()
    table = etl.wrap(left).asofjoin(right, key='ts', by='device')
    eq_(('device', 'ts', 'event', 'state'), table.header())
    assert etl.asofjoin is etl.Table.asofjoin


def test_asofjoin_directions():
    left, right = _example_tables()

    backward = etl.asofjoin(left, right, key='ts', by='device')
    ieq((('device', 'ts', 'event', 'state'),
         ('A', 5, 'x', 'old'),
         ('A', 10, 'y', 'new'),
         ('B', 7, 'z', None)),
        backward)

    forward = etl.asofjoin(left, right, key='ts', by='device',
                           direction='forward')
    ieq((('device', 'ts', 'event', 'state'),
         ('A', 5, 'x', 'new'),
         ('A', 10, 'y', None),
         ('B', 7, 'z', 'future')),
        forward)

    nearest = etl.asofjoin(left, right, key='ts', by='device',
                           direction='nearest')
    ieq((('device', 'ts', 'event', 'state'),
         ('A', 5, 'x', 'old'),
         ('A', 10, 'y', 'new'),
         ('B', 7, 'z', 'future')),
        nearest)


def test_asofjoin_nearest_equidistant_prefers_backward():
    left = (('d', 'ts', 'e'), ('A', 5, 'x'))
    right = (('d', 'ts', 's'), ('A', 3, 'back'), ('A', 7, 'fwd'))
    table = etl.asofjoin(left, right, key='ts', by='d',
                         direction='nearest')
    ieq((('d', 'ts', 'e', 's'), ('A', 5, 'x', 'back')), table)


def test_asofjoin_duplicate_right_keys_backward_takes_last():
    left = (('d', 'ts', 'e'),
            ('A', 3, 'a'), ('A', 5, 'b'), ('A', 8, 'c'), ('A', 9, 'd'))
    right = (('d', 'ts', 's'),
             ('A', 3, 'first3'), ('A', 3, 'last3'),
             ('A', 8, 'first8'), ('A', 8, 'last8'))
    table = etl.asofjoin(left, right, key='ts', by='d')
    ieq((('d', 'ts', 'e', 's'),
         ('A', 3, 'a', 'last3'),
         ('A', 5, 'b', 'last3'),
         ('A', 8, 'c', 'last8'),
         ('A', 9, 'd', 'last8')),
        table)


def test_asofjoin_duplicate_right_keys_forward_takes_first():
    left = (('d', 'ts', 'e'),
            ('A', 3, 'a'), ('A', 5, 'b'), ('A', 8, 'c'), ('A', 9, 'd'))
    right = (('d', 'ts', 's'),
             ('A', 3, 'first3'), ('A', 3, 'last3'),
             ('A', 8, 'first8'), ('A', 8, 'last8'))
    table = etl.asofjoin(left, right, key='ts', by='d',
                         direction='forward')
    ieq((('d', 'ts', 'e', 's'),
         ('A', 3, 'a', 'first3'),
         ('A', 5, 'b', 'first8'),
         ('A', 8, 'c', 'first8'),
         ('A', 9, 'd', None)),
        table)


def test_asofjoin_duplicate_right_keys_nearest_exact_is_backward():
    left = (('d', 'ts', 'e'), ('A', 3, 'a'), ('A', 5, 'b'))
    right = (('d', 'ts', 's'), ('A', 3, 'first3'), ('A', 3, 'last3'))
    table = etl.asofjoin(left, right, key='ts', by='d',
                         direction='nearest')
    ieq((('d', 'ts', 'e', 's'),
         ('A', 3, 'a', 'last3'),
         ('A', 5, 'b', 'last3')),
        table)


def test_asofjoin_exclude_exact_matches_keeps_looking():
    left = (('d', 'ts', 'e'),
            ('A', 3, 'a'), ('A', 5, 'b'), ('A', 8, 'c'))
    right = (('d', 'ts', 's'),
             ('A', 3, 'x'), ('A', 3, 'y'),
             ('A', 8, 'p'), ('A', 8, 'q'))

    backward = etl.asofjoin(left, right, key='ts', by='d',
                            allow_exact_matches=False)
    ieq((('d', 'ts', 'e', 's'),
         ('A', 3, 'a', None),
         ('A', 5, 'b', 'y'),
         ('A', 8, 'c', 'y')),
        backward)

    forward = etl.asofjoin(left, right, key='ts', by='d',
                           direction='forward',
                           allow_exact_matches=False)
    ieq((('d', 'ts', 'e', 's'),
         ('A', 3, 'a', 'p'),
         ('A', 5, 'b', 'p'),
         ('A', 8, 'c', None)),
        forward)

    nearest = etl.asofjoin(left, right, key='ts', by='d',
                           direction='nearest',
                           allow_exact_matches=False)
    # at ts 5 both sides are distance 3; tie resolves backward
    ieq((('d', 'ts', 'e', 's'),
         ('A', 3, 'a', 'p'),
         ('A', 5, 'b', 'y'),
         ('A', 8, 'c', 'y')),
        nearest)


def test_asofjoin_tolerance_boundary_and_zero():
    left = (('d', 'ts', 'e'), ('A', 4, 'a'), ('A', 10, 'b'))
    right = (('d', 'ts', 's'), ('A', 3, 'x'), ('A', 8, 'y'))

    # boundary is inclusive: distance 1 accepted
    table = etl.asofjoin(left, right, key='ts', by='d', tolerance=1)
    ieq((('d', 'ts', 'e', 's'),
         ('A', 4, 'a', 'x'),
         ('A', 10, 'b', None)),
        table)

    # zero tolerance only accepts exact matches
    table = etl.asofjoin(left, right, key='ts', by='d', tolerance=0)
    ieq((('d', 'ts', 'e', 's'),
         ('A', 4, 'a', None),
         ('A', 10, 'b', None)),
        table)

    # forward tolerance also bounded
    table = etl.asofjoin(left, right, key='ts', by='d',
                         direction='forward', tolerance=1)
    ieq((('d', 'ts', 'e', 's'),
         ('A', 4, 'a', None),
         ('A', 10, 'b', None)),
        table)


def test_asofjoin_tolerance_does_not_fall_back_to_other_candidate():
    # the selected candidate fails tolerance: no second-choice candidate
    left = (('d', 'ts', 'e'), ('A', 10, 'a'))
    right = (('d', 'ts', 's'), ('A', 1, 'near'), ('A', 8, 'chosen'))
    table = etl.asofjoin(left, right, key='ts', by='d', tolerance=1)
    ieq((('d', 'ts', 'e', 's'), ('A', 10, 'a', None)), table)


def test_asofjoin_compound_group_isolation():
    left = (('g', 'h', 'ts', 'e'),
            ('A', 'x', 1, 'e1'),
            ('A', 'y', 1, 'e2'),
            ('B', 'x', 1, 'e3'))
    right = (('g', 'h', 'ts', 's'),
             ('A', 'x', 0, 's1'),
             ('A', 'y', 0, 's2'),
             ('B', 'x', 2, 's3'),
             ('A', 'x', 2, 's4'))
    table = etl.asofjoin(left, right, key='ts', by=['g', 'h'])
    ieq((('g', 'h', 'ts', 'e', 's'),
         ('A', 'x', 1, 'e1', 's1'),
         ('A', 'y', 1, 'e2', 's2'),
         ('B', 'x', 1, 'e3', None)),
        table)
    # tuple form of the same by selector
    table2 = etl.asofjoin(left, right, key='ts', by=('g', 'h'))
    ieq(table, table2)


def test_asofjoin_no_by_single_partition():
    left = (('ts', 'e'), (1, 'a'), (3, 'b'), (5, 'c'))
    right = (('ts', 's'), (2, 'x'), (4, 'y'))
    table = etl.asofjoin(left, right, key='ts')
    ieq((('ts', 'e', 's'),
         (1, 'a', None),
         (3, 'b', 'x'),
         (5, 'c', 'y')),
        table)


def test_asofjoin_different_column_order_and_integer_selectors():
    # both headers resolve selector key=1 to ts and by=0 to device, but the
    # trailing value columns differ in ordering
    left = (('device', 'ts', 'event'),
            ('A', 5, 'x'), ('A', 10, 'y'), ('B', 7, 'z'))
    right = (('device', 'ts', 'state'),
             ('A', 3, 'old'), ('A', 8, 'new'), ('B', 9, 'future'))
    by_index = etl.asofjoin(left, right, key=1, by=0)
    by_name = etl.asofjoin(left, right, key='ts', by='device')
    ieq(by_name, by_index)

    # fully reordered headers: on both sides the key index 1 resolves to
    # the ts field and the by index 2 to device, while each side's value
    # column occupies its own position
    left2 = (('event', 'ts', 'device'),
             ('x', 5, 'A'), ('y', 10, 'A'), ('z', 7, 'B'))
    right2 = (('state', 'ts', 'device'),
              ('old', 3, 'A'), ('new', 8, 'A'), ('future', 9, 'B'))
    expect2 = (('event', 'ts', 'device', 'state'),
               ('x', 5, 'A', 'old'),
               ('y', 10, 'A', 'new'),
               ('z', 7, 'B', None))
    # unsorted inputs resolve the positional selectors against each header
    table = etl.asofjoin(left2, right2, key=1, by=2)
    ieq(expect2, table)
    # the same data physically ordered by (device, ts) gives identical
    # answers in presorted mode
    left2_pre = (left2[0],) + tuple(
        sorted(left2[1:], key=lambda r: (r[2], r[1])))
    right2_pre = (right2[0],) + tuple(
        sorted(right2[1:], key=lambda r: (r[2], r[1])))
    table_pre = etl.asofjoin(left2_pre, right2_pre, key=1, by=2,
                             presorted=True)
    ieq(expect2, table_pre)
    # name selectors agree with the positional ones
    ieq(expect2, etl.asofjoin(left2, right2, key='ts', by='device'))


def test_asofjoin_no_match_and_empty_tables():
    left, right = _example_tables()

    # right table header only: every left row emitted with missing values
    empty_right = (('device', 'ts', 'state'),)
    table = etl.asofjoin(left, empty_right, key='ts', by='device')
    ieq((('device', 'ts', 'event', 'state'),
         ('A', 5, 'x', None),
         ('A', 10, 'y', None),
         ('B', 7, 'z', None)),
        table)

    # left table header only: output header only
    empty_left = (('device', 'ts', 'event'),)
    table = etl.asofjoin(empty_left, right, key='ts', by='device')
    ieq((('device', 'ts', 'event', 'state'),), table)

    # missing sentinel is respected
    table = etl.asofjoin(left, empty_right, key='ts', by='device',
                         missing='NA')
    ieq((('device', 'ts', 'event', 'state'),
         ('A', 5, 'x', 'NA'),
         ('A', 10, 'y', 'NA'),
         ('B', 7, 'z', 'NA')),
        table)


def test_asofjoin_prefixes_only_affect_header():
    left, right = _example_tables()
    table = etl.asofjoin(left, right, key='ts', by='device',
                         lprefix='L_', rprefix='R_')
    expect = (('L_device', 'L_ts', 'L_event', 'R_state'),
              ('A', 5, 'x', 'old'),
              ('A', 10, 'y', 'new'),
              ('B', 7, 'z', None))
    ieq(expect, table)


def test_asofjoin_uneven_rows_are_squared_up():
    left = (('d', 'ts', 'e', 'note'),
            ('A', 5, 'x'),
            ('A', 10, 'y', 'q', 'toolong'))
    right = (('d', 'ts', 's'),
             ('A', 3, 'old'),
             ('A', 8))
    table = etl.asofjoin(left, right, key='ts', by='d', missing='NA')
    ieq((('d', 'ts', 'e', 'note', 's'),
         ('A', 5, 'x', 'NA', 'old'),
         ('A', 10, 'y', 'q', 'NA')),
        table)


def test_asofjoin_datetime_keys_and_timedelta_tolerance():
    noon = datetime(2020, 1, 1, 12)
    left = (('d', 'ts', 'e'), ('A', noon, 'x'))
    right = (('d', 'ts', 's'),
             ('A', datetime(2020, 1, 1, 10), 'am'),
             ('A', datetime(2020, 1, 1, 14), 'pm'))

    backward = etl.asofjoin(left, right, key='ts', by='d')
    ieq((('d', 'ts', 'e', 's'), (('A', noon, 'x', 'am'))), backward)

    # equidistant nearest (2h each) resolves to the backward side
    nearest = etl.asofjoin(left, right, key='ts', by='d',
                           direction='nearest')
    ieq((('d', 'ts', 'e', 's'), (('A', noon, 'x', 'am'))), nearest)

    # tolerance of 3h accepts the 2h-distant candidate
    within = etl.asofjoin(left, right, key='ts', by='d',
                          direction='nearest',
                          tolerance=timedelta(hours=3))
    ieq((('d', 'ts', 'e', 's'), (('A', noon, 'x', 'am'))), within)

    # tolerance of 1h rejects both candidates
    outside = etl.asofjoin(left, right, key='ts', by='d',
                           direction='nearest',
                           tolerance=timedelta(hours=1))
    ieq((('d', 'ts', 'e', 's'), (('A', noon, 'x', None))), outside)

    # zero timedelta tolerance only accepts exact hits
    left_exact = (('d', 'ts', 'e'),
                  ('A', datetime(2020, 1, 1, 10), 'x'))
    exact = etl.asofjoin(left_exact, right, key='ts', by='d',
                         tolerance=timedelta(0))
    ieq((('d', 'ts', 'e', 's'),
         (('A', datetime(2020, 1, 1, 10), 'x', 'am'))),
        exact)


def test_asofjoin_unsorted_input_is_stably_sorted():
    left = (('device', 'ts', 'event'),
            ('B', 7, 'z'),
            ('A', 10, 'y'),
            ('A', 5, 'x'))
    right = (('device', 'ts', 'state'),
             ('B', 9, 'future'),
             ('A', 8, 'new'),
             ('A', 3, 'old'))
    table = etl.asofjoin(left, right, key='ts', by='device')
    expect = (('device', 'ts', 'event', 'state'),
              ('A', 5, 'x', 'old'),
              ('A', 10, 'y', 'new'),
              ('B', 7, 'z', None))
    ieq(expect, table)

    # stable order within equal (by, key) values is preserved
    left2 = (('d', 'ts', 'e'), ('A', 5, 'second'), ('A', 5, 'first'))
    right2 = (('d', 'ts', 's'), ('A', 5, 'r2'), ('A', 5, 'r1'))
    table2 = etl.asofjoin(left2, right2, key='ts', by='d')
    ieq((('d', 'ts', 'e', 's'),
         ('A', 5, 'second', 'r1'),
         ('A', 5, 'first', 'r1')),
        table2)


def test_asofjoin_reiterable_does_not_reuse_candidate_state():
    left, right = _example_tables()
    for presorted in (False, True):
        table = etl.asofjoin(left, right, key='ts', by='device',
                             presorted=presorted)
        ieq(list(table), table)


# ---------------------------------------------------------------------------
# presorted laziness and large-group history
# ---------------------------------------------------------------------------

def _big_history_tables(n=10000):
    right = [('device', 'ts', 'state')]
    right.extend([('A', i, 's%d' % i) for i in range(n)])
    left = [('device', 'ts', 'event'),
            ('A', 5000, 'e1'), ('A', 9999, 'e2')]
    return left, right


def test_asofjoin_presorted_header_reads_no_data():
    left, right = _big_history_tables()
    lt, rt = CountingTable(left), CountingTable(right)
    table = etl.asofjoin(lt, rt, key='ts', by='device', presorted=True)
    eq_(('device', 'ts', 'event', 'state'), etl.header(table))
    eq_(0, lt.count)
    eq_(0, rt.count)


def test_asofjoin_presorted_bounded_lookahead_large_group():
    left, right = _big_history_tables()
    lt, rt = CountingTable(left), CountingTable(right)
    table = etl.asofjoin(lt, rt, key='ts', by='device', presorted=True)
    it = iter(table)
    next(it)  # header
    first = next(it)
    eq_(('A', 5000, 'e1', 's5000'), first)
    # only the rows up to the first left key are consumed, not the history
    assert rt.count <= 5002
    second = next(it)
    eq_(('A', 9999, 'e2', 's9999'), second)
    eq_(10000, rt.count)


def test_asofjoin_presorted_prefix_does_not_scan_right_tail():
    right = [('device', 'ts', 'state')]
    right.extend([('A', i, 's%d' % i) for i in range(5)])
    # a long tail belonging to a later partition must remain untouched
    right.extend([('Z', i, 'z%d' % i) for i in range(10000)])
    left = [('device', 'ts', 'event'),
            ('A', 0, 'only'), ('A', 1, 'second')]
    lt, rt = CountingTable(left), CountingTable(right)
    table = etl.asofjoin(lt, rt, key='ts', by='device',
                         direction='forward', presorted=True)
    rows = list(table)
    eq_([('device', 'ts', 'event', 'state'),
         ('A', 0, 'only', 's0'),
         ('A', 1, 'second', 's1')],
        rows)
    # only the small A-group look-ahead is consumed; none of the Z tail
    assert rt.count <= 6


def test_asofjoin_presorted_duplicate_run_may_be_buffered_but_not_tail():
    right = [('device', 'ts', 'state')]
    right.extend([('A', 5, 'dup')] * 100)
    right.append(('A', 6, 'tail'))
    left = [('device', 'ts', 'event'), ('A', 5, 'e1')]
    lt, rt = CountingTable(left), CountingTable(right)
    table = etl.asofjoin(lt, rt, key='ts', by='device', presorted=True)
    rows = list(table)
    eq_([('device', 'ts', 'event', 'state'),
         ('A', 5, 'e1', 'dup')], rows)
    # the equal-time run is read, but only one additional look-ahead row is
    # consumed; no grouping beyond that is materialised
    assert rt.count <= 101


def test_asofjoin_presorted_modes_agree_on_large_group():
    left, right = _big_history_tables(n=2000)
    presorted = etl.asofjoin(left, right, key='ts', by='device',
                             presorted=True)
    sorted_join = etl.asofjoin(left, right, key='ts', by='device')
    ieq(sorted_join, presorted)


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_asofjoin_unknown_direction_rejected_at_construction():
    left, right = _example_tables()
    with pytest.raises(ArgumentError):
        etl.asofjoin(left, right, key='ts', by='device',
                     direction='sideways')


def test_asofjoin_non_boolean_allow_exact_rejected_at_construction():
    left, right = _example_tables()
    for bad in ('yes', 1, 0, None):
        with pytest.raises(ArgumentError):
            etl.asofjoin(left, right, key='ts', by='device',
                         allow_exact_matches=bad)


def test_asofjoin_negative_tolerance_rejected_at_construction():
    left, right = _example_tables()
    with pytest.raises(ArgumentError):
        etl.asofjoin(left, right, key='ts', by='device', tolerance=-1)
    with pytest.raises(ArgumentError):
        etl.asofjoin(left, right, key='ts', by='device',
                     tolerance=timedelta(seconds=-1))
    with pytest.raises(ArgumentError):
        etl.asofjoin(left, right, key='ts', by='device', tolerance='1')


def test_asofjoin_compound_key_rejected_at_construction():
    left, right = _example_tables()
    with pytest.raises(ArgumentError):
        etl.asofjoin(left, right, key=['ts', 'device'])
    with pytest.raises(ArgumentError):
        etl.asofjoin(left, right, key=('ts',))
    with pytest.raises(ArgumentError):
        etl.asofjoin(left, right, key=None)


def test_asofjoin_unknown_field_rejected_on_first_iteration():
    left, right = _example_tables()
    view = etl.asofjoin(left, right, key='nope', by='device')
    with pytest.raises(FieldSelectionError):
        list(view)
    view = etl.asofjoin(left, right, key='ts', by='nope')
    with pytest.raises(FieldSelectionError):
        list(view)
    # construction itself never touched the sources
    view = etl.asofjoin(left, right, key='nope', by='device',
                        presorted=True)
    with pytest.raises(FieldSelectionError):
        list(view)


def test_asofjoin_overlapping_key_and_by_rejected():
    left, right = _example_tables()
    view = etl.asofjoin(left, right, key='ts', by=['device', 'ts'],
                        presorted=True)
    with pytest.raises(ArgumentError):
        list(view)
    view = etl.asofjoin(left, right, key=0, by=0, presorted=True)
    with pytest.raises(ArgumentError):
        list(view)


def test_asofjoin_construction_does_not_consume_sources():
    class ExplodingTable(etl.Table):
        def __iter__(self):
            raise RuntimeError('source consumed during construction')

    source = ExplodingTable()
    etl.asofjoin(source, source, key='ts', by='device')
    etl.asofjoin(source, source, key='ts', by='device', presorted=True)
    etl.asofjoin(source, source, key='ts', by='device',
                 direction='forward', tolerance=2)
