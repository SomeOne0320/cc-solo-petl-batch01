from __future__ import absolute_import, print_function, division


import os
import gc
import logging
from datetime import datetime
import platform

import pytest

from petl.compat import next

from petl.errors import FieldSelectionError
from petl.test.helpers import ieq, eq_
from petl.util import nrows
from petl.transform.basics import cat
from petl.transform.sorts import sort, mergesort, issorted


logger = logging.getLogger(__name__)
debug = logger.debug


def test_sort_1():

    table = (('foo', 'bar'),
             ('C', '2'),
             ('A', '9'),
             ('A', '6'),
             ('F', '1'),
             ('D', '10'))

    result = sort(table, 'foo')
    expectation = (('foo', 'bar'),
                   ('A', '9'),
                   ('A', '6'),
                   ('C', '2'),
                   ('D', '10'),
                   ('F', '1'))
    ieq(expectation, result)


def test_sort_2():

    table = (('foo', 'bar'),
             ('C', '2'),
             ('A', '9'),
             ('A', '6'),
             ('F', '1'),
             ('D', '10'))

    result = sort(table, key=('foo', 'bar'))
    expectation = (('foo', 'bar'),
                   ('A', '6'),
                   ('A', '9'),
                   ('C', '2'),
                   ('D', '10'),
                   ('F', '1'))
    ieq(expectation, result)

    result = sort(table)  # default is lexical sort
    expectation = (('foo', 'bar'),
                   ('A', '6'),
                   ('A', '9'),
                   ('C', '2'),
                   ('D', '10'),
                   ('F', '1'))
    ieq(expectation, result)


def test_sort_3():

    table = (('foo', 'bar'),
             ('C', '2'),
             ('A', '9'),
             ('A', '6'),
             ('F', '1'),
             ('D', '10'))

    result = sort(table, 'bar')
    expectation = (('foo', 'bar'),
                   ('F', '1'),
                   ('D', '10'),
                   ('C', '2'),
                   ('A', '6'),
                   ('A', '9'))
    ieq(expectation, result)


def test_sort_4():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', 6),
             ('F', 1),
             ('D', 10))

    result = sort(table, 'bar')
    expectation = (('foo', 'bar'),
                   ('F', 1),
                   ('C', 2),
                   ('A', 6),
                   ('A', 9),
                   ('D', 10))
    ieq(expectation, result)


def test_sort_5():

    table = (('foo', 'bar'),
             (2.3, 2),
             (1.2, 9),
             (2.3, 6),
             (3.2, 1),
             (1.2, 10))

    expectation = (('foo', 'bar'),
                   (1.2, 9),
                   (1.2, 10),
                   (2.3, 2),
                   (2.3, 6),
                   (3.2, 1))

    # can use either field names or indices (from 1) to specify sort key
    result = sort(table, key=('foo', 'bar'))
    ieq(expectation, result)
    result = sort(table, key=(0, 1))
    ieq(expectation, result)
    result = sort(table, key=('foo', 1))
    ieq(expectation, result)
    result = sort(table, key=(0, 'bar'))
    ieq(expectation, result)


def test_sort_6():

    table = (('foo', 'bar'),
             (2.3, 2),
             (1.2, 9),
             (2.3, 6),
             (3.2, 1),
             (1.2, 10))

    expectation = (('foo', 'bar'),
                   (3.2, 1),
                   (2.3, 6),
                   (2.3, 2),
                   (1.2, 10),
                   (1.2, 9))

    result = sort(table, key=('foo', 'bar'), reverse=True)
    ieq(expectation, result)


def test_sort_buffered():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', 6),
             ('F', 1),
             ('D', 10))

    # test sort forwards
    expectation = (('foo', 'bar'),
                   ('F', 1),
                   ('C', 2),
                   ('A', 6),
                   ('A', 9),
                   ('D', 10))
    result = sort(table, 'bar')
    ieq(expectation, result)
    result = sort(table, 'bar', buffersize=2)
    ieq(expectation, result)

    # sort in reverse
    expectation = (('foo', 'bar'),
                   ('D', 10),
                   ('A', 9),
                   ('A', 6),
                   ('C', 2),
                   ('F', 1))

    result = sort(table, 'bar', reverse=True)
    ieq(expectation, result)
    result = sort(table, 'bar', reverse=True, buffersize=2)
    ieq(expectation, result)

    # no key
    expectation = (('foo', 'bar'),
                   ('F', 1),
                   ('D', 10),
                   ('C', 2),
                   ('A', 9),
                   ('A', 6))
    result = sort(table, reverse=True)
    ieq(expectation, result)
    result = sort(table, reverse=True, buffersize=2)
    ieq(expectation, result)


def test_sort_buffered_tempdir():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', 6),
             ('F', 1),
             ('D', 10))

    # test sort forwards
    expectation = (('foo', 'bar'),
                   ('F', 1),
                   ('C', 2),
                   ('A', 6),
                   ('A', 9),
                   ('D', 10))
    result = sort(table, 'bar')
    ieq(expectation, result)
    tempdir = 'tmp'
    if not os.path.exists(tempdir):
        os.mkdir(tempdir)
    result = sort(table, 'bar', buffersize=2, tempdir=tempdir)
    ieq(expectation, result)


def test_sort_buffered_independent():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', 6),
             ('F', 1),
             ('D', 10))
    expectation = (('foo', 'bar'),
                   ('F', 1),
                   ('C', 2),
                   ('A', 6),
                   ('A', 9),
                   ('D', 10))

    result = sort(table, 'bar', buffersize=4)
    nrows(result)  # cause data to be cached
    # check that two row iterators are independent, i.e., consuming rows
    # from one does not affect the other
    it1 = iter(result)
    it2 = iter(result)
    eq_(expectation[0], next(it1))
    eq_(expectation[1], next(it1))
    eq_(expectation[0], next(it2))
    eq_(expectation[1], next(it2))
    eq_(expectation[2], next(it2))
    eq_(expectation[2], next(it1))


def _get_names(l):
    return [x.name for x in l]


def test_sort_buffered_cleanup():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', 6),
             ('F', 1),
             ('D', 10))
    result = sort(table, 'bar', buffersize=2)
    debug('initially filecache should be empty')
    eq_(None, result._filecache)
    debug('pull rows through, should populate file cache')
    eq_(5, nrows(result))
    eq_(3, len(result._filecache))
    debug('check all files exist')
    filenames = _get_names(result._filecache.runs)
    for fn in filenames:
        assert os.path.exists(fn), fn
    debug('delete object and garbage collect')
    del result
    gc.collect()
    debug('check all files have been deleted')
    for fn in filenames:
        assert not os.path.exists(fn), fn


@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='SKIP sort cleanup test (PyPy)')
def test_sort_buffered_cleanup_open_iterator():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', 6),
             ('F', 1),
             ('D', 10))
    # check if cleanup is robust against open iterators
    result = sort(table, 'bar', buffersize=2)
    debug('pull rows through, should populate file cache')
    eq_(5, nrows(result))
    eq_(3, len(result._filecache))
    debug('check all files exist')
    filenames = _get_names(result._filecache.runs)
    for fn in filenames:
        assert os.path.exists(fn), fn
    debug(filenames)
    debug('open an iterator')
    it = iter(result)
    next(it)
    next(it)
    debug('delete objects and garbage collect')
    del result
    del it
    gc.collect()
    for fn in filenames:
        assert not os.path.exists(fn), fn


def test_sort_empty():
    table = (('foo', 'bar'),)
    expect = (('foo', 'bar'),)
    actual = sort(table)
    ieq(expect, actual)


def test_sort_none():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', None),
             ('F', 1),
             ('D', 10))

    result = sort(table, 'bar')
    print(list(result))
    expectation = (('foo', 'bar'),
                   ('A', None),
                   ('F', 1),
                   ('C', 2),
                   ('A', 9),
                   ('D', 10))
    ieq(expectation, result)

    dt = datetime.now().replace

    table = (('foo', 'bar'),
             ('C', dt(hour=5)),
             ('A', dt(hour=1)),
             ('A', None),
             ('F', dt(hour=9)),
             ('D', dt(hour=17)))

    result = sort(table, 'bar')
    expectation = (('foo', 'bar'),
                   ('A', None),
                   ('A', dt(hour=1)),
                   ('C', dt(hour=5)),
                   ('F', dt(hour=9)),
                   ('D', dt(hour=17)))
    ieq(expectation, result)


def test_sort_headerless_no_keys():
    """
    Sorting a headerless table without specifying cols should be a no-op.
    """
    table = []
    result = sort(table)
    expectation = []
    ieq(expectation, result)


def test_sort_headerless_explicit():
    """
    But if you specify keys, they must exist.
    """
    table = []
    with pytest.raises(FieldSelectionError):
        for i in sort(table, 'foo'):
            pass

# TODO test sort with native comparison


def test_mergesort_1():

    table1 = (('foo', 'bar'),
              ('A', 6),
              ('C', 2),
              ('D', 10),
              ('A', 9),
              ('F', 1))

    table2 = (('foo', 'bar'),
              ('B', 3),
              ('D', 10),
              ('A', 10),
              ('F', 4))

    # should be same as concatenate then sort (but more efficient, esp. when
    # presorted)
    expect = sort(cat(table1, table2))

    actual = mergesort(table1, table2)
    ieq(expect, actual)
    ieq(expect, actual)

    actual = mergesort(sort(table1), sort(table2), presorted=True)
    ieq(expect, actual)
    ieq(expect, actual)


def test_mergesort_2():

    table1 = (('foo', 'bar'),
              ('A', 9),
              ('C', 2),
              ('D', 10),
              ('A', 6),
              ('F', 1))

    table2 = (('foo', 'baz'),
              ('B', 3),
              ('D', 10),
              ('A', 10),
              ('F', 4))

    # should be same as concatenate then sort (but more efficient, esp. when
    # presorted)
    expect = sort(cat(table1, table2), key='foo')

    actual = mergesort(table1, table2, key='foo')
    ieq(expect, actual)
    ieq(expect, actual)

    actual = mergesort(sort(table1, key='foo'), sort(table2, key='foo'),
                       key='foo', presorted=True)
    ieq(expect, actual)
    ieq(expect, actual)


def test_mergesort_3():

    table1 = (('foo', 'bar'),
              ('A', 9),
              ('C', 2),
              ('D', 10),
              ('A', 6),
              ('F', 1))

    table2 = (('foo', 'baz'),
              ('B', 3),
              ('D', 10),
              ('A', 10),
              ('F', 4))

    # should be same as concatenate then sort (but more efficient, esp. when
    # presorted)
    expect = sort(cat(table1, table2), key='foo', reverse=True)

    actual = mergesort(table1, table2, key='foo', reverse=True)
    ieq(expect, actual)
    ieq(expect, actual)

    actual = mergesort(sort(table1, key='foo', reverse=True),
                       sort(table2, key='foo', reverse=True),
                       key='foo', reverse=True, presorted=True)
    ieq(expect, actual)
    ieq(expect, actual)


def test_mergesort_4():

    table1 = (('foo', 'bar', 'baz'),
              (1, 'A', True),
              (2, 'B', None),
              (4, 'C', True))
    table2 = (('bar', 'baz', 'quux'),
              ('A', True, 42.0),
              ('B', False, 79.3),
              ('C', False, 12.4))

    expect = sort(cat(table1, table2), key='bar')

    actual = mergesort(table1, table2, key='bar')
    ieq(expect, actual)
    ieq(expect, actual)


def test_mergesort_empty():

    table1 = (('foo', 'bar'),
              ('A', 9),
              ('C', 2),
              ('D', 10),
              ('F', 1))

    table2 = (('foo', 'bar'),)

    expect = table1
    actual = mergesort(table1, table2, key='foo')
    ieq(expect, actual)
    ieq(expect, actual)


def test_mergesort_mixed_types_lexical():
    # no key, mixed int/str values must be compared using the same
    # Comparable rules as sort(), not raise TypeError
    table1 = [['k', 'v'], [1, 'a']]
    table2 = [['k', 'v'], ['2', 'b']]

    expectation = [('k', 'v'), (1, 'a'), ('2', 'b')]
    actual = mergesort(table1, table2)
    ieq(expectation, actual)
    ieq(expectation, actual)

    expectation_reverse = [('k', 'v'), ('2', 'b'), (1, 'a')]
    actual = mergesort(table1, table2, reverse=True)
    ieq(expectation_reverse, actual)
    ieq(expectation_reverse, actual)

    # same result when inputs are presorted
    actual = mergesort(sort(table1), sort(table2), presorted=True)
    ieq(expectation, actual)
    actual = mergesort(sort(table1, reverse=True),
                       sort(table2, reverse=True),
                       presorted=True, reverse=True)
    ieq(expectation_reverse, actual)


def test_mergesort_mixed_types_values():
    # None, numbers, dates, bytes and text are ordered following the same
    # rules as sort(), with or without an explicit key
    dt = datetime
    table1 = [['v'], [None], [10], ['apple'], [b'x'], [dt(2020, 1, 2)]]
    table2 = [['v'], [5], [b'bytes'], ['banana'], [dt(2019, 12, 31)]]

    expectation = [('v',),
                   (None,),
                   (5,),
                   (10,),
                   (dt(2019, 12, 31),),
                   (dt(2020, 1, 2),),
                   (b'bytes',),
                   (b'x',),
                   ('apple',),
                   ('banana',)]
    actual = mergesort(table1, table2)
    ieq(expectation, actual)
    ieq(expectation, actual)
    ieq(sort(cat(table1, table2)), actual)

    actual = mergesort(table1, table2, reverse=True)
    ieq(sort(cat(table1, table2), reverse=True), actual)

    # explicit key on a heterogeneous column
    table3 = [['k', 'v'], [1, 'a'], [None, 'b'], ['x', 'c']]
    table4 = [['k', 'v'], [2, 'd'], ['a', 'e']]
    expect = sort(cat(table3, table4), key='k')
    actual = mergesort(table3, table4, key='k')
    ieq(expect, actual)
    ieq(expect, actual)
    actual = mergesort(table3, table4, key='k', reverse=True)
    ieq(sort(cat(table3, table4), key='k', reverse=True), actual)


def test_mergesort_field_order():
    # tables with different field orders must be sorted according to the
    # output field order, not their own field order
    table1 = [['a', 'b'], [100, 100]]
    table2 = [['b', 'a'], [1, 9], [2, 3]]

    expectation = [('a', 'b'), (3, 2), (9, 1), (100, 100)]
    actual = mergesort(table1, table2)
    ieq(expectation, actual)
    ieq(expectation, actual)

    expectation_reverse = [('a', 'b'), (100, 100), (9, 1), (3, 2)]
    actual = mergesort(table1, table2, reverse=True)
    ieq(expectation_reverse, actual)
    ieq(expectation_reverse, actual)

    # same with an explicit key
    table1 = [['a', 'b'], [1, 100], [4, 4]]
    table2 = [['b', 'a'], [50, 2], [3, 3]]
    expectation = [('a', 'b'), (1, 100), (2, 50), (3, 3), (4, 4)]
    actual = mergesort(table1, table2, key='a')
    ieq(expectation, actual)
    ieq(expectation, actual)
    ieq(sort(cat(table1, table2), key='a'), actual)

    # key given as field index refers to the output fields
    actual = mergesort(table1, table2, key=0)
    ieq(expectation, actual)


def test_mergesort_header_short_rows():
    # explicit header selects output fields; excluded fields do not take
    # part in comparisons; short rows are filled with missing
    table1 = [['unused', 'k'], [1]]
    table2 = [['unused', 'k'], [2, 3]]

    expectation = [('k',), (None,), (3,)]
    actual = mergesort(table1, table2, header=['k'], key='k')
    ieq(expectation, actual)
    ieq(expectation, actual)

    # custom missing value (numbers sort before text)
    expectation = [('k',), (3,), ('NA',)]
    actual = mergesort(table1, table2, header=['k'], key='k', missing='NA')
    ieq(expectation, actual)
    ieq(expectation, actual)

    # long rows are trimmed to the output fields
    table1 = [['a'], [1, 2, 3]]
    table2 = [['a', 'b'], [0, 0]]
    expectation = [('a', 'b'), (0, 0), (1, None)]
    actual = mergesort(table1, table2)
    ieq(expectation, actual)
    ieq(expectation, actual)
    ieq(sort(cat(table1, table2)), actual)


def test_mergesort_compound_key():

    table1 = [['a', 'b', 'v'], [1, 2, 'x'], [1, 1, 'y']]
    table2 = [['b', 'a', 'v'], [2, 1, 'p'], [0, 1, 'q']]

    expectation = (('a', 'b', 'v'),
                   (1, 0, 'q'),
                   (1, 1, 'y'),
                   (1, 2, 'x'),
                   (1, 2, 'p'))

    # can use either field names or indices to specify the sort key
    actual = mergesort(table1, table2, key=['a', 'b'])
    ieq(expectation, actual)
    ieq(expectation, actual)
    actual = mergesort(table1, table2, key=('a', 'b'))
    ieq(expectation, actual)
    actual = mergesort(table1, table2, key=[0, 1])
    ieq(expectation, actual)
    actual = mergesort(table1, table2, key=('a', 1))
    ieq(expectation, actual)

    # reverse
    actual = mergesort(table1, table2, key=['a', 'b'], reverse=True)
    ieq(sort(cat(table1, table2), key=['a', 'b'], reverse=True), actual)


def test_mergesort_stable():
    # rows with equal keys keep their original order within each input
    # table, and ties between tables are broken by table order
    table1 = [['k', 'v'], [1, 'a1'], [2, 'b1'], [1, 'c1']]
    table2 = [['k', 'v'], [1, 'a2'], [3, 'd2']]
    table3 = [['k', 'v'], [2, 'b3'], [1, 'a3']]

    expectation = (('k', 'v'),
                   (1, 'a1'),
                   (1, 'c1'),
                   (1, 'a2'),
                   (1, 'a3'),
                   (2, 'b1'),
                   (2, 'b3'),
                   (3, 'd2'))
    actual = mergesort(table1, table2, table3, key='k')
    ieq(expectation, actual)
    ieq(expectation, actual)
    ieq(sort(cat(table1, table2, table3), key='k'), actual)

    expectation_reverse = (('k', 'v'),
                           (3, 'd2'),
                           (2, 'b1'),
                           (2, 'b3'),
                           (1, 'a1'),
                           (1, 'c1'),
                           (1, 'a2'),
                           (1, 'a3'))
    actual = mergesort(table1, table2, table3, key='k', reverse=True)
    ieq(expectation_reverse, actual)
    ieq(expectation_reverse, actual)

    # presorted inputs are merged stably too
    actual = mergesort(sort(table1, key='k'), sort(table2, key='k'),
                       sort(table3, key='k'), key='k', presorted=True)
    ieq(expectation, actual)


def test_mergesort_buffered():

    table1 = (('foo', 'bar'),
              ('A', 9),
              ('C', 2),
              ('D', 10),
              ('A', 6),
              ('F', 1))
    table2 = (('foo', 'baz'),
              ('B', 3),
              ('D', 10),
              ('A', 10),
              ('F', 4))

    # force the sort of each input table to spill to disk
    for buffersize in (1, 2, 3):
        expect = sort(cat(table1, table2), key='foo')
        actual = mergesort(table1, table2, key='foo', buffersize=buffersize)
        ieq(expect, actual)
        ieq(expect, actual)

        expect = sort(cat(table1, table2), key='foo', reverse=True)
        actual = mergesort(table1, table2, key='foo', reverse=True,
                           buffersize=buffersize)
        ieq(expect, actual)
        ieq(expect, actual)

        # default lexical sort
        expect = sort(cat(table1, table2))
        actual = mergesort(table1, table2, buffersize=buffersize)
        ieq(expect, actual)
        ieq(expect, actual)


def test_mergesort_buffered_tempdir(tmp_path):

    table1 = (('foo', 'bar'),
              ('A', 9),
              ('C', 2),
              ('D', 10),
              ('A', 6),
              ('F', 1))
    table2 = (('foo', 'baz'),
              ('B', 3),
              ('D', 10),
              ('A', 10),
              ('F', 4))

    tempdir = str(tmp_path)
    expect = sort(cat(table1, table2), key='foo')
    actual = mergesort(table1, table2, key='foo', buffersize=2,
                       tempdir=tempdir)
    ieq(expect, actual)
    ieq(expect, actual)


def test_mergesort_buffered_cleanup(tmp_path):

    table1 = (('foo', 'bar'),
              ('A', 9),
              ('C', 2),
              ('D', 10),
              ('A', 6),
              ('F', 1))
    table2 = (('foo', 'baz'),
              ('B', 3),
              ('D', 10),
              ('A', 10),
              ('F', 4))

    result = mergesort(table1, table2, key='foo', buffersize=2,
                       tempdir=str(tmp_path))
    debug('pull rows through, should populate file caches')
    eq_(9, nrows(result))
    debug('check each input table was sorted in chunks on disk')
    filenames = list()
    # 5 data rows in table1 and 4 in table2, in chunks of at most 2 rows
    for sortview, nchunks in zip(result.sortviews, (3, 2)):
        eq_(nchunks, len(sortview._filecache))
        filenames.extend(_get_names(sortview._filecache.runs))
    for fn in filenames:
        assert os.path.exists(fn), fn
    debug('cached results are repeatable')
    eq_(9, nrows(result))
    debug('delete object and garbage collect')
    del sortview
    del result
    gc.collect()
    debug('check all files have been deleted')
    for fn in filenames:
        assert not os.path.exists(fn), fn


@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='SKIP sort cleanup test (PyPy)')
def test_mergesort_buffered_cleanup_closed_iterator(tmp_path):

    table1 = (('foo', 'bar'),
              ('A', 9),
              ('C', 2),
              ('D', 10),
              ('A', 6),
              ('F', 1))
    table2 = (('foo', 'baz'),
              ('B', 3),
              ('D', 10),
              ('A', 10),
              ('F', 4))

    # closing an iterator early must not leave temporary files behind
    result = mergesort(table1, table2, key='foo', buffersize=2,
                       tempdir=str(tmp_path), cache=False)
    it = iter(result)
    next(it)
    next(it)
    it.close()
    del it
    del result
    gc.collect()
    eq_([], os.listdir(str(tmp_path)))


def test_mergesort_cache():

    table1 = [['k', 'v'], [2, 'b'], [1, 'a']]
    table2 = [['k', 'v'], [3, 'c']]

    expectation = (('k', 'v'),
                   (1, 'a'),
                   (2, 'b'),
                   (3, 'c'))

    # by default results are cached and iteration is repeatable
    actual = mergesort(table1, table2, key='k')
    ieq(expectation, actual)
    ieq(expectation, actual)

    # cached results do not reflect later changes to the sources
    table1.append([0, 'z'])
    ieq(expectation, actual)

    # with cache=False, later passes reflect changes to the sources
    expectation_changed = (('k', 'v'),
                           (0, 'z'),
                           (1, 'a'),
                           (2, 'b'),
                           (3, 'c'))
    actual = mergesort(table1, table2, key='k', cache=False)
    ieq(expectation_changed, actual)
    table2.append([-1, 'y'])
    expectation_changed2 = (('k', 'v'),
                            (-1, 'y'),
                            (0, 'z'),
                            (1, 'a'),
                            (2, 'b'),
                            (3, 'c'))
    ieq(expectation_changed2, actual)


def test_mergesort_presorted_streaming():
    # presorted inputs are merged in a single streaming pass, without
    # being reordered, re-scanned or fully materialised

    consumed = list()

    def generator(rows, tag):
        yield ['k', 'v']
        for row in rows:
            consumed.append(tag)
            yield row

    gen1 = generator([[1, 'a'], [3, 'c']], 'gen1')
    gen2 = generator([[2, 'b'], [4, 'd']], 'gen2')

    result = mergesort(gen1, gen2, key='k', presorted=True)
    it = iter(result)
    eq_(('k', 'v'), next(it))
    # only the headers have been read so far
    eq_([], consumed)
    eq_((1, 'a'), next(it))
    # at most one row per input has been pulled beyond what was yielded
    eq_(2, len(consumed))
    eq_([(2, 'b'), (3, 'c'), (4, 'd')], list(it))

    # one-shot inputs are consumed in a single pass
    result = mergesort(generator([[1, 'a'], [3, 'c']], 'gen1'),
                       generator([[2, 'b'], [4, 'd']], 'gen2'),
                       key='k', presorted=True)
    ieq((('k', 'v'), (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')), result)


def test_mergesort_empty_and_short_rows():

    # header-only tables
    actual = mergesort([['a', 'b']], [['a', 'b'], [1, 2]])
    ieq((('a', 'b'), (1, 2)), actual)
    ieq((('a', 'b'), (1, 2)), actual)

    # all tables header-only
    actual = mergesort([['a', 'b']], [['a', 'b']])
    ieq((('a', 'b'),), actual)

    # completely empty tables (not even a header row)
    actual = mergesort([], [['a', 'b'], [1, 2]])
    ieq((('a', 'b'), (1, 2)), actual)
    actual = mergesort([], [])
    ieq([()], actual)

    # short rows are filled with missing
    table1 = [['a', 'b'], [1]]
    table2 = [['a', 'b'], [2, 3]]
    actual = mergesort(table1, table2)
    ieq((('a', 'b'), (1, None), (2, 3)), actual)
    ieq(sort(cat(table1, table2)), actual)


def test_mergesort_equivalent_to_cat_then_sort():

    table1 = (('foo', 'bar'),
              ('A', 9),
              ('C', 2),
              ('D', 10),
              ('A', 6),
              ('F', 1))
    table2 = (('foo', 'baz'),
              ('B', 3),
              ('D', 10),
              ('A', 10),
              ('F', 4))

    for key in (None, 'foo', 'bar', ('foo', 'bar'), ('foo', 'baz'), 0, (0, 1)):
        for reverse in (False, True):
            expect = sort(cat(table1, table2), key=key, reverse=reverse)
            # unsorted inputs, in memory and on disk
            actual = mergesort(table1, table2, key=key, reverse=reverse)
            ieq(expect, actual)
            ieq(expect, actual)
            actual = mergesort(table1, table2, key=key, reverse=reverse,
                               buffersize=2)
            ieq(expect, actual)
            # presorted inputs (only keys present in both tables)
            if key in (None, 'foo', 0):
                actual = mergesort(sort(table1, key=key, reverse=reverse),
                                   sort(table2, key=key, reverse=reverse),
                                   key=key, reverse=reverse, presorted=True)
                ieq(expect, actual)
                ieq(expect, actual)


def test_mergesort_bad_key():

    table1 = (('foo', 'bar'),
              ('A', 9))
    table2 = (('foo', 'baz'),
              ('B', 3))

    with pytest.raises(FieldSelectionError):
        for i in mergesort(table1, table2, key='quux'):
            pass


def test_issorted():

    table1 = (('foo', 'bar', 'baz'),
              ('a', 1, True),
              ('b', 3, True),
              ('b', 2))
    assert issorted(table1, key='foo')
    assert not issorted(table1, key='foo', reverse=True)
    assert not issorted(table1, key='foo', strict=True)

    table2 = (('foo', 'bar', 'baz'),
              ('b', 2, True),
              ('a', 1, True),
              ('b', 3))
    assert not issorted(table2, key='foo')

    table3 = (('foo', 'bar', 'baz'),
              ('a', 1, True),
              ('b', 2, True),
              ('b', 3))
    assert issorted(table3, key=('foo', 'bar'))
    assert issorted(table3)

    table4 = (('foo', 'bar', 'baz'),
              ('a', 1, True),
              ('b', 3, True),
              ('b', 2))
    assert not issorted(table4, key=('foo', 'bar'))
    assert not issorted(table4)

    table5 = (('foo', 'bar', 'baz'),
              ('b', 3, True),
              ('b', 2),
              ('a', 1, True))
    assert not issorted(table5, key='foo')
    assert issorted(table5, key='foo', reverse=True)
    assert not issorted(table5, key='foo', reverse=True, strict=True)


def test_sort_missing_cell_numeric():
    """ Sorting table with missing values raises IndexError #385 """
    tbl = (('a', 'b'), ('4',), ('2', '1'), ('1',))
    expect = (('a', 'b'), ('1',), ('2', '1'), ('4',))

    tbl_sorted = sort(tbl)
    ieq(expect, tbl_sorted)


def test_sort_missing_cell_text():
    """ Sorting table with missing values raises IndexError #385 """
    tbl = (('a', 'b', 'c'), ('C',), ('A', '4', '5'))
    expect = (('a', 'b', 'c'), ('A', '4', '5'), ('C',))

    tbl_sorted = sort(tbl)
    ieq(expect, tbl_sorted)
