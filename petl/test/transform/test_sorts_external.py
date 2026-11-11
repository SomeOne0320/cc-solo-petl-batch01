# -*- coding: utf-8 -*-
"""
Tests for bounded fan-in external sorting.

Covers:

* low RLIMIT_NOFILE operation in a real subprocess with far more initial
  runs than F**2 (the original OSError(EMFILE) scenario);
* peak open run-file handles for builds and cached readers, plus merge
  pass / temporary file reclamation accounting;
* stable ordering of equal keys across multiple merge rounds, forward and
  reverse, with compound keys and incomparable payloads;
* the shared ``petl.config.sort_merge_fan_in`` default and validation of
  explicit/configured fan-in values;
* upper-level transforms (join, aggregate) that use sort() without an
  explicit fan-in argument;
* cache=True/False semantics, repeat iteration, two independent cached
  readers, clearcache() generations and early close();
* fault injection at the source iteration, serialisation and merge read
  stages, including recovery of the same view after failure.
"""
from __future__ import absolute_import, print_function, division


import os
import sys
import json
import subprocess
import textwrap

import pytest

from petl.compat import pickle
from petl.test.helpers import ieq, eq_
from petl.util import nrows
import petl.config as config
import petl.transform.sorts as sorts
from petl.transform.sorts import sort, mergesort, SortView, MergeSortView
from petl.transform.joins import join
from petl.transform.reductions import aggregate


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _table(nrows_, nkeys=7):
    return [('k', 'id')] + [(i % nkeys, i) for i in range(nrows_)]


def _expected(nrows_, nkeys=7):
    return [('k', 'id')] + sorted(
        ((i % nkeys, i) for i in range(nrows_)), key=lambda r: (r[0], r[1]))


def _tmp_names(view):
    return [r.name for r in view._filecache.runs]


def _merge_passes(nruns, fan_in):
    passes = 0
    while nruns > fan_in:
        nruns = (nruns + fan_in - 1) // fan_in
        passes += 1
    return passes


def _count_open_run_files(prefix):
    # count real OS file descriptors of this process that point at regular
    # files inside the given run-file directory
    fd_dir = '/proc/self/fd'
    count = 0
    for name in os.listdir(fd_dir):
        try:
            target = os.readlink(os.path.join(fd_dir, name))
        except OSError:
            continue
        if target.startswith(prefix) and os.path.isfile(target):
            count += 1
    return count


# ---------------------------------------------------------------------------
# parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('bad', [1, 0, -1, 2.0, '3', True, False])
def test_merge_fan_in_explicit_validation(bad):
    table = _table(4)
    with pytest.raises(ValueError):
        SortView(table, merge_fan_in=bad)
    with pytest.raises(ValueError):
        MergeSortView([table], merge_fan_in=bad)
    with pytest.raises(ValueError):
        sort(table, merge_fan_in=bad)
    with pytest.raises(ValueError):
        mergesort(table, merge_fan_in=bad)


def test_merge_fan_in_config_validation(tmp_path):
    # an invalid configured default must be refused before any temporary
    # file is created in the tempdir
    table = _table(4)
    old = config.sort_merge_fan_in
    for bad in (None, 1, 2.5, True, '3'):
        config.sort_merge_fan_in = bad
        try:
            with pytest.raises(ValueError):
                sort(table, buffersize=1, tempdir=str(tmp_path))
            eq_([], os.listdir(str(tmp_path)))
        finally:
            config.sort_merge_fan_in = old


def test_merge_fan_in_default_is_32():
    eq_(32, config.sort_merge_fan_in)
    view = sort(_table(4))
    eq_(32, view.merge_fan_in)


def test_merge_fan_in_forwarded_by_mergesort():
    table = _table(4)
    view = mergesort(table, merge_fan_in=5)
    eq_(5, view.merge_fan_in)
    eq_([5], [sv.merge_fan_in for sv in view.sortviews])


# ---------------------------------------------------------------------------
# accounting: fan-in bound, merge rounds, reclamation
# ---------------------------------------------------------------------------


def test_fan_in_2_resource_accounting():
    # 40 initial runs with F=2 -> 40 -> 20 -> 10 -> 5 -> 3 -> 2:
    # 5 merge passes (the last 2 runs are merged while streaming); at most
    # F inputs + 1 output open at any time
    view = sort(_table(40), key='k', buffersize=1, cache=False,
                merge_fan_in=2)
    it = iter(view)
    next(it)  # header
    rows = list(it)
    ieq(_expected(40), [('k', 'id')] + rows)
    stats = view._last_sort_stats
    eq_(40, stats['initial_runs'])
    eq_(_merge_passes(40, 2), stats['merge_passes'])
    eq_(5, stats['merge_passes'])
    assert stats['peak_open_run_files'] <= 3, stats


def test_fan_in_3_resource_accounting():
    # 60 initial runs with F=3 -> 60 -> 20 -> 7 -> 3: 3 passes
    view = sort(_table(60), key='k', buffersize=1, cache=False,
                merge_fan_in=3)
    nrows(view)
    stats = view._last_sort_stats
    eq_(60, stats['initial_runs'])
    eq_(3, stats['merge_passes'])
    assert stats['peak_open_run_files'] <= 4, stats
    # no cache is retained in cache=False mode
    assert view._filecache is None


@pytest.mark.parametrize('fan_in', [2, 3])
def test_intermediate_runs_reclaimed_during_build(tmp_path, fan_in):
    view = sort(_table(80), key='k', buffersize=1, cache=True,
                merge_fan_in=fan_in, tempdir=str(tmp_path))
    nrows(view)
    # intermediate runs have been deleted during the build; only the final
    # (at most fan_in) cached run files remain on disk
    names = _tmp_names(view)
    assert 1 <= len(names) <= fan_in
    for name in names:
        assert os.path.exists(name)
    # nothing extra lingering in the tempdir: every remaining file is a
    # final cached run
    leftovers = [n for n in os.listdir(str(tmp_path))
                 if os.path.isfile(os.path.join(str(tmp_path), n))]
    eq_(set(os.path.basename(n) for n in names), set(leftovers))


def test_cached_reader_peak_handles(tmp_path):
    # a cached view reader opens at most F run input files
    fan_in = 3
    view = sort(_table(60), key='k', buffersize=1, cache=True,
                merge_fan_in=fan_in, tempdir=str(tmp_path))
    nrows(view)
    assert len(view._filecache) == fan_in

    it = iter(view)
    next(it)
    try:
        peak = _count_open_run_files(str(tmp_path))
        for _ in range(10):
            next(it)
        peak = max(peak, _count_open_run_files(str(tmp_path)))
        assert peak <= fan_in, peak
    finally:
        it.close()
    eq_(0, _count_open_run_files(str(tmp_path)))


def test_default_fan_in_resource_accounting():
    # with the default F=32, 160 runs need a single pass (160 -> 5) and
    # never more than F+1 open run files
    view = sort(_table(160), key='k', buffersize=1, cache=False)
    nrows(view)
    stats = view._last_sort_stats
    eq_(160, stats['initial_runs'])
    eq_(1, stats['merge_passes'])
    assert stats['peak_open_run_files'] <= config.sort_merge_fan_in + 1


def test_many_runs_default_fan_in_multipass():
    # 400 runs at F=32 -> 400 -> 13: one pass keeps > F runs? no: 13 <= 32
    view = sort(_table(400), key='k', buffersize=1, cache=False)
    nrows(view)
    stats = view._last_sort_stats
    eq_(1, stats['merge_passes'])
    assert stats['peak_open_run_files'] <= 33
    # push past F**2 to force multiple passes
    view = sort(_table(1100), key='k', buffersize=1, cache=False)
    nrows(view)
    stats = view._last_sort_stats
    eq_(_merge_passes(1100, 32), stats['merge_passes'])
    eq_(2, stats['merge_passes'])
    assert stats['peak_open_run_files'] <= 33


# ---------------------------------------------------------------------------
# correctness: stability, reverse, compound keys, incomparable payloads
# ---------------------------------------------------------------------------


def _stable_table(nrows_):
    # equal keys recur; the payload distinguishes original input order
    return [('k', 'payload')] + [(i % 5, 'p%04d' % i) for i in range(nrows_)]


def _stable_expected(nrows_, reverse):
    data = [(i % 5, 'p%04d' % i) for i in range(nrows_)]
    return [('k', 'payload')] + sorted(
        data, key=lambda r: r[0], reverse=reverse)


@pytest.mark.parametrize('fan_in', [2, 3])
@pytest.mark.parametrize('reverse', [False, True])
def test_stability_across_merge_rounds(fan_in, reverse):
    nrows_ = 120  # > fan_in**2 for both F=2 and F=3, forcing several rounds
    view = sort(_stable_table(nrows_), key='k', buffersize=3,
                reverse=reverse, cache=False, merge_fan_in=fan_in)
    ieq(_stable_expected(nrows_, reverse), view)
    # repeatable
    ieq(_stable_expected(nrows_, reverse), view)


def test_compound_key_multipass():
    table = [('a', 'b', 'id')]
    for i in range(90):
        table.append((i % 4, i % 6, i))
    expected = [('a', 'b', 'id')] + sorted(table[1:], key=lambda r: (r[0],
                                                                      r[1]))
    view = sort(table, key=('a', 'b'), buffersize=2, merge_fan_in=2,
                cache=False)
    ieq(expected, view)


class _Weird(object):
    # payload type whose values are deliberately incomparable; defined at
    # module level so instances survive pickling into run files
    def __init__(self, tag):
        self.tag = tag

    def __lt__(self, other):
        raise TypeError('payloads are not comparable')


class _ReiterableTable(object):
    """Minimal re-iterable table built from a row factory, for cache=False
    tests that need to read the source more than once."""

    def __init__(self, header, factory):
        self.header = header
        self.factory = factory

    def __iter__(self):
        yield tuple(self.header)
        for row in self.factory():
            yield row


def test_incomparable_payload_not_used_for_tie_break():
    # payloads of equal-key rows are mutually incomparable; the merge must
    # decide on keys only and preserve input order
    table = [('k', 'v')]
    for i in range(60):
        table.append((i % 4, _Weird(i)))
    view = sort(table, key='k', buffersize=2, merge_fan_in=2, cache=False)
    it = iter(view)
    hdr = next(it)
    eq_(('k', 'v'), hdr)
    ids = [row[1].tag for row in it]
    # stable order of the original ids within each key group
    grouped = {}
    for i in range(60):
        grouped.setdefault(i % 4, []).append(i)
    flat = []
    for k in sorted(grouped):
        flat.extend(grouped[k])
    eq_(flat, ids)


def test_lexical_key_none_multipass():
    table = [('k', 'id')] + [(i % 7, i) for i in range(80)]
    expected = [('k', 'id')] + sorted(table[1:])
    view = sort(table, buffersize=2, merge_fan_in=2, cache=False)
    ieq(expected, view)


# ---------------------------------------------------------------------------
# cache semantics
# ---------------------------------------------------------------------------


def test_cache_true_builds_once():
    # a re-iterable source wrapped so re-reading can be observed
    reads = [0]

    def source():
        yield ('k', 'id')
        for i in range(40):
            reads[0] += 1
            yield (i % 5, i)

    view = sort(source(), key='k', buffersize=2, cache=True,
                merge_fan_in=2)
    ieq(_expected(40, 5), view)
    eq_(40, reads[0])
    ieq(_expected(40, 5), view)
    ieq(_expected(40, 5), view)
    # source was not re-read and no rebuild happened
    eq_(40, reads[0])


def test_cache_false_rereads_source():
    state = {'n': 40}

    def rows():
        for i in range(state['n']):
            yield (i % 5, i)

    # cache=False re-reads a repeatable source on every iteration
    view = sort(_ReiterableTable(('k', 'id'), rows), key='k',
                buffersize=2, cache=False, merge_fan_in=2)
    ieq(_expected(40, 5), view)
    state['n'] = 41
    ieq(_expected(41, 5), view)
    assert view._filecache is None and view._memcache is None


def test_two_independent_cached_readers_interleave(tmp_path):
    view = sort(_table(60), key='k', buffersize=1, cache=True,
                merge_fan_in=3, tempdir=str(tmp_path))
    nrows(view)
    names = _tmp_names(view)

    it1 = iter(view)
    it2 = iter(view)
    out1 = [next(it1)]
    out2 = []
    for _ in range(15):
        out2.append(next(it2))
    for _ in range(10):
        out1.append(next(it1))
    out1.extend(it1)
    out2.extend(it2)
    expected = _expected(60)
    eq_(expected, out1)
    eq_(expected, out2)
    # cached files survive both readers and remain owned by the view
    for name in names:
        assert os.path.exists(name)


def test_clearcache_generation_lifecycle(tmp_path):
    view = sort(_table(60), key='k', buffersize=1, cache=True,
                merge_fan_in=3, tempdir=str(tmp_path))
    nrows(view)
    old_names = _tmp_names(view)
    old_generation = view._filecache

    # an iterator already reading the old generation keeps it alive
    it_old = iter(view)
    eq_(('k', 'id'), next(it_old))
    # owner + active reader
    assert old_generation._refs >= 2

    view.clearcache()
    assert view._filecache is None
    # old files still readable while the old reader is active
    for name in old_names:
        assert os.path.exists(name)

    # a new reader builds a fresh generation from the current source
    nrows(view)
    new_names = _tmp_names(view)
    assert set(new_names) != set(old_names)
    for name in new_names:
        assert os.path.exists(name)

    # finishing the old reader reaps the old generation only
    rest = list(it_old)
    eq_(60, len(rest))
    for name in old_names:
        assert not os.path.exists(name)
    for name in new_names:
        assert os.path.exists(name)


def test_early_close_cache_false(tmp_path):
    view = sort(_table(80), key='k', buffersize=1, cache=False,
                merge_fan_in=2, tempdir=str(tmp_path))
    it = iter(view)
    next(it)
    next(it)
    it.close()
    del it
    # deterministic cleanup without gc.collect(): all build files gone
    eq_([], os.listdir(str(tmp_path)))


def test_header_only_does_not_read_source_rows(tmp_path):
    consumed = []

    def source():
        yield ('k', 'id')
        for i in range(20):
            consumed.append(i)
            yield (i, i)

    view = sort(source(), key='k', buffersize=1, merge_fan_in=2,
                tempdir=str(tmp_path))
    it = iter(view)
    eq_(('k', 'id'), next(it))
    it.close()
    eq_([], consumed)
    eq_([], os.listdir(str(tmp_path)))


# ---------------------------------------------------------------------------
# fault injection and recovery
# ---------------------------------------------------------------------------


class _SourceFailure(Exception):
    pass


class _PickleFailure(Exception):
    pass


class _MergeReadFailure(Exception):
    pass


def _failing_source(fail_after):
    def gen():
        yield ('k', 'id')
        for i in range(100):
            if i == fail_after:
                raise _SourceFailure('boom at row %d' % i)
            yield (i % 5, i)
    return gen()


def test_failure_source_iteration_cleans_up(tmp_path):
    view = sort(_failing_source(37), key='k', buffersize=2, cache=True,
                merge_fan_in=2, tempdir=str(tmp_path))
    it = iter(view)
    next(it)
    with pytest.raises(_SourceFailure):
        list(it)
    eq_([], os.listdir(str(tmp_path)))
    assert view._filecache is None and view._memcache is None


def test_failure_pickle_dump_cleans_up(tmp_path, monkeypatch):
    real_dump = pickle.dump
    calls = [0]

    def failing_dump(obj, f, *a, **k):
        calls[0] += 1
        if calls[0] == 25:
            raise _PickleFailure('serialiser boom')
        return real_dump(obj, f, *a, **k)

    monkeypatch.setattr(sorts.pickle, 'dump', failing_dump)
    view = sort(_table(80), key='k', buffersize=1, cache=True,
                merge_fan_in=2, tempdir=str(tmp_path))
    with pytest.raises(_PickleFailure):
        list(iter(view))
    eq_([], os.listdir(str(tmp_path)))
    assert view._filecache is None


def test_failure_merge_read_cleans_up_and_preserves_error(tmp_path,
                                                          monkeypatch):
    real_load = pickle.load
    calls = [0]

    def failing_load(f, *a, **k):
        calls[0] += 1
        # fail during a multi-pass merge read, well after chunk writing
        if calls[0] == 70:
            raise _MergeReadFailure('merge read boom')
        return real_load(f, *a, **k)

    monkeypatch.setattr(sorts.pickle, 'load', failing_load)
    view = sort(_table(80), key='k', buffersize=1, cache=False,
                merge_fan_in=2, tempdir=str(tmp_path))
    it = iter(view)
    next(it)
    with pytest.raises(_MergeReadFailure):
        list(it)
    eq_([], os.listdir(str(tmp_path)))


def test_recovery_after_failure_same_view(tmp_path):
    # fail once, remove the fault, then re-iterate the same view successfully
    state = {'fail': True}

    def rows():
        for i in range(50):
            if state['fail'] and i == 25:
                raise _SourceFailure('transient')
            yield (i % 5, i)

    view = sort(_ReiterableTable(('k', 'id'), rows), key='k',
                buffersize=2, cache=True, merge_fan_in=2,
                tempdir=str(tmp_path))
    with pytest.raises(_SourceFailure):
        list(iter(view))
    eq_([], os.listdir(str(tmp_path)))

    state['fail'] = False
    ieq(_expected(50, 5), view)
    # successful build is now cached and repeatable
    ieq(_expected(50, 5), view)


def test_failed_build_does_not_publish_half_cache(tmp_path, monkeypatch):
    real_dump = pickle.dump

    def failing_dump(obj, f, *a, **k):
        if getattr(f, '_fail_here', False):
            raise _PickleFailure('boom')
        return real_dump(obj, f, *a, **k)

    monkeypatch.setattr(sorts.pickle, 'dump', failing_dump)

    state = {'armed': True}

    def rows():
        for i in range(30):
            yield (i % 4, i)

    view = sort(_ReiterableTable(('k', 'id'), rows), key='k',
                buffersize=1, cache=True, merge_fan_in=2,
                tempdir=str(tmp_path))

    # arm the failure on the third temporary output only
    real_open = sorts._open_run_output
    opened = [0]

    def arming_open(tempdir):
        f = real_open(tempdir)
        opened[0] += 1
        if state['armed'] and opened[0] == 3:
            f._fail_here = True
        return f

    monkeypatch.setattr(sorts, '_open_run_output', arming_open)
    with pytest.raises(_PickleFailure):
        list(iter(view))
    assert view._filecache is None
    eq_([], os.listdir(str(tmp_path)))

    # disarm and recover through the same view
    state['armed'] = False
    monkeypatch.undo()
    ieq(_expected(30, 4), view)


# ---------------------------------------------------------------------------
# upper-level transforms benefit from the shared default config
# ---------------------------------------------------------------------------


def test_join_uses_shared_sort_default(tmp_path, monkeypatch):
    # join() does not accept a fan-in argument; it must work under a tiny
    # configured fan-in with many on-disk runs. Its inner sort views cache
    # by default, so their final runs stay on disk until the join is dropped.
    import gc
    monkeypatch.setattr(config, 'sort_merge_fan_in', 2)
    left = [('k', 'lv')] + [(i % 6, i) for i in range(60)]
    right = [('k', 'rv')] + [(i % 6, 1000 + i) for i in range(60)]
    result = join(left, right, key='k', buffersize=2, tempdir=str(tmp_path))
    rows = list(result)
    eq_(1 + 60 * (60 // 6), len(rows))
    # output is ordered by the join key
    keys = [r[0] for r in rows[1:]]
    eq_(sorted(keys), keys)
    # each cached inner sort keeps at most the fan-in number of final runs
    inner_views = [result.left, result.right]
    for view in inner_views:
        assert view._filecache is not None
        assert len(view._filecache) <= 2
    # release every reference to the join and its inner views before the
    # GC fallback can reap the cached run files
    del result, rows, view, inner_views
    gc.collect()
    eq_([], os.listdir(str(tmp_path)))


def test_aggregate_uses_shared_sort_default(tmp_path, monkeypatch):
    import gc
    monkeypatch.setattr(config, 'sort_merge_fan_in', 2)
    table = [('k', 'v')] + [(i % 5, i) for i in range(60)]
    result = aggregate(table, key='k', aggregation={'v': len},
                       buffersize=2, tempdir=str(tmp_path))
    rows = sorted(list(result)[1:], key=lambda r: r[0])
    eq_([(0, 12), (1, 12), (2, 12), (3, 12), (4, 12)],
        [(r[0], r[1]) for r in rows])
    del result, rows
    gc.collect()
    eq_([], os.listdir(str(tmp_path)))


def test_mergesort_bounded_fan_in_end_to_end():
    t1 = [('k', 'id')] + [(i % 4, i) for i in range(50)]
    t2 = [('k', 'id')] + [(i % 4, 50 + i) for i in range(50)]
    actual = mergesort(t1, t2, key='k', buffersize=2, merge_fan_in=2,
                       cache=False)
    from petl.transform.basics import cat
    ieq(sort(cat(t1, t2), key='k'), actual)


# ---------------------------------------------------------------------------
# memory / chunk boundary compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('nrows_,buffersize', [
    (0, 2),       # header only
    (1, 2),       # less than one chunk (in-memory path)
    (2, 2),       # exactly one full chunk (disk path, single run)
    (3, 2),       # one full chunk plus remainder (two runs)
    (4, 2),       # exactly two chunks (single streaming merge)
])
def test_chunk_boundaries(nrows_, buffersize):
    table = [('k', 'id')] + [(i, i) for i in range(nrows_)]
    expected = [('k', 'id')] + sorted(table[1:], key=lambda r: r[0])
    view = sort(table, key='k', buffersize=buffersize, merge_fan_in=2)
    ieq(expected, view)


def test_buffersize_none_forces_in_memory():
    old = config.sort_buffersize
    config.sort_buffersize = None
    try:
        view = sort(_table(20), key='k', merge_fan_in=2)
        nrows(view)
        assert view._memcache is not None
        assert view._filecache is None
    finally:
        config.sort_buffersize = old


# ---------------------------------------------------------------------------
# real subprocess under a low RLIMIT_NOFILE soft cap
# ---------------------------------------------------------------------------


SUBPROCESS_SCRIPT = textwrap.dedent("""
    import json
    import os
    import resource
    import sys
    import tempfile

    sys.path.insert(0, %(repo)r)

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (%(limit)d, hard))

    import petl as etl
    import petl.config as config
    from petl.transform.joins import join
    from petl.transform.reductions import aggregate

    report = {}
    tmp_prefix = tempfile.gettempdir()

    def run_fds():
        out = []
        for name in os.listdir('/proc/self/fd'):
            try:
                target = os.readlink('/proc/self/fd/' + name)
            except OSError:
                continue
            if target.startswith(tmp_prefix) and os.path.isfile(target):
                out.append(target)
        return out

    n = %(nrows)d
    fan_in = %(fan_in)s
    reverse = %(reverse)s
    kwargs = dict(key='k', buffersize=1, cache=%(cache)s)
    if fan_in is not None:
        kwargs['merge_fan_in'] = fan_in
    else:
        fan_in = config.sort_merge_fan_in

    table = [('k', 'id')] + [(i %% 7, i) for i in range(n)]
    view = etl.sort(table, reverse=reverse, **kwargs)
    it = iter(view)
    hdr = next(it)
    peak = 0
    rows = []
    for row in it:
        rows.append(row)
        peak = max(peak, len(run_fds()))
    # expected order: groups by key (descending when reverse), but rows
    # keep their original input order within a key because the sort is
    # stable
    groups = {}
    for i in range(n):
        groups.setdefault(i %% 7, []).append((i %% 7, i))
    expected = []
    for k in sorted(groups, reverse=reverse):
        expected.extend(groups[k])
    assert hdr == ('k', 'id')
    assert rows == expected, (rows[:5], expected[:5])
    report['peak_run_fds'] = peak
    report['stats'] = view._last_sort_stats
    report['final_runs'] = (len(view._filecache)
                            if view._filecache is not None else None)

    # stability with equal keys across many merge rounds: equal keys must
    # keep original input order even when reversing
    stable = [('k', 'p')] + [(i %% 5, i) for i in range(n)]
    sview = etl.sort(stable, key='k', buffersize=2, cache=False,
                     merge_fan_in=fan_in, reverse=reverse)
    got = [r[1] for r in list(sview)[1:]]
    groups = {}
    for i in range(n):
        groups.setdefault(i %% 5, []).append(i)
    want = []
    for k in sorted(groups, reverse=reverse):
        want.extend(groups[k])
    assert got == want, 'unstable ordering across merge rounds'

    # join and aggregate without an explicit fan-in argument
    left = [('k', 'lv')] + [(i %% 6, i) for i in range(120)]
    right = [('k', 'rv')] + [(i %% 6, 1000 + i) for i in range(120)]
    jrows = list(join(left, right, key='k', buffersize=2))
    assert len(jrows) == 1 + 120 * (120 // 6)
    agg = list(aggregate(
        [('k', 'v')] + [(i %% 5, i) for i in range(120)],
        key='k', aggregation={'v': len}, buffersize=2))
    assert sorted(r[1] for r in agg[1:]) == [24] * 5

    # two independent cached readers interleaved from the start
    if %(cache)s:
        i1 = iter(view)
        i2 = iter(view)
        next(i1)  # header
        next(i2)  # header
        a = [next(i2) for _ in range(20)]
        b = [next(i1) for _ in range(10)]
        assert len(a) == 20 and len(b) == 10
        rest1 = list(i1)
        rest2 = list(i2)
        # each reader has consumed its header plus its data rows
        assert len(rest1) + 10 == n and len(rest2) + 20 == n

    print('REPORT ' + json.dumps(report))
""")


def _run_low_fd_subprocess(nrows_, fan_in, cache, reverse, limit=64):
    script = SUBPROCESS_SCRIPT % {
        'repo': '/workspace',
        'limit': limit,
        'nrows': nrows_,
        'fan_in': 'None' if fan_in is None else fan_in,
        'cache': 'True' if cache else 'False',
        'reverse': 'True' if reverse else 'False',
    }
    proc = subprocess.run(
        [sys.executable, '-I', '-c', script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    line = [ln for ln in proc.stdout.decode().splitlines()
            if ln.startswith('REPORT ')][0]
    return json.loads(line[len('REPORT '):])


@pytest.mark.parametrize('fan_in', [2, 3, None])
@pytest.mark.parametrize('cache', [True, False])
@pytest.mark.parametrize('reverse', [False, True])
def test_low_fd_subprocess(fan_in, cache, reverse):
    # 200 initial runs is far more than F**2 for every fan-in under test
    report = _run_low_fd_subprocess(200, fan_in, cache, reverse)
    effective_f = fan_in if fan_in is not None else config.sort_merge_fan_in
    assert report['peak_run_fds'] <= effective_f + 1, report
    assert report['stats']['initial_runs'] == 200
    expected_passes = _merge_passes(200, effective_f)
    eq_(expected_passes, report['stats']['merge_passes'])
    if cache:
        assert report['final_runs'] is not None
        assert report['final_runs'] <= effective_f
    else:
        assert report['final_runs'] is None


def test_low_fd_subprocess_matches_baseline_scenario():
    # exact reproduction of the reported failure shape
    script = textwrap.dedent("""
        import resource, sys
        sys.path.insert(0, '/workspace')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
        import petl as etl
        it = iter(etl.sort(
            [('k', 'id')] + [(i % 7, i) for i in range(160)],
            key='k', buffersize=1, cache=False))
        next(it)
        row = next(it)
        assert row == (0, 0), row
        print('OK')
    """)
    proc = subprocess.run([sys.executable, '-I', '-c', script],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode()
    assert b'OK' in proc.stdout
