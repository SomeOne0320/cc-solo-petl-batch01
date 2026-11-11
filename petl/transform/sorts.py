from __future__ import absolute_import, print_function, division


import os
import errno
import heapq
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
import itertools
import logging
from collections import namedtuple
import operator
from petl.compat import pickle, next, text_type, integer_types


import petl.config as config
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, asindices


logger = logging.getLogger(__name__)
warning = logger.warning
info = logger.info
debug = logger.debug


def sort(table, key=None, reverse=False, buffersize=None, tempdir=None,
         cache=True, merge_fan_in=None):
    """
    Sort the table. Field names or indices (from zero) can be used to specify
    the key. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['C', 2],
        ...           ['A', 9],
        ...           ['A', 6],
        ...           ['F', 1],
        ...           ['D', 10]]
        >>> table2 = etl.sort(table1, 'foo')
        >>> table2
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   9 |
        +-----+-----+
        | 'A' |   6 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'F' |   1 |
        +-----+-----+

        >>> # sorting by compound key is supported
        ... table3 = etl.sort(table1, key=['foo', 'bar'])
        >>> table3
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   6 |
        +-----+-----+
        | 'A' |   9 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'F' |   1 |
        +-----+-----+

        >>> # if no key is specified, the default is a lexical sort
        ... table4 = etl.sort(table1)
        >>> table4
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   6 |
        +-----+-----+
        | 'A' |   9 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'F' |   1 |
        +-----+-----+

    The `buffersize` argument should be an `int` or `None`.

    If the number of rows in the table is less than `buffersize`, the table
    will be sorted in memory. Otherwise, the table is sorted in chunks of
    no more than `buffersize` rows, each chunk is written to a temporary file,
    and then a merge sort is performed on the temporary files.

    If `buffersize` is `None`, the value of
    `petl.config.sort_buffersize` will be used. By default this is
    set to 100000 rows, but can be changed, e.g.::

        >>> import petl.config
        >>> petl.config.sort_buffersize = 500000

    If `petl.config.sort_buffersize` is set to `None`, this forces
    all sorting to be done entirely in memory.

    By default the results of the sort will be cached, and so a second pass over
    the sorted table will yield rows from the cache and will not repeat the
    sort operation. To turn off caching, set the `cache` argument to `False`.

    The `merge_fan_in` argument should be an `int` no smaller than 2, or
    `None`. It bounds the number of temporary run files that a single external
    sort merge step opens at once. When more than `merge_fan_in` initial run
    files are produced, they are combined through repeated merge passes, each
    merging at most `merge_fan_in` runs at a time, so that the number of open
    file descriptors stays bounded regardless of the input size. If `None`,
    the value of `petl.config.sort_merge_fan_in` is used (32 by default).

    """

    return SortView(table, key=key, reverse=reverse, buffersize=buffersize,
                    tempdir=tempdir, cache=cache, merge_fan_in=merge_fan_in)


Table.sort = sort


def _open_run_output(tempdir):
    # open a new temporary file for writing a sorted run; module level so the
    # file handling can be observed/instrumented in tests
    return NamedTemporaryFile(dir=tempdir, delete=False, mode='wb')

def _open_run_input(name):
    # open an existing sorted run for reading; module level so the file
    # handling can be observed/instrumented in tests
    return open(name, 'rb')


def _iter_open_run(f):
    # iterate pickled rows from an already open run file
    try:
        while True:
            yield pickle.load(f)
    except EOFError:
        pass


def _iterchunk(fn):
    # reopen so iterators from file cache are independent
    debug('iterchunk, opening %s' % fn)
    with _open_run_input(fn) as f:
        for row in _iter_open_run(f):
            yield row
    debug('end of iterchunk, closed %s' % fn)


def _closing_generator(gen):
    # drive a generator and close it as soon as it is exhausted, so any
    # resource-releasing finally blocks run deterministically rather than
    # waiting for garbage collection
    try:
        for item in gen:
            yield item
    finally:
        gen.close()


class _Keyed(namedtuple('Keyed', ['key', 'obj'])):
    #  Override default behavior of namedtuple comparisons, only keys need to be compared for heapmerge
    def __eq__(self, other):
        return self.key == other.key

    def __lt__(self, other):
        return self.key < other.key

    def __le__(self, other):
        return self.key <= other.key

    def __ne__(self, other):
        return self.key != other.key

    def __gt__(self, other):
        return self.key > other.key

    def __ge__(self, other):
        return self.key >= other.key


def _heapqmergesorted(key=None, *iterables):
    """Return a single iterator over the given iterables, sorted by the
    given `key` function, assuming the input iterables are already sorted by
    the same function. (I.e., the merge part of a general merge sort.) Uses
    :func:`heapq.merge` for the underlying implementation."""

    if key is None:
        keyed_iterables = iterables
        for element in heapq.merge(*keyed_iterables):
            yield element
    else:
        keyed_iterables = [(_Keyed(key(obj), obj) for obj in iterable)
                           for iterable in iterables]
        for element in heapq.merge(*keyed_iterables):
            yield element.obj


def _shortlistmergesorted(key=None, reverse=False, *iterables):
    """Return a single iterator over the given iterables, sorted by the
    given `key` function, assuming the input iterables are already sorted by
    the same function. (I.e., the merge part of a general merge sort.) Uses
    :func:`min` (or :func:`max` if ``reverse=True``) for the underlying
    implementation."""

    if reverse:
        op = max
    else:
        op = min
    if key is not None:
        opkwargs = {'key': key}
    else:
        opkwargs = dict()
    # populate initial shortlist
    # (remember some iterables might be empty)
    iterators = list()
    shortlist = list()
    for iterable in iterables:
        it = iter(iterable)
        try:
            first = next(it)
            iterators.append(it)
            shortlist.append(first)
        except StopIteration:
            pass
    # do the mergesort
    while iterators:
        nxt = op(shortlist, **opkwargs)
        yield nxt
        nextidx = shortlist.index(nxt)
        try:
            shortlist[nextidx] = next(iterators[nextidx])
        except StopIteration:
            del shortlist[nextidx]
            del iterators[nextidx]


def _mergesorted(key=None, reverse=False, *iterables):
    # N.B., I've used heapq for normal merge sort and shortlist merge sort for
    # reverse merge sort because I've assumed that heapq.merge is faster and
    # so is preferable but it doesn't support reverse sorting so the shortlist
    # merge sort has to be used for reverse sorting. Some casual profiling
    # suggests there isn't much between the two in terms of speed, but might be
    # worth profiling more carefully

    if reverse:
        return _shortlistmergesorted(key, True, *iterables)
    else:
        return _heapqmergesorted(key, *iterables)


def _validate_merge_fan_in(value):
    # bool is a subclass of int; reject it explicitly to avoid silent
    # misconfiguration (e.g. merge_fan_in=True would behave like fan-in 1)
    if isinstance(value, bool) or not isinstance(value, integer_types):
        raise ValueError(
            'merge_fan_in must be an integer no smaller than 2, found %r'
            % (value,))
    if value < 2:
        raise ValueError(
            'merge_fan_in must be an integer no smaller than 2, found %r'
            % (value,))
    return value


class _NamedTempFileDeleteOnGC(object):

    def __init__(self, name):
        self.name = name
        # N.B., normal cleanup is deterministic; this guards the GC fallback
        # so files already reaped explicitly are not deleted a second time
        self._deleted = False

    def delete(self, unlink=os.unlink, log=logger.debug):
        if self._deleted:
            return
        name = self.name
        try:
            log('deleting %s' % name)
            unlink(name)
        except OSError as e:
            # treat a vanished file as successfully reaped: something else
            # (e.g. an OS temp cleanup) removed it first
            if e.errno != errno.ENOENT:
                log('exception deleting %s: %s' % (name, e))
                raise
        else:
            log('deleted %s' % name)
        self._deleted = True

    def safe_delete(self):
        # used on cleanup paths where a secondary failure must not mask the
        # original exception
        try:
            self.delete()
        except Exception as e:
            logger.warning('could not delete temporary run %s: %s',
                           self.name, e)

    def __del__(self):
        # last-resort cleanup if explicit lifecycle management missed the
        # file; deterministic cleanup must not depend on this running
        try:
            self.safe_delete()
        except Exception:
            pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class _FileCacheGeneration(object):
    """A published generation of sorted run files.

    Run files are shared by all readers of the same cache generation. The
    underlying temporary files are deleted once the generation has been
    retired (e.g. by :meth:`SortView.clearcache` or a rebuild) and every
    reader started from it has finished. The header and key function needed
    to read the runs are snapshotted here, so a reader stays independent of
    later cache generations on the same view.
    """

    def __init__(self, runs, header, getkey, reverse):
        self.runs = list(runs)
        self.header = header
        self.getkey = getkey
        self.reverse = reverse
        # the owning view holds one reference; each active reader adds one
        self._refs = 1
        self._retired = False
        self._cleaned = False

    def __len__(self):
        return len(self.runs)

    def acquire(self):
        self._refs += 1
        return self

    def release(self):
        self._refs -= 1
        self._maybe_cleanup()

    def retire(self):
        # the owning view no longer publishes this generation: release its
        # owner reference. Active readers keep their own references, so the
        # files are reaped only when the last of them finishes.
        if not self._retired:
            self._retired = True
            self._refs -= 1
            self._maybe_cleanup()

    def _maybe_cleanup(self):
        if self._cleaned:
            return
        if self._retired and self._refs == 0:
            self._cleaned = True
            for run in self.runs:
                run.safe_delete()

    def __del__(self):
        # fallback only; normal cleanup is fully deterministic
        try:
            if not self._cleaned:
                self._cleaned = True
                for run in self.runs:
                    run.safe_delete()
        except Exception:
            pass


class _ExternalSorter(object):
    """Build and stream an external (disk-backed) sort with bounded fan-in.

    Owns all temporary run files created for one SortView build so that:

    * a merge step never opens more than `fan_in` input runs plus one output
      run, regardless of how many initial runs the input is split into;
    * intermediate runs are deleted as soon as the merge pass consuming them
      has finished;
    * every file is deterministically closed and deleted on completion,
      early close or failure, without relying on garbage collection.
    """

    def __init__(self, tempdir, fan_in):
        self.tempdir = tempdir
        self.fan_in = fan_in
        # runs making up the current state of the merge pipeline
        self.runs = []
        # every run still owned by this build (initial and intermediate)
        self._owned = set()
        # bookkeeping for resource accounting / tests
        self._open_files = set()
        self.peak_open_run_files = 0
        self.runs_created = 0
        self.initial_runs = 0
        self.merge_passes = 0

    # ------------------------------------------------------------------
    # open file accounting
    # ------------------------------------------------------------------

    def _note_opened(self, f):
        self._open_files.add(f)
        if len(self._open_files) > self.peak_open_run_files:
            self.peak_open_run_files = len(self._open_files)

    def _note_closed(self, f):
        self._open_files.discard(f)

    def _close(self, f):
        if f is not None and not f.closed:
            f.close()
        self._note_closed(f)

    @contextmanager
    def _track(self, f):
        self._note_opened(f)
        try:
            yield f
        finally:
            self._close(f)

    # ------------------------------------------------------------------
    # run lifecycle
    # ------------------------------------------------------------------

    def _register_run(self, run):
        # ownership accounting only; the current pipeline list (`self.runs`)
        # is maintained by the caller so an in-progress merge pass can never
        # pick up runs created by that same pass
        self._owned.add(run)
        self.runs_created += 1

    def _discard(self, run):
        # an input run has been fully consumed by a merge pass: remove it
        # from the build ownership set and delete it immediately
        self._owned.discard(run)
        run.safe_delete()

    def write_run(self, rows):
        # write one fully sorted chunk to a new temporary run file
        debug('creating temporary chunk run in %r', self.tempdir)
        f = _open_run_output(self.tempdir)
        run = _NamedTempFileDeleteOnGC(f.name)
        self._register_run(run)
        self.initial_runs += 1
        with self._track(f):
            for row in rows:
                pickle.dump(row, f, protocol=-1)
            f.flush()
        self.runs.append(run)
        return run

    def _merge_batch(self, batch, getkey, reverse):
        # merge one batch of at most fan_in sorted runs into a new run;
        # at most len(batch) input files + 1 output file are open here
        debug('merging batch of %d run(s)', len(batch))
        out = _open_run_output(self.tempdir)
        out_run = _NamedTempFileDeleteOnGC(out.name)
        self._register_run(out_run)
        infiles = []
        try:
            with self._track(out):
                try:
                    for run in batch:
                        f = _open_run_input(run.name)
                        infiles.append(f)
                        self._note_opened(f)
                    chunkiters = [_iter_open_run(f) for f in infiles]
                    merged = _mergesorted(getkey, reverse, *chunkiters)
                    for row in merged:
                        pickle.dump(row, out, protocol=-1)
                    out.flush()
                finally:
                    # release input handles before any outer cleanup runs
                    for f in infiles:
                        self._close(f)
        except BaseException:
            # the partial output run must not survive a failed build
            self._discard(out_run)
            raise
        return out_run

    def reduce_runs(self, getkey, reverse):
        # bounded fan-in multi-pass merge until the remaining runs can be
        # merged directly while streaming the output (or cached, with at most
        # fan_in final run files kept on disk)
        while len(self.runs) > self.fan_in:
            debug('merge pass %d over %d runs with fan-in %d',
                  self.merge_passes + 1, len(self.runs), self.fan_in)
            current = self.runs
            next_runs = []
            for start in range(0, len(current), self.fan_in):
                batch = current[start:start + self.fan_in]
                if len(batch) == 1:
                    # nothing to merge: carry the single run forward
                    # unchanged, preserving run ordering; it stays owned
                    next_runs.append(batch[0])
                    continue
                # merge the batch into a fresh run; freshly merged runs
                # live in next_runs and never feed back into the same pass
                out_run = self._merge_batch(batch, getkey, reverse)
                next_runs.append(out_run)
                # the batch inputs are now fully consumed and no longer
                # needed; singleton runs carried forward are kept
                for run in batch:
                    self._discard(run)
            # swap only after the whole pass completes: freshly merged runs
            # live in next_runs and never feed back into the same pass
            self.runs = next_runs
            self.merge_passes += 1

    def stream(self, getkey, reverse, final_runs=None):
        # final streaming merge over at most fan_in runs; closes every input
        # and deletes every run file deterministically when exhausted, closed
        # early or abandoned on error. `final_runs` also keeps the wrappers
        # referenced so their GC fallback cannot reap files mid-stream.
        if final_runs is None:
            final_runs = list(self.runs)
        files = []
        try:
            for run in final_runs:
                f = _open_run_input(run.name)
                files.append(f)
                self._note_opened(f)
            chunkiters = [_iter_open_run(f) for f in files]
            for row in _mergesorted(getkey, reverse, *chunkiters):
                yield tuple(row)
        finally:
            for f in files:
                self._close(f)
            # N.B., close files before deleting (Windows cannot unlink open
            # files); deletion is idempotent, so any later GC fallback is a
            # no-op
            for run in final_runs:
                run.safe_delete()

    def publish(self, header, getkey, reverse):
        # transfer ownership of the final runs to a cache generation,
        # snapshotting everything a reader needs so it is unaffected by
        # later clearcache()/rebuild cycles on the owning view
        runs = list(self.runs)
        self.runs = []
        self._owned.clear()
        return _FileCacheGeneration(runs, header, getkey, reverse)

    def release(self):
        # stop tracking the final runs without deleting them: the streaming
        # output merge still has them open and is responsible for closing
        # the handles; their wrappers' GC fallback (or an error-path call
        # to cleanup) reaps the files themselves
        self.runs = []
        self._owned.clear()

    def cleanup(self):
        # delete every run still owned by this build; runs transferred to a
        # published cache generation or released to a live stream are not
        # touched
        for run in list(self._owned):
            run.safe_delete()
        self._owned.clear()
        self.runs = []
        for f in list(self._open_files):
            self._close(f)

    def stats(self):
        return {
            'initial_runs': self.initial_runs,
            'runs_created': self.runs_created,
            'merge_passes': self.merge_passes,
            'peak_open_run_files': self.peak_open_run_files,
        }


class SortView(Table):
    def __init__(self, source, key=None, reverse=False, buffersize=None,
                 tempdir=None, cache=True, merge_fan_in=None):
        self.source = source
        self.key = key
        self.reverse = reverse
        if buffersize is None:
            self.buffersize = config.sort_buffersize
        else:
            self.buffersize = buffersize
        self.tempdir = tempdir
        self.cache = cache
        if merge_fan_in is None:
            merge_fan_in = config.sort_merge_fan_in
        # validate before any temporary file could be created
        self.merge_fan_in = _validate_merge_fan_in(merge_fan_in)
        self._hdrcache = None
        self._memcache = None
        self._filecache = None
        self._getkey = None
        # resource accounting from the most recent external build (tests and
        # diagnostics; None when no disk-backed build has run)
        self._last_sort_stats = None

    def clearcache(self):
        debug('clear cache')
        if self._filecache is not None:
            # readers already iterating the old generation may keep reading;
            # its files are reaped when the last one finishes
            self._filecache.retire()
        self._hdrcache = None
        self._memcache = None
        self._filecache = None
        self._getkey = None
        self._last_sort_stats = None

    def __iter__(self):
        if self.cache and self._memcache is not None:
            return self._iterfrommemcache()
        elif self.cache and self._filecache is not None:
            # wrap so the generation reference release and run handle closing
            # happen on normal exhaustion, not only on close() or GC
            return _closing_generator(self._iterfromfilecache())
        else:
            # wrap so the build's deterministic cleanup (closing run handles
            # and deleting temp files) runs on normal exhaustion too
            return _closing_generator(
                self._iternocache(self.source, self.key, self.reverse))

    def _iterfrommemcache(self):
        debug('iterate from memory cache')
        yield tuple(self._hdrcache)
        for row in self._memcache:
            yield tuple(row)

    def _iterfromfilecache(self):
        # create a reference to the filecache here, so cleanup happens in the
        # correct order
        filecache = self._filecache
        generation = filecache.acquire()
        try:
            for row in self._itergeneration(generation):
                yield row
        finally:
            generation.release()
            del filecache
            debug('exiting generator')

    def _itergeneration(self, generation):
        # yield rows from one cache generation, using its snapshotted header
        # and key function so a clearcache()/rebuild racing with this reader
        # cannot null out what it needs. The caller owns the generation
        # reference and is responsible for acquire()/release().
        filenames = [run.name for run in generation.runs]
        debug('iterate from file cache: %r', filenames)
        chunkiters = None
        rows = None
        try:
            yield tuple(generation.header)
            # at most merge_fan_in run files are opened by one reader
            chunkiters = [_iterchunk(fn) for fn in filenames]
            rows = _mergesorted(generation.getkey, generation.reverse,
                                *chunkiters)
            for row in rows:
                yield tuple(row)
        finally:
            # N.B., close open files before any temp file deletion (deletion
            # of an open file fails on Windows)
            debug('attempt cleanup from generator')
            del chunkiters
            del rows

    def _iternocache(self, source, key, reverse):
        debug('iterate without cache')
        # a previous cache generation must remain readable for iterators
        # already using it, but no longer be published to new readers
        self.clearcache()
        it = iter(source)

        try:
            hdr = next(it)
        except StopIteration:
            if key is None:
                return  # nothing to do on a table without headers
            hdr = []
        yield tuple(hdr)

        if key is not None:
            # convert field selection into field indices
            indices = asindices(hdr, key)
        else:
            indices = range(len(hdr))
        if len(indices) == 0:
            # nothing to compare on, e.g., a table with no fields
            getkey = None
        else:
            # now use field indices to construct a _getkey function
            getkey = comparable_itemgetter(*indices)

        # TODO support native comparison

        # initialise the first chunk
        rows = list(itertools.islice(it, 0, self.buffersize))
        rows.sort(key=getkey, reverse=reverse)

        # have we exhausted the source iterator?
        if self.buffersize is None or len(rows) < self.buffersize:
            # yes, table fits within sort buffer

            if self.cache:
                debug('caching mem')
                self._hdrcache = hdr
                self._memcache = rows
                # actually not needed to iterate from memcache
                self._getkey = getkey

            for row in rows:
                yield tuple(row)

        else:
            # no, table is too big, need to sort in chunks on disk
            sorter = _ExternalSorter(self.tempdir, self.merge_fan_in)
            try:
                sorter.write_run(rows)

                # grab and dump further chunks as they arrive
                while True:
                    rows = list(itertools.islice(it, 0, self.buffersize))
                    if not rows:
                        break
                    rows.sort(key=getkey, reverse=reverse)
                    sorter.write_run(rows)

                # bounded fan-in multi-pass merge down to at most
                # merge_fan_in final runs, deleting intermediate runs as
                # soon as each pass completes
                sorter.reduce_runs(getkey, reverse)
                self._last_sort_stats = sorter.stats()

                if self.cache:
                    # publish only after the whole build has succeeded;
                    # a failed build never leaves a half-finished cache
                    debug('caching files: %d final run(s)', len(sorter.runs))
                    generation = sorter.publish(hdr, getkey, reverse)
                    self._hdrcache = hdr
                    self._filecache = generation
                    self._getkey = getkey
                    # stream from the published generation through the same
                    # reader path used by subsequent iterations. Take an
                    # explicit reader reference and return it in the finally
                    # so accounting is deterministic regardless of generator
                    # finalisation timing.
                    generation.acquire()
                    cache_iter = self._itergeneration(generation)
                    try:
                        next(cache_iter)  # header already yielded above
                        for row in cache_iter:
                            yield row
                    finally:
                        cache_iter.close()
                        generation.release()
                else:
                    # streaming final merge. Take ownership of the final
                    # runs away from the build (so the outer cleanup cannot
                    # delete files the stream is still reading); the stream
                    # itself closes its handles and deletes the run files
                    # deterministically on completion, early close or error
                    final_runs = sorter.runs
                    sorter.release()
                    for row in sorter.stream(getkey, reverse, final_runs):
                        yield row
            finally:
                # cache=False builds (and failed builds of either mode):
                # close handles and delete every run of this operation.
                # For a successfully published cache, publish() transferred
                # ownership so cleanup() has nothing to delete.
                sorter.cleanup()


def mergesort(*tables, **kwargs):
    """
    Combine multiple input tables into one sorted output table. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['A', 9],
        ...           ['C', 2],
        ...           ['D', 10],
        ...           ['A', 6],
        ...           ['F', 1]]
        >>> table2 = [['foo', 'bar'],
        ...           ['B', 3],
        ...           ['D', 10],
        ...           ['A', 10],
        ...           ['F', 4]]
        >>> table3 = etl.mergesort(table1, table2, key='foo')
        >>> table3.lookall()
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   9 |
        +-----+-----+
        | 'A' |   6 |
        +-----+-----+
        | 'A' |  10 |
        +-----+-----+
        | 'B' |   3 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'F' |   1 |
        +-----+-----+
        | 'F' |   4 |
        +-----+-----+

    If the input tables are already sorted by the given key, give
    ``presorted=True`` as a keyword argument.

    This function is equivalent to concatenating the input tables using
    :func:`cat` then sorting, however this function will typically be more
    efficient, especially if the input tables are presorted.

    The output header is determined in the same way as for :func:`cat`. If
    no ``header`` is given, output fields are gathered from all of the
    input tables in order of first appearance. Otherwise the given
    ``header`` selects and orders the output fields, and fields not
    included in it do not take part in comparisons. Before merging, each
    input row is standardised to the output fields: missing fields and
    short rows are filled with the value of ``missing``, and values
    belonging to fields not in the output header are discarded.

    Comparisons use the same rules as :func:`sort`, so values of mixed
    types (e.g., numbers and text, or values and ``None``) are ordered
    consistently whether or not a ``key`` is given. The merge is stable:
    rows with equal keys keep their original order within each input
    table, and ties between input tables are broken by the order in which
    the tables were given.

    If ``presorted`` is ``False`` (the default), each input table is
    standardised and then sorted individually, using the same external
    sorting machinery as :func:`sort` (see the ``buffersize``, ``tempdir``,
    ``cache`` and ``merge_fan_in`` arguments), and the sorted tables are
    then merged. If ``presorted`` is ``True``, the input tables are assumed
    to already be sorted according to the same output semantics and are
    merged directly, in a single streaming pass over the inputs.

    Keyword arguments:

    key : string or tuple of strings, optional
        Field name or tuple of fields to sort by (defaults to `None` lexical
        sort)
    reverse : bool, optional
        `True` if sort in reverse (descending) order (defaults to `False`)
    presorted : bool, optional
        `True` if inputs are already sorted by the given key (defaults to
        `False`)
    missing : object
        Value to fill with when input tables have different fields (defaults
        to `None`)
    header : sequence of strings, optional
        Specify a fixed header for the output table
    buffersize : int, optional
        Limit the number of rows in memory per input table when inputs are
        not presorted
    tempdir : string, optional
        Directory to use for temporary files when sorting inputs that do
        not fit in the sort buffer (defaults to the system default)
    cache : bool, optional
        If `True` (the default) and inputs are not presorted, the results
        of sorting each input table are cached, so that a second pass over
        the merged table does not repeat the sort
    merge_fan_in : int, optional
        Maximum number of temporary run files opened by one merge step of
        the external sort used for each input table (defaults to
        `petl.config.sort_merge_fan_in`, 32); must be an integer no smaller
        than 2

    """

    return MergeSortView(tables, **kwargs)


Table.mergesort = mergesort


class MergeSortView(Table):
    def __init__(self, tables, key=None, reverse=False, presorted=False,
                 missing=None, header=None, buffersize=None, tempdir=None,
                 cache=True, merge_fan_in=None):
        self.key = key
        self.reverse = reverse
        self.presorted = presorted
        self.missing = missing
        self.header = header
        # validate fan-in up front, before any temporary file could exist
        if merge_fan_in is None:
            merge_fan_in = config.sort_merge_fan_in
        self.merge_fan_in = _validate_merge_fan_in(merge_fan_in)
        if presorted:
            self.tables = tables
        else:
            self.sources = list(tables)
            # N.B., each input table is standardised to the output fields
            # *before* being sorted, so that each input is sorted according
            # to the same output semantics. The sort views are created here
            # and kept on self so that their caches persist across multiple
            # iterations when cache=True; their sources are (re-)assigned
            # on each pass in _itersortedmerge.
            self.sortviews = [SortView(None, key=key, reverse=reverse,
                                       buffersize=buffersize,
                                       tempdir=tempdir, cache=cache,
                                       merge_fan_in=merge_fan_in)
                              for _ in self.sources]

    def __iter__(self):
        if self.presorted:
            return itermergesort(self.tables, self.key, self.header,
                                 self.missing, self.reverse)
        else:
            return self._itersortedmerge()

    def _itersortedmerge(self):
        # standardise all inputs to a common output header, sort each
        # standardised input individually, then merge
        state = _StandardiseState(self.sources, self.header, self.missing)
        for i, sortview in enumerate(self.sortviews):
            sortview.source = state.standardise_table(i)
        for row in itermergesort(self.sortviews, self.key, self.header,
                                 self.missing, self.reverse):
            yield row


def _standardiserows(rows, flds, ofs, missing):
    # standardise the data rows of one input table to the output fields
    # `ofs`, given the table's own fields `flds`; fill missing values with
    # `missing` and discard values belonging to fields not in `ofs`
    for row in rows:
        try:
            # should be quickest to do this way
            yield tuple(row[flds.index(fo)] if fo in flds else missing
                        for fo in ofs)
        except IndexError:
            # handle short rows
            outrow = [missing] * len(ofs)
            for i, fi in enumerate(flds):
                if fi in ofs:
                    try:
                        outrow[ofs.index(fi)] = row[i]
                    except IndexError:
                        pass  # be relaxed about short rows
            yield tuple(outrow)


class _StandardiseState(object):
    """Coordinate the standardisation of one or more input tables to a
    common output header within a single iteration pass."""

    def __init__(self, sources, header, missing):
        self.sources = sources
        self.missing = missing
        # N.B., one header row is read from each input table in order to
        # determine the output fields
        self.iterators = [iter(t) for t in sources]
        self.headers = list()
        for it in self.iterators:
            try:
                self.headers.append(next(it))
            except StopIteration:
                self.headers.append([])
        if header is None:
            # determine output fields by gathering all fields found in the
            # sources, in order of first appearance
            outhdr = list()
            for hdr in self.headers:
                for f in list(map(text_type, hdr)):
                    if f not in outhdr:
                        # add any new field as we find them
                        outhdr.append(f)
        else:
            # predetermined output fields
            outhdr = list(header)
        self.outhdr = outhdr

    def standardise_rows(self, i):
        # iterate over the data rows of input table `i`, standardised to
        # the output fields
        flds = list(map(text_type, self.headers[i]))
        return _standardiserows(self.iterators[i], flds, self.outhdr,
                                self.missing)

    def standardise_table(self, i):
        # as standardise_rows(), but yield the output header row first, so
        # the result can be used as a table in its own right
        yield tuple(self.outhdr)
        for row in self.standardise_rows(i):
            yield row


def itermergesort(sources, key, header, missing, reverse):
    # first need to standardise headers of all input tables
    state = _StandardiseState(sources, header, missing)
    outhdr = state.outhdr
    yield tuple(outhdr)

    # wrap all iterators to standardise fields
    sits = [state.standardise_rows(i) for i in range(len(sources))]

    # now determine key function
    if key is not None:
        # convert field selection into field indices
        indices = asindices(outhdr, key)
        # now use field indices to construct a _getkey function
        getkey = comparable_itemgetter(*indices)
    elif outhdr:
        # N.B., compare complete output rows using Comparable, so that
        # mixed types are ordered using the same rules as petl.sort
        getkey = comparable_itemgetter(*range(len(outhdr)))
    else:
        getkey = None

    # OK, do the merge sort
    for row in _mergesorted(getkey, reverse, *sits):
        yield row


def issorted(table, key=None, reverse=False, strict=False):
    """
    Return True if the table is ordered (i.e., sorted) by the given key. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['a', 1, True],
        ...           ['b', 3, True],
        ...           ['b', 2]]
        >>> etl.issorted(table1, key='foo')
        True
        >>> etl.issorted(table1, key='bar')
        False
        >>> etl.issorted(table1, key='foo', strict=True)
        False
        >>> etl.issorted(table1, key='foo', reverse=True)
        False

    """

    # determine the operator to use when comparing rows
    if reverse and strict:
        op = operator.lt
    elif reverse and not strict:
        op = operator.le
    elif strict:
        op = operator.gt
    else:
        op = operator.ge

    it = iter(table)
    try:
        flds = [text_type(f) for f in next(it)]
    except StopIteration:
        flds = []
    if key is None:
        prev = next(it)
        for curr in it:
            if not op(curr, prev):
                return False
            prev = curr
    else:
        getkey = comparable_itemgetter(*asindices(flds, key))
        prev = next(it)
        prevkey = getkey(prev)
        for curr in it:
            currkey = getkey(curr)
            if not op(currkey, prevkey):
                return False
            prevkey = currkey
    return True


Table.issorted = issorted
