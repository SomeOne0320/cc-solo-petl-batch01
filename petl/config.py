"""Global configuration defaults for petl.

The values in this module are used when a transformation does not receive
the corresponding argument explicitly. In particular, external sorting
(:func:`petl.transform.sorts.sort` and transforms built on it such as
``join`` or ``aggregate``) reads:

* :data:`sort_buffersize` - number of rows held in the in-memory sort
  buffer before spilling to a temporary run file (``None`` forces an
  entirely in-memory sort);
* :data:`sort_merge_fan_in` - maximum number of temporary run files opened
  by a single external sort merge step.

Both can be overridden per call; ``sort_merge_fan_in`` can also be changed
globally, e.g.::

    >>> import petl.config
    >>> petl.config.sort_merge_fan_in = 16
"""
from __future__ import division, print_function, absolute_import


from petl.compat import text_type


look_style = 'grid'  # alternatives: 'simple', 'minimal'
look_limit = 5
look_index_header = False
look_vrepr = repr
look_width = None
see_limit = 5
see_index_header = False
see_vrepr = repr
display_limit = 5
display_index_header = False
display_vrepr = text_type
sort_buffersize = 100000
sort_merge_fan_in = 32
"""
Default maximum number of temporary run files that a single external sort
merge step opens at once. Used when ``merge_fan_in`` is not given explicitly
to :func:`petl.transform.sorts.sort`, :class:`petl.transform.sorts.SortView`,
:class:`petl.transform.sorts.MergeSortView` or
:func:`petl.transform.sorts.mergesort`. Must be an ``int`` (not ``bool``)
no smaller than 2. When an external sort produces more than this many
initial run files, they are combined through repeated, bounded fan-in merge
passes instead of being opened all at once.
"""
failonerror=False # False, True, 'inline'
"""
Controls what happens when unhandled exceptions are raised in a
transformation:

    - If `False`, exceptions are suppressed.  If present, the value
      provided in the `errorvalue` argument is returned.

    - If `True`, the first unhandled exception is raised.

    - If `'inline'`, unhandled exceptions are returned.
"""
