"""ETL helpers for config preprocessing."""


def mask_dict_to_list(d, fillna=0, length=None):
    """Convert a sparse index-keyed dict to a dense list.

    Keys must be non-negative integers.  Missing indices are filled with
    ``fillna``; the fill value also determines the output element type via
    ``type(fillna)``.

    Parameters
    ----------
    d:
        Dict mapping integer layer indices to values (e.g. ``{0: True, 2: True}``).
    fillna:
        Value used for missing indices; its type is applied to every element.
    length:
        Desired list length.  Defaults to ``max(d.keys()) + 1`` so that all
        keys in *d* are included.

    Returns
    -------
    list
        Dense list of length *length* with ``type(fillna)`` elements.

    Examples
    --------
    >>> mask_dict_to_list({0: True, 2: True}, fillna=False, length=4)
    [True, False, True, False]
    >>> mask_dict_to_list({1: 1, 3: 1}, fillna=0)
    [0, 1, 0, 1]
    """
    if not d:
        return []
    length = length or max(d.keys()) + 1
    return [type(fillna)(d.get(i, fillna)) for i in range(length)]
