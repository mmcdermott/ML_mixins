import functools


def doublewrap(f):
    """
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """

    @functools.wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec


QUANT_T = tuple[float | None, str]
UNITS_LIST_T = list[QUANT_T]


def normalize_unit(val: QUANT_T, cutoffs_and_units: UNITS_LIST_T) -> QUANT_T:
    """Converts a quantity to the largest possible unit and returns the quantity and unit.

    Args:
        val: A tuple of a number and a unit to be normalized.
        cutoffs_and_units: A list of tuples of valid cutoffs and units.

    Returns:
        A tuple of the normalized value and unit.

    Raises:
        LookupError: If the unit is not in the list of valid units.

    Examples:
        >>> normalize_unit((1000, "ms"), [(1000, "ms"), (60, "s"), (60, "min"), (None, "h")])
        (1.0, 's')
        >>> normalize_unit((720000, "ms"), [(1000, "ms"), (60, "s"), (60, "min"), (None, "h")])
        (12.0, 'min')
        >>> normalize_unit((3600, "s"), [(1000, "ms"), (60, "s"), (60, "min"), (None, "h")])
        (1.0, 'h')
        >>> normalize_unit((5000, "ns"), [(1000, "ms"), (60, "s"), (60, "min"), (None, "h")])
        Traceback (most recent call last):
            ...
        LookupError: Passed unit ns invalid! Must be one of ms, s, min, h.
    """
    x, x_unit = val
    x_unit_factor = 1
    for fac, unit in cutoffs_and_units:
        if unit == x_unit:
            break
        if fac is None:
            raise LookupError(
                f"Passed unit {x_unit} invalid! "
                f"Must be one of {', '.join(u for f, u in cutoffs_and_units)}."
            )
        x_unit_factor *= fac

    min_unit = x * x_unit_factor
    upper_bound = 1
    for upper_bound_factor, unit in cutoffs_and_units:
        if (upper_bound_factor is None) or (min_unit < upper_bound * upper_bound_factor):
            return min_unit / upper_bound, unit
        upper_bound *= upper_bound_factor


def pprint_quant(
    cutoffs_and_units: UNITS_LIST_T, val: QUANT_T, n_times: int = 1, std_val: float | None = None
) -> str:
    """Pretty prints a quantity with its unit, reflecting both a possible variance and # of times measured.

    Args:
        val: A tuple of a number and a unit to be normalized.
        cutoffs_and_units: A list of tuples of valid cutoffs and units to normalize the passed units.
        n_times: The number of times the quantity was measured.
        std_val: The standard deviation of the quantity (in the same units as val).

    Returns:
        A string representation of the quantity with its unit.

    Examples:
        >>> pprint_quant([(1000, "ms"), (60, "s"), (60, "min"), (None, "h")], (1000, "ms"))
        '1.0 s'
        >>> pprint_quant([(1000, "ms"), (60, "s"), (60, "min"), (None, "h")], (1000, "ms"), n_times=3)
        '1.0 s (x3)'
        >>> pprint_quant([(1000, "ms"), (60, "s"), (60, "min"), (None, "h")], (1000, "ms"), std_val=100)
        '1.0 ± 0.1 s'
        >>> pprint_quant([(1000, "ms"), (60, "s"), (60, "min")], (1200, "ms"), n_times=3, std_val=100)
        '1.2 ± 0.1 s (x3)'
    """
    norm_val, unit = normalize_unit(val, cutoffs_and_units)
    if std_val is not None:
        unit_conversion_factor = norm_val / val[0]
        std_norm_val = std_val * unit_conversion_factor

        mean_std_str = f"{norm_val:.1f} ± {std_norm_val:.1f} {unit}"
    else:
        mean_std_str = f"{norm_val:.1f} {unit}"

    if n_times > 1:
        return f"{mean_std_str} (x{n_times})"
    else:
        return mean_std_str


def pprint_stats_map(
    stats: dict[str, tuple[QUANT_T, int, float | None]], cutoffs_and_units: UNITS_LIST_T
) -> str:
    """Pretty prints a dictionary of summary statistics of quantities with their units, respecting key length.

    Args:
        stats: A dictionary of tuples of a number and a unit to be normalized, the number of times measured,
            and the standard deviation of the quantity (in the same units as the number).
        cutoffs_and_units: A list of tuples of valid cutoffs and units to normalize the passed units.

    Returns:
        A string representation of the dictionary of quantities with their units. This string representation
        will be ordered by the greatest total value (mean times number of measurements) of the quantities.

    Examples:
        >>> print(pprint_stats_map(
        ...     {"foo": ((1000, "ms"), 3, 100), "foobar": ((1000, "ms"), 1, None)},
        ...     [(1000, "ms"), (60, "s"), (60, "min"), (None, "h")]
        ... ))
        foobar: 1.0 s
        foo:    1.0 ± 0.1 s (x3)
        >>> print(pprint_stats_map(
        ...     {"foo": ((1000, "ms"), 3, 100), "foobar": ((72, "s"), 1, None)},
        ...     [(1000, "ms"), (60, "s"), (60, "min"), (None, "h")]
        ... ))
        foo:    1.0 ± 0.1 s (x3)
        foobar: 1.2 min
        >>> pprint_stats_map(
        ...     {"foo": ((1000, "ms"), 3, 100), "foobar": ((72, "y"), 1, None)},
        ...     [(1000, "ms"), (60, "s"), (60, "min"), (None, "h")]
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Unit y in stats key foobar not found in cutoffs_and_units!
    """

    def total_val(key: str) -> float:
        X_val, X_unit = stats[key][0]
        factor = 1
        for fac, unit in cutoffs_and_units:
            if unit == X_unit:
                break
            if fac is None:
                raise ValueError(f"Unit {X_unit} in stats key {key} not found in cutoffs_and_units!")
            factor *= fac

        base_unit_val = X_val * factor
        n_times = stats[key][1]

        return base_unit_val * n_times

    longest_key_length = max(len(k) for k in stats)
    ordered_keys = sorted(stats.keys(), key=total_val, reverse=False)
    return "\n".join(
        (f"{k}:{' '*(longest_key_length - len(k))} " f"{pprint_quant(cutoffs_and_units, *stats[k])}")
        for k in ordered_keys
    )
