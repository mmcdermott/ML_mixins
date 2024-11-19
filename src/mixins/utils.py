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

    Example:
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
