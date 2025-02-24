from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
M = TypeVar("M")


def add_mixin(cls: type[T], mixin: type[M], methods_to_decorate: dict[str, Callable]) -> type[T]:
    """Add a mixin to a class and decorate its methods.

    Args:
        cls: The class to add the mixin to.
        mixin: The mixin class to add to the class.
        methods_to_decorate: A dictionary of method names and decorators to apply to the methods.

    Returns:
        A new class with the mixin added and the methods decorated.

    Raises:
        ValueError: If an attribute in methods_to_decorate is not callable or not found in the class, or if
            the decorator is not callable.

    Examples:
    If we define a simple base class, of course it only has the prescribed methods.
        >>> class Base:
        ...     BAZ = 'hi'
        ...     def __init__(self, x):
        ...         self.x = x
        ...     def foo(self):
        ...         return self.x
        >>> base = Base(1)
        >>> base.BAZ
        'hi'
        >>> base.foo()
        1
        >>> base.seen
        Traceback (most recent call last):
            ...
        AttributeError: 'Base' object has no attribute 'seen'
        >>> base.bar()
        Traceback (most recent call last):
            ...
        AttributeError: 'Base' object has no attribute 'bar'
        >>> base.__class__.__name__
        'Base'

    Now, if we add the BarableMixin, we gain those capabilities.
        >>> class BarableMixin:
        ...     def bar(self):
        ...         return self.x + 1
        ...     @staticmethod
        ...     def decorator(func):
        ...         def wrapper(self):
        ...             self.seen = True
        ...             return func(self)
        ...         return wrapper
        >>> new_class = add_mixin(Base, BarableMixin, {"foo": BarableMixin.decorator})
        >>> obj = new_class(1)
        >>> obj.BAZ
        'hi'
        >>> obj.foo()
        1
        >>> obj.seen
        True
        >>> obj.bar()
        2
        >>> obj.__class__.__name__
        'BarableBase'

    If we try to add a decorator to a method that doesn't exist in the base class or to a non-method we get an
    error.
        >>> new_class = add_mixin(Base, BarableMixin, {"BAZ": BarableMixin.decorator})
        Traceback (most recent call last):
            ...
        ValueError: Attribute BAZ is not callable but is in methods_to_decorate!
        >>> new_class = add_mixin(Base, BarableMixin, {"not_found": BarableMixin.decorator})
        Traceback (most recent call last):
            ...
        ValueError: Method not_found not found in class Base
        >>> new_class = add_mixin(Base, BarableMixin, {"foo": 123})
        Traceback (most recent call last):
            ...
        ValueError: Decorator for foo is not callable!
    """
    new_attrs = {}

    for name, decorator in methods_to_decorate.items():
        if not callable(decorator):
            raise ValueError(f"Decorator for {name} is not callable!")
        if name not in cls.__dict__:
            raise ValueError(f"Method {name} not found in class {cls.__name__}")
        elif not callable(getattr(cls, name)):
            raise ValueError(f"Attribute {name} is not callable but is in methods_to_decorate!")

    for name, attr in cls.__dict__.items():
        if name in methods_to_decorate:
            decorator = methods_to_decorate[name]
            new_attrs[name] = decorator(attr)
        else:
            new_attrs[name] = attr

    new_name = f"{mixin.__name__.replace('Mixin', '')}{cls.__name__}"
    new_class = type(new_name, (cls, mixin), new_attrs)

    return new_class
