from __future__ import annotations

from collections.abc import Hashable
from typing import Union

import numpy as np
import torch


class TensorableMixin:
    Tensorable_T = Union[np.ndarray, list[float], tuple["Tensorable_T"], dict[Hashable, "Tensorable_T"]]
    Tensor_T = Union[torch.Tensor, tuple["Tensor_T"], dict[Hashable, "Tensor_T"]]

    def __init__(self, *args, **kwargs):
        self.do_cuda = kwargs.get("do_cuda", torch.cuda.is_available())

    def _cuda(self, T: torch.Tensor, do_cuda: bool | None = None):
        if do_cuda is None:
            do_cuda = self.do_cuda if hasattr(self, "do_cuda") else torch.cuda.is_available()

        return T.cuda() if do_cuda else T

    def _from_numpy(self, obj: np.ndarray) -> torch.Tensor:
        # I keep getting errors about "RuntimeError: expected scalar type Float but found Double"
        if obj.dtype == np.float64:
            obj = obj.astype(np.float32)
        return self._cuda(torch.from_numpy(obj))

    def _nested_to_tensor(self, obj: TensorableMixin.Tensorable_T) -> TensorableMixin.Tensor_T:
        if isinstance(obj, np.ndarray):
            return self._from_numpy(obj)
        elif isinstance(obj, list):
            return self._from_numpy(np.array(obj))
        elif isinstance(obj, dict):
            return {k: self._nested_to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return tuple(self._nested_to_tensor(e) for e in obj)

        raise ValueError(f"Don't know how to convert {type(obj)} object {obj} to tensor!")
