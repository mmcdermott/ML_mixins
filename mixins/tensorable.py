from __future__ import annotations

try:
    import torch, numpy as np
    from typing import Dict, Hashable, List, Optional, Tuple, Union

    class TensorableMixin():
        Tensorable_T = Union[np.ndarray, List[float], Tuple['Tensorable_T'], Dict[Hashable, 'Tensorable_T']]
        Tensor_T = Union[torch.Tensor, Tuple['Tensor_T'], Dict[Hashable, 'Tensor_T']]

        def __init__(self, *args, **kwargs):
            self.do_cuda = kwargs.get('do_cuda', torch.cuda.is_available)

        def _cuda(self, T: torch.Tensor, do_cuda: Optional[bool] = None):
            if do_cuda is None:
                do_cuda = self.do_cuda if hasattr(self, 'do_cuda') else torch.cuda.is_available

            return T.cuda() if do_cuda else T

        def _from_numpy(self, obj: np.ndarray) -> torch.Tensor:
            # I keep getting errors about "RuntimeError: expected scalar type Float but found Double"
            if obj.dtype == np.float64: obj = obj.astype(np.float32)
            return self._cuda(torch.from_numpy(obj))

        def _nested_to_tensor(self, obj: TensorableMixin.Tensorable_T) -> TensorableMixin.Tensor_T:
            if isinstance(obj, np.ndarray): return self._from_numpy(obj)
            elif isinstance(obj, list): return self._from_numpy(np.array(obj))
            elif isinstance(obj, dict): return {k: self._nested_to_tensor(v) for k, v in obj.items()}
            elif isinstance(obj, tuple): return tuple((self._nested_to_tensor(e) for e in obj))

            raise ValueError(f"Don't know how to convert {type(obj)} object {obj} to tensor!")

except ImportError as e: pass
