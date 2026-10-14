"""Read PyTorch ZIP metadata without PyTorch, tensor loading, or arbitrary globals.

Tensor/storage and NumPy RNG objects are represented by metadata-only records.
This does not validate model execution or replace a strict PyTorch model reload.
"""
from collections import OrderedDict
import codecs
import io
import pickle
from pathlib import Path
import zipfile
import numpy as np


class Opaque:
    """Unused RNG object; retain state without executing NumPy reconstruction."""
    def __init__(self, *args):
        self.args = args

    def __setstate__(self, state):
        self.state = state


class FloatStorage:
    pass


class ByteStorage:
    pass


def rebuild_tensor(storage, offset, size, stride, requires_grad=False, hooks=None, *args):
    return dict(tensor_metadata=True, storage=storage, offset=offset,
                shape=list(size), stride=list(stride), requires_grad=requires_grad)


def rebuild_numpy_scalar(dtype, data):
    """Decode only bounded numeric scalar bytes (e.g. logged learning rates)."""
    if not isinstance(dtype, np.dtype) or dtype.kind not in 'biuf' or len(data) != dtype.itemsize or len(data)>16:
        raise pickle.UnpicklingError('Unsupported NumPy scalar metadata')
    return np.frombuffer(data,dtype=dtype,count=1)[0].item()


class MetadataUnpickler(pickle.Unpickler):
    """Whitelist observed constructors; unknown checkpoint types fail closed."""
    allowed = {
        ('collections', 'OrderedDict'): OrderedDict,
        ('_codecs', 'encode'): codecs.encode,
        ('torch', 'FloatStorage'): FloatStorage,
        ('torch', 'ByteStorage'): ByteStorage,
        ('torch._utils', '_rebuild_tensor_v2'): rebuild_tensor,
        ('numpy', 'dtype'): np.dtype,
        ('numpy', 'ndarray'): Opaque,
        ('numpy._core.multiarray', '_reconstruct'): Opaque,
        ('numpy._core.multiarray', 'scalar'): rebuild_numpy_scalar,
        ('numpy.core.multiarray', '_reconstruct'): Opaque,
        ('numpy.core.multiarray', 'scalar'): rebuild_numpy_scalar,
    }

    def find_class(self, module, name):
        if (module, name) not in self.allowed:
            raise pickle.UnpicklingError(f'Unsupported metadata global: {module}.{name}')
        return self.allowed[module, name]

    def persistent_load(self, value):
        if not isinstance(value, tuple) or len(value) != 5 or value[0] != 'storage':
            raise pickle.UnpicklingError('Unsupported persistent metadata')
        return dict(dtype=value[1].__name__, key=value[2], location=value[3], elements=value[4])


def read_checkpoint_metadata(path):
    with zipfile.ZipFile(Path(path)) as archive:
        members = [i for i in archive.infolist() if i.filename.endswith('/data.pkl')]
        if len(members) != 1 or members[0].file_size > 16*1024*1024:
            raise ValueError('No unique, bounded checkpoint metadata payload')
        return MetadataUnpickler(io.BytesIO(archive.read(members[0]))).load()
