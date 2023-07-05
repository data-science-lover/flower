# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Parameter conversion."""


from io import BytesIO
from typing import cast

import numpy as np

from .typing import NDArray, NDArrays, Parameters
import tenseal as ts
import pickle
import torch


def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    # tensors = [ndarray_to_bytes(ndarray) for ndarray in ndarrays]
    # return Parameters(tensors=tensors, tensor_type="numpy.ndarray")
    return ndarrays_to_parameters_custom(ndarrays)


def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""
    # return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]
    secret_path = "./secret.pkl"
    with open(secret_path, 'rb') as f:
        query = pickle.load(f)

    context_client = ts.context_from(query["contexte"])
    return parameters_to_ndarrays_custom(parameters, context_client=context_client)


def ndarray_to_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    """
    bytes_io = BytesIO()
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(bytes_io, ndarray, allow_pickle=False)  # type: ignore
    return bytes_io.getvalue()
    """
    return ndarray_to_bytes_custom(ndarray)


def bytes_to_ndarray(tensor: bytes, context_client) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)

    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)  # type: ignore

    return cast(NDArray, ndarray_deserialized)


# Redefine the ndarray_to_bytes function (defined in Flower)
def ndarray_to_bytes_custom(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    if type(ndarray) == ts.tensors.CKKSTensor:
        return ndarray.serialize()  # ndarray = ndarray.serialize()
        # ndarray = ndarray.serialize()

    bytes_io = BytesIO()
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(bytes_io, ndarray.cpu().detach().numpy() if type(ndarray) == torch.Tensor else ndarray, allow_pickle=False)  # type: ignore
    return bytes_io.getvalue()


# Redefine the bytes_to_ndarray function (defined in Flower)
def bytes_to_ndarray_custom(tensor: bytes, context_client) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""

    try:
        ndarray_deserialized = ts.ckks_tensor_from(context_client, tensor)
    except:
        bytes_io = BytesIO(tensor)

        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
        ndarray_deserialized = np.load(bytes_io, allow_pickle=False)  # type: ignore

    return cast(NDArray, ndarray_deserialized)


# redefine the ndarrays_to_parameters function (defined in Flower)
def ndarrays_to_parameters_custom(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    tensors = [ndarray_to_bytes_custom(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


# redefine the parameters_to_ndarrays function (defined in Flower)
def parameters_to_ndarrays_custom(parameters: Parameters, context_client) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""
    return [bytes_to_ndarray_custom(tensor, context_client) for tensor in parameters.tensors]
