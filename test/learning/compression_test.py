#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Tests for compression module."""

import pickle
import zlib
from typing import Any

import numpy as np
import pytest

from p2pfl.learning.compression.lra_strategy import LowRankApproximation
from p2pfl.learning.compression.lzma_strategy import LZMACompressor
from p2pfl.learning.compression.manager import CompressionManager
from p2pfl.learning.compression.quantization_strategy import PTQuantization
from p2pfl.learning.compression.topk_strategy import TopKSparsification
from p2pfl.learning.compression.zlib_strategy import ZlibCompressor

###
# PTQuantization
###


@pytest.fixture
def ptq_compressor():
    """Return an instance of PTQuantization."""
    return PTQuantization()


@pytest.mark.parametrize(
    "shape",
    [
        (10,),  # 1D tensor
        (5, 5),  # 2D tensor
        (3, 4, 5),  # 3D tensor
    ],
)
def test_ptq_shapes(ptq_compressor, shape):
    """Test that quantization preserves shape."""
    original = np.random.randn(*shape).astype(np.float32)
    params = [original]

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy(params, dtype="float16")
    assert quantized[0].shape == original.shape

    # Test dequantization
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    assert dequantized[0].shape == original.shape


@pytest.mark.parametrize(
    "dtype",
    [
        "float16",
        "float32",  # Float types
        "int8",
        "uint8",  # Integer types
    ],
)
def test_ptq_dtypes(ptq_compressor, dtype):
    """Test quantization with different dtypes."""
    original = np.random.randn(10, 10).astype(np.float32)
    params = [original]

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy(params, dtype=dtype)
    assert quantized[0].dtype == np.dtype(dtype)

    # Test dequantization
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    assert dequantized[0].dtype == original.dtype


@pytest.mark.parametrize("scheme", ["symmetric", "asymmetric"])
def test_ptq_schemes(ptq_compressor, scheme):
    """Test different quantization schemes."""
    original = np.random.randn(10, 10).astype(np.float32)
    params = [original]

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy(params, dtype="int8", scheme=scheme)
    assert info["ptq_scheme"] == scheme

    # Test dequantization
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    # Check reconstruction quality (should be close but not identical)
    assert np.allclose(dequantized[0], original, rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    "granularity,channel_axis",
    [
        ("per_tensor", 0),
        ("per_channel", 0),
        ("per_channel", 1),
    ],
)
def test_ptq_granularity(ptq_compressor, granularity, channel_axis):
    """Test per-tensor and per-channel quantization."""
    original = np.random.randn(5, 5).astype(np.float32)
    params = [original]

    # Skip invalid combinations
    if granularity == "per_channel" and channel_axis >= original.ndim:
        pytest.skip("Invalid channel_axis for tensor shape")

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy(params, dtype="int8", granularity=granularity, channel_axis=channel_axis)

    assert info["ptq_granularity"] == granularity

    # Test dequantization
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    assert np.allclose(dequantized[0], original, rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    "value_range",
    [
        (-1.0, 1.0),  # Symmetric around zero
        (0.0, 1.0),  # Positive only
        (-10.0, -5.0),  # Negative only
    ],
)
def test_ptq_value_ranges(ptq_compressor, value_range):
    """Test quantization with different value ranges."""
    min_val, max_val = value_range
    # Create tensor with specific range
    original = np.random.uniform(min_val, max_val, size=(10, 10)).astype(np.float32)
    params = [original]

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy(params, dtype="int8")

    # Test dequantization
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    # Check reconstruction quality
    assert np.allclose(dequantized[0], original, rtol=0.1, atol=0.1)


def test_ptq_multiple_tensors(ptq_compressor):
    """Test quantization of multiple tensors at once."""
    original1 = np.random.randn(5, 5).astype(np.float32)
    original2 = np.random.randn(3, 7).astype(np.float32)
    params = [original1, original2]

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy(params, dtype="int8")
    assert len(quantized) == 2

    # Test dequantization
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    assert len(dequantized) == 2
    assert np.allclose(dequantized[0], original1, rtol=0.1, atol=0.1)
    assert np.allclose(dequantized[1], original2, rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    "dtype,scheme",
    [
        ("int8", "symmetric"),
        ("uint8", "asymmetric"),
    ],
)
def test_ptq_constant_tensors(ptq_compressor, dtype, scheme):
    """Test quantization of tensors with constant values."""
    # Test cases:
    constant_cases = [
        np.zeros((5, 5), dtype=np.float32),  # All zeros
        np.ones((5, 5), dtype=np.float32),  # All ones
        np.full((5, 5), 10.0, dtype=np.float32),  # All 10s
    ]

    for original in constant_cases:
        # Apply quantization
        quantized, info = ptq_compressor.apply_strategy([original], dtype=dtype, scheme=scheme)

        # Test dequantization
        dequantized = ptq_compressor.reverse_strategy(quantized, info)

        # For constant tensors, we should get exactly the same value back
        # (within a small tolerance due to rounding)
        assert np.allclose(dequantized[0], original, rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    "invalid_param",
    [
        {"dtype": "complex64"},  # Invalid dtype
        {"scheme": "invalid_scheme"},  # Invalid scheme
        {"granularity": "invalid_granularity"},  # Invalid granularity
    ],
)
def test_ptq_invalid_params(ptq_compressor, invalid_param):
    """Test that invalid parameters raise appropriate errors."""
    original = np.random.randn(5, 5).astype(np.float32)

    with pytest.raises(ValueError):
        kwargs = {"dtype": "int8"}
        kwargs.update(invalid_param)
        ptq_compressor.apply_strategy([original], **kwargs)


def test_ptq_empty_params(ptq_compressor):
    """Test handling of empty parameter list."""
    with pytest.raises(ValueError):
        ptq_compressor.apply_strategy([], dtype="float16")


def test_ptq_missing_info(ptq_compressor):
    """Test error handling when required info keys are missing."""
    original = np.random.randn(5, 5).astype(np.float32)
    quantized, info = ptq_compressor.apply_strategy([original], dtype="int8")

    # Remove a required key
    bad_info = info.copy()
    del bad_info["ptq_scales"]

    with pytest.raises(ValueError):
        ptq_compressor.reverse_strategy(quantized, bad_info)


def test_ptq_compression_ratio(ptq_compressor):
    """Test compression ratio for float32 to int8 quantization."""
    original = np.random.randn(100, 100).astype(np.float32)

    # Measure original size
    original_bytes = original.nbytes

    # Apply quantization
    quantized, info = ptq_compressor.apply_strategy([original], dtype="int8")
    quantized_bytes = quantized[0].nbytes

    # Calculate compression ratio
    compression_ratio = original_bytes / quantized_bytes

    # Expected ratio for float32 to int8 should be around 4
    assert 3.5 <= compression_ratio <= 4.5


def test_ptq_invalid_dtype_type_error(ptq_compressor):
    """Test that a non-string dtype that causes TypeError is wrapped in ValueError."""
    original = np.random.randn(5, 5).astype(np.float32)
    with pytest.raises(ValueError, match="Invalid dtype"):
        ptq_compressor.apply_strategy([original], dtype=12345)


def test_ptq_unsupported_float_dtype(ptq_compressor):
    """Test that an unsupported float dtype (float128/longdouble) raises ValueError."""
    original = np.random.randn(5, 5).astype(np.float32)
    with pytest.raises(ValueError, match="Unsupported float dtype"):
        ptq_compressor.apply_strategy([original], dtype="longdouble")


def test_ptq_unsupported_int_dtype(ptq_compressor):
    """Test that an unsupported integer dtype (int64) raises ValueError."""
    original = np.random.randn(5, 5).astype(np.float32)
    with pytest.raises(ValueError, match="Unsupported integer dtype"):
        ptq_compressor.apply_strategy([original], dtype="int64")


def test_ptq_reverse_empty_params(ptq_compressor):
    """Test reverse_strategy raises ValueError on empty param list."""
    with pytest.raises(ValueError, match="Empty parameter list"):
        ptq_compressor.reverse_strategy([], {"ptq_original_dtype": np.float32})


def test_ptq_reverse_missing_original_dtype(ptq_compressor):
    """Test reverse_strategy raises ValueError when ptq_original_dtype is missing."""
    dummy = [np.zeros(5, dtype=np.int8)]
    with pytest.raises(ValueError, match="Missing 'ptq_original_dtype'"):
        ptq_compressor.reverse_strategy(dummy, {})


def test_ptq_reverse_invalid_scheme_in_info(ptq_compressor):
    """Test reverse_strategy raises ValueError for invalid scheme stored in additional_info."""
    dummy = [np.zeros(5, dtype=np.int8)]
    info = {
        "ptq_original_dtype": np.float32,
        "ptq_type": "int",
        "ptq_scheme": "invalid",
        "ptq_granularity": "per_tensor",
        "ptq_scales": [1.0],
        "ptq_zero_points": [0],
        "ptq_channel_axis": None,
    }
    with pytest.raises(ValueError, match="Unsupported quantization scheme"):
        ptq_compressor.reverse_strategy(dummy, info)


def test_ptq_reverse_invalid_granularity_in_info(ptq_compressor):
    """Test reverse_strategy raises ValueError for invalid granularity stored in additional_info."""
    dummy = [np.zeros(5, dtype=np.int8)]
    info = {
        "ptq_original_dtype": np.float32,
        "ptq_type": "int",
        "ptq_scheme": "symmetric",
        "ptq_granularity": "invalid",
        "ptq_scales": [1.0],
        "ptq_zero_points": [0],
        "ptq_channel_axis": None,
    }
    with pytest.raises(ValueError, match="Unsupported quantization granularity"):
        ptq_compressor.reverse_strategy(dummy, info)


def test_ptq_reverse_missing_channel_axis_for_per_channel(ptq_compressor):
    """Test reverse_strategy raises ValueError when per_channel but channel_axis is None."""
    dummy = [np.zeros((3, 4), dtype=np.int8)]
    info = {
        "ptq_original_dtype": np.float32,
        "ptq_type": "int",
        "ptq_scheme": "symmetric",
        "ptq_granularity": "per_channel",
        "ptq_scales": [np.array([1.0, 1.0, 1.0], dtype=np.float32)],
        "ptq_zero_points": [np.array([0, 0, 0], dtype=np.int32)],
        "ptq_channel_axis": None,
    }
    with pytest.raises(ValueError, match="Missing 'ptq_channel_axis'"):
        ptq_compressor.reverse_strategy(dummy, info)


def test_ptq_reverse_insufficient_scales(ptq_compressor):
    """Test reverse_strategy raises ValueError when there are fewer scales than params."""
    dummy = [np.zeros(5, dtype=np.int8), np.zeros(5, dtype=np.int8)]
    info = {
        "ptq_original_dtype": np.float32,
        "ptq_type": "int",
        "ptq_scheme": "symmetric",
        "ptq_granularity": "per_tensor",
        "ptq_scales": [1.0],  # Only 1 scale for 2 params
        "ptq_zero_points": [0],
        "ptq_channel_axis": None,
    }
    with pytest.raises(ValueError, match="Not enough scale/zero_point"):
        ptq_compressor.reverse_strategy(dummy, info)


@pytest.mark.parametrize(
    "dtype,expected_range",
    [
        ("int16", (-32768, 32767)),
        ("uint16", (0, 65535)),
        ("int32", (-2147483648, 2147483647)),
    ],
)
def test_ptq_wider_int_dtypes(ptq_compressor, dtype, expected_range):
    """Test quantization with int16, uint16, and int32 dtypes exercises the qmin/qmax branches."""
    original = np.random.randn(5, 5).astype(np.float32)
    target_dtype = np.dtype(dtype)
    # Call internal method directly to exercise dtype-specific quantization range branches
    q_tensor, scale, zero_point = ptq_compressor._quantize_tensor(original, target_dtype, "symmetric")
    assert q_tensor.dtype == target_dtype
    assert q_tensor.shape == original.shape
    assert q_tensor.min() >= expected_range[0]
    assert q_tensor.max() <= expected_range[1]
    assert scale > 0


def test_ptq_empty_tensor(ptq_compressor):
    """Test quantization of an empty tensor via _quantize_tensor."""
    empty = np.array([], dtype=np.float32).reshape(0, 5)
    params = [np.random.randn(3, 5).astype(np.float32), empty]
    quantized, info = ptq_compressor.apply_strategy(params, dtype="int8")
    assert quantized[1].size == 0
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    assert dequantized[1].size == 0


def test_ptq_per_channel_scalar_tensor_raises(ptq_compressor):
    """Test that per-channel quantization on a scalar raises ValueError."""
    scalar = np.float32(5.0)
    with pytest.raises(ValueError, match="scalar tensor"):
        ptq_compressor.apply_strategy([scalar], dtype="int8", granularity="per_channel")


def test_ptq_per_channel_invalid_axis(ptq_compressor):
    """Test that per-channel with invalid channel_axis raises ValueError."""
    tensor = np.random.randn(3, 4).astype(np.float32)
    with pytest.raises(ValueError, match="Invalid channel_axis"):
        ptq_compressor.apply_strategy([tensor], dtype="int8", granularity="per_channel", channel_axis=5)


def test_ptq_per_channel_1d_fallback(ptq_compressor):
    """Test that per-channel quantization on a 1D tensor falls back to per-tensor."""
    tensor_1d = np.random.randn(10).astype(np.float32)
    q_tensor, scales, zero_points = ptq_compressor._quantize_per_channel(tensor_1d, np.dtype("int8"), "symmetric", 0)
    assert q_tensor.shape == tensor_1d.shape
    assert q_tensor.dtype == np.int8
    assert scales.shape == (1,)
    assert zero_points.shape == (1,)


@pytest.mark.parametrize("dtype", ["uint8", "int16", "uint16", "int32"])
def test_ptq_per_channel_various_int_dtypes(ptq_compressor, dtype):
    """Test per-channel quantization with wider integer dtypes to cover qmin/qmax branches."""
    tensor = np.random.randn(4, 6).astype(np.float32)
    target_dtype = np.dtype(dtype)
    # Call internal method directly to exercise the dtype-specific qmin/qmax branches
    q_tensor, scales, zero_points = ptq_compressor._quantize_per_channel(tensor, target_dtype, "symmetric", 0)
    assert q_tensor.dtype == target_dtype
    assert q_tensor.shape == tensor.shape
    assert scales.shape == (4,)
    assert zero_points.shape == (4,)


def test_ptq_per_channel_symmetric_zero_channel(ptq_compressor):
    """Test per-channel symmetric quantization where one channel is all zeros."""
    tensor = np.random.randn(3, 4).astype(np.float32)
    tensor[1, :] = 0.0  # Make channel 1 all zeros
    quantized, info = ptq_compressor.apply_strategy([tensor], dtype="int8", scheme="symmetric", granularity="per_channel", channel_axis=0)
    dequantized = ptq_compressor.reverse_strategy(quantized, info)
    # The zero channel should remain zero
    assert np.allclose(dequantized[0][1, :], 0.0, atol=1e-6)


def test_ptq_per_channel_asymmetric(ptq_compressor):
    """Test per-channel asymmetric quantization covering the asymmetric branch in _quantize_per_channel."""
    tensor = np.random.randn(4, 5).astype(np.float32)
    q_tensor, scales, zero_points = ptq_compressor._quantize_per_channel(tensor, np.dtype("int8"), "asymmetric", 0)
    assert q_tensor.dtype == np.int8
    assert q_tensor.shape == tensor.shape
    assert scales.shape == (4,)
    assert zero_points.shape == (4,)
    assert np.all(scales > 0)  # All scales should be positive for non-constant channels


def test_ptq_per_channel_asymmetric_constant_zero_channel(ptq_compressor):
    """Test per-channel asymmetric where a channel has all zeros (tmin==tmax==0)."""
    tensor = np.random.randn(3, 4).astype(np.float32)
    tensor[0, :] = 0.0
    q_tensor, scales, zero_points = ptq_compressor._quantize_per_channel(tensor, np.dtype("int8"), "asymmetric", 0)
    # The zero channel should be quantized to all zeros with scale=1.0
    assert np.all(q_tensor[0, :] == 0)
    assert scales[0] == 1.0
    assert zero_points[0] == 0


def test_ptq_per_channel_asymmetric_constant_nonzero_channel(ptq_compressor):
    """Test per-channel asymmetric quantization where a channel is constant nonzero."""
    tensor = np.random.randn(3, 4).astype(np.float32)
    tensor[2, :] = 7.0  # Constant nonzero channel
    q_tensor, scales, zero_points = ptq_compressor._quantize_per_channel(tensor, np.dtype("int8"), "asymmetric", 0)
    assert q_tensor.dtype == np.int8
    assert q_tensor.shape == tensor.shape
    # The constant channel should be mapped to mid_q
    mid_q = (-128 + 127) // 2  # = -1 for int8
    assert np.all(q_tensor[2, :] == mid_q)


def test_ptq_dequantize_tensor_invalid_scale_type(ptq_compressor):
    """Test _dequantize_tensor raises ValueError for non-numeric scale."""
    tensor = np.zeros(5, dtype=np.int8)
    with pytest.raises(ValueError, match="Invalid scale factor"):
        ptq_compressor._dequantize_tensor(tensor, "bad", 0, np.float32)


def test_ptq_dequantize_tensor_negative_scale(ptq_compressor):
    """Test _dequantize_tensor raises ValueError for negative scale."""
    tensor = np.zeros(5, dtype=np.int8)
    with pytest.raises(ValueError, match="Scale must be positive"):
        ptq_compressor._dequantize_tensor(tensor, -1.0, 0, np.float32)


def test_ptq_dequantize_tensor_invalid_zero_point_type(ptq_compressor):
    """Test _dequantize_tensor raises ValueError for non-integer zero_point."""
    tensor = np.zeros(5, dtype=np.int8)
    with pytest.raises(ValueError, match="Zero point must be an integer"):
        ptq_compressor._dequantize_tensor(tensor, 1.0, 0.5, np.float32)


def test_ptq_dequantize_tensor_empty(ptq_compressor):
    """Test _dequantize_tensor handles empty tensor."""
    empty = np.array([], dtype=np.int8)
    result = ptq_compressor._dequantize_tensor(empty, 1.0, 0, np.float32)
    assert result.size == 0
    assert result.dtype == np.float32


def test_ptq_dequantize_per_channel_invalid_scales_type(ptq_compressor):
    """Test _dequantize_per_channel raises ValueError when scales is not ndarray."""
    tensor = np.zeros((3, 4), dtype=np.int8)
    with pytest.raises(ValueError, match="Scales must be a numpy array"):
        ptq_compressor._dequantize_per_channel(tensor, [1.0, 1.0, 1.0], np.array([0, 0, 0], dtype=np.int32), 0, np.float32)


def test_ptq_dequantize_per_channel_invalid_zero_points_type(ptq_compressor):
    """Test _dequantize_per_channel raises ValueError when zero_points is not ndarray."""
    tensor = np.zeros((3, 4), dtype=np.int8)
    with pytest.raises(ValueError, match="Zero points must be a numpy array"):
        ptq_compressor._dequantize_per_channel(tensor, np.array([1.0, 1.0, 1.0], dtype=np.float32), [0, 0, 0], 0, np.float32)


def test_ptq_dequantize_per_channel_non_1d_scales(ptq_compressor):
    """Test _dequantize_per_channel raises ValueError when scales is not 1D."""
    tensor = np.zeros((3, 4), dtype=np.int8)
    with pytest.raises(ValueError, match="Scales must be a 1D array"):
        ptq_compressor._dequantize_per_channel(tensor, np.ones((3, 1), dtype=np.float32), np.zeros(3, dtype=np.int32), 0, np.float32)


def test_ptq_dequantize_per_channel_non_1d_zero_points(ptq_compressor):
    """Test _dequantize_per_channel raises ValueError when zero_points is not 1D."""
    tensor = np.zeros((3, 4), dtype=np.int8)
    with pytest.raises(ValueError, match="Zero points must be a 1D array"):
        ptq_compressor._dequantize_per_channel(tensor, np.ones(3, dtype=np.float32), np.zeros((3, 1), dtype=np.int32), 0, np.float32)


def test_ptq_dequantize_per_channel_mismatched_lengths(ptq_compressor):
    """Test _dequantize_per_channel raises ValueError when scales/zero_points length mismatch."""
    tensor = np.zeros((3, 4), dtype=np.int8)
    with pytest.raises(ValueError, match="must match number of zero points"):
        ptq_compressor._dequantize_per_channel(tensor, np.ones(3, dtype=np.float32), np.zeros(2, dtype=np.int32), 0, np.float32)


def test_ptq_dequantize_per_channel_empty_tensor(ptq_compressor):
    """Test _dequantize_per_channel handles empty tensor."""
    empty = np.array([], dtype=np.int8).reshape(0, 4)
    result = ptq_compressor._dequantize_per_channel(empty, np.array([], dtype=np.float32), np.array([], dtype=np.int32), 0, np.float32)
    assert result.size == 0


def test_ptq_dequantize_per_channel_single_scale_fallback(ptq_compressor):
    """Test _dequantize_per_channel falls back to per-tensor when scales has single element."""
    tensor = np.array([10, 20, 30], dtype=np.int8)
    # Use float64 scales since np.float64 passes isinstance(x, float) but np.float32 does not
    result = ptq_compressor._dequantize_per_channel(tensor, np.array([0.5], dtype=np.float64), np.array([0], dtype=np.int32), 0, np.float32)
    expected = tensor.astype(np.float32) * 0.5
    np.testing.assert_allclose(result, expected)


def test_ptq_dequantize_per_channel_scalar_tensor_raises(ptq_compressor):
    """Test _dequantize_per_channel raises ValueError on scalar tensor with multiple scales."""
    scalar = np.int8(5)
    with pytest.raises(ValueError, match="scalar tensor"):
        ptq_compressor._dequantize_per_channel(
            scalar, np.array([1.0, 2.0], dtype=np.float32), np.array([0, 0], dtype=np.int32), 0, np.float32
        )


def test_ptq_dequantize_per_channel_invalid_axis(ptq_compressor):
    """Test _dequantize_per_channel raises ValueError for invalid channel_axis."""
    tensor = np.zeros((3, 4), dtype=np.int8)
    with pytest.raises(ValueError, match="Invalid channel_axis"):
        ptq_compressor._dequantize_per_channel(tensor, np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.int32), 5, np.float32)


def test_ptq_dequantize_per_channel_channel_count_mismatch(ptq_compressor):
    """Test _dequantize_per_channel raises ValueError when channels != len(scales)."""
    tensor = np.zeros((3, 4), dtype=np.int8)
    with pytest.raises(ValueError, match="Number of channels"):
        ptq_compressor._dequantize_per_channel(tensor, np.ones(5, dtype=np.float32), np.zeros(5, dtype=np.int32), 0, np.float32)


def test_ptq_dequantize_per_channel_invalid_scale_value(ptq_compressor):
    """Test _dequantize_per_channel raises ValueError for non-finite scale in a channel."""
    tensor = np.zeros((3, 4), dtype=np.int8)
    scales = np.array([1.0, np.inf, 1.0], dtype=np.float32)
    zero_points = np.zeros(3, dtype=np.int32)
    with pytest.raises(ValueError, match="Invalid scale factor at index"):
        ptq_compressor._dequantize_per_channel(tensor, scales, zero_points, 0, np.float32)


def test_ptq_dequantize_per_channel_negative_scale_value(ptq_compressor):
    """Test _dequantize_per_channel raises ValueError for negative scale in a channel."""
    tensor = np.zeros((3, 4), dtype=np.int8)
    scales = np.array([1.0, -0.5, 1.0], dtype=np.float32)
    zero_points = np.zeros(3, dtype=np.int32)
    with pytest.raises(ValueError, match="Scale must be positive"):
        ptq_compressor._dequantize_per_channel(tensor, scales, zero_points, 0, np.float32)


def test_ptq_dequantize_per_channel_invalid_zero_point_value(ptq_compressor):
    """Test _dequantize_per_channel raises ValueError for non-integer zero_point in a channel."""
    tensor = np.zeros((3, 4), dtype=np.int8)
    scales = np.ones(3, dtype=np.float32)
    zero_points = np.array([0.0, 0.5, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="Zero point must be an integer"):
        ptq_compressor._dequantize_per_channel(tensor, scales, zero_points, 0, np.float32)


def test_ptq_per_channel_asymmetric_scale_zero_guard(ptq_compressor):
    """Test per-channel asymmetric quantization where scale rounds to zero due to subnormal values."""
    # Craft a channel where tmin != tmax but (tmax - tmin) / (qmax - qmin) == 0.0
    eps = np.finfo(np.float32).smallest_subnormal
    tensor = np.array([[0.0, eps, 0.0, eps], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    q_tensor, scales, zero_points = ptq_compressor._quantize_per_channel(tensor, np.dtype("int8"), "asymmetric", 0)
    # The first channel has near-zero range, scale should be set to 1.0 (the guard value)
    assert scales[0] == 1.0
    assert q_tensor.dtype == np.int8


def test_ptq_quantize_tensor_unsupported_dtype(ptq_compressor):
    """Test _quantize_tensor raises ValueError for unsupported integer dtype directly."""
    tensor = np.random.randn(5).astype(np.float32)
    with pytest.raises(ValueError, match="Unsupported integer dtype"):
        ptq_compressor._quantize_tensor(tensor, np.dtype("int64"), "symmetric")


def test_ptq_quantize_per_channel_unsupported_dtype(ptq_compressor):
    """Test _quantize_per_channel raises ValueError for unsupported integer dtype directly."""
    tensor = np.random.randn(3, 4).astype(np.float32)
    with pytest.raises(ValueError, match="Unsupported integer dtype"):
        ptq_compressor._quantize_per_channel(tensor, np.dtype("int64"), "symmetric", 0)


def test_ptq_quantize_per_channel_empty_2d_tensor(ptq_compressor):
    """Test _quantize_per_channel handles empty 2D tensor (num_channels=0)."""
    empty = np.array([], dtype=np.float32).reshape(0, 4)
    q_tensor, scales, zero_points = ptq_compressor._quantize_per_channel(empty, np.dtype("int8"), "symmetric", 0)
    assert q_tensor.size == 0
    assert scales.size == 0
    assert zero_points.size == 0


def test_ptq_dequantize_per_channel_asymmetric_path(ptq_compressor):
    """Test _dequantize_per_channel with nonzero zero_points exercises the asymmetric branch."""
    # Create a quantized tensor and manually set up scales/zero_points with nonzero zero_point
    tensor = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.int8)
    scales = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    zero_points = np.array([5, 10, 15], dtype=np.int32)
    result = ptq_compressor._dequantize_per_channel(tensor, scales, zero_points, 0, np.float32)
    # Channel 0: (x - 5) * 0.1, Channel 1: (x - 10) * 0.2, Channel 2: (x - 15) * 0.3
    expected_ch0 = (tensor[0].astype(np.float32) - 5) * 0.1
    expected_ch1 = (tensor[1].astype(np.float32) - 10) * 0.2
    expected_ch2 = (tensor[2].astype(np.float32) - 15) * 0.3
    np.testing.assert_allclose(result[0], expected_ch0, rtol=1e-5)
    np.testing.assert_allclose(result[1], expected_ch1, rtol=1e-5)
    np.testing.assert_allclose(result[2], expected_ch2, rtol=1e-5)


###
# TopK
###


@pytest.mark.parametrize("sample_k", [0.1, 0.5, 1.0])
def test_topk_sparsification(sample_k: float):
    """
    Test the TopK sparsification strategy.

    Args:
        dummy_payload: Payload to test.
        sample_k : k to test.

    """
    technique = TopKSparsification()
    original_params = [np.random.randn(10, 10) for i in range(3)]
    total_original_size = sum(layer.size for layer in original_params)

    compressed_parameters, technique_params = technique.apply_strategy(original_params, k=sample_k)
    total_compressed_size = sum(layer.size for layer in compressed_parameters)
    assert "topk_sparse_metadata" in technique_params, "Missing metadata on compressed model"
    assert total_compressed_size <= total_original_size, "compression resulted in more parameters than the original model"
    if sample_k != 1.0:
        assert total_compressed_size < total_original_size, "compression did not remove any parameters"

    decompressed_parameters = technique.reverse_strategy(compressed_parameters, technique_params)
    total_decompressed_size = sum(layer.size for layer in decompressed_parameters)
    assert total_decompressed_size == total_original_size
    for orig, decomp in zip(original_params, decompressed_parameters, strict=True):
        assert orig.shape == decomp.shape, "Decompressed shape does not match original"


###
# LoRa
###


@pytest.mark.parametrize("threshold", [0.5, 0.7])
def test_lowrank(threshold: float):
    """
    Test LowRank compression algorithm.

    Args:
        dummy_payload: Payload to test.
        threshold: Percentage between 0 and 1 of the energy to preserve.

    """
    technique = LowRankApproximation()
    original_params = [np.random.randn(10, 10) for i in range(3)]
    compressed_parameters, technique_params = technique.apply_strategy(original_params, threshold=threshold)
    assert "lowrank_compressed_state" in technique_params, "Missing compression metadata"
    assert sum(layer.size for layer in compressed_parameters) < sum(
        layer.size for layer in original_params
    ), "compression resulted in more parameters than the original model"

    decompressed_parameters = technique.reverse_strategy(compressed_parameters, technique_params)
    total_original = sum(layer.size for layer in original_params)
    total_decompressed = sum(layer.size for layer in decompressed_parameters)
    assert total_original == total_decompressed, "Number of elements not matching after reverse strategy."

    tol = 0.05
    for orig, decomp in zip(original_params, decompressed_parameters, strict=True):
        if orig.ndim == 2:
            # relative error to compressed layers, expected ~= 1 - threshold
            energy_total = np.sum(np.linalg.svd(orig, full_matrices=False)[1] ** 2)
            error = np.linalg.norm(orig - decomp, ord="fro") ** 2 / energy_total
            assert error <= (1 - threshold + tol), f"Relative error {error:.3f} exceeds allowed limit for threshold {threshold}"
        else:
            np.testing.assert_array_equal(orig, decomp, err_msg="Non-compressed layer has changed.")


###
# ZLIB
###


@pytest.mark.parametrize("level", [1, 5])
def test_zlib(level: int):
    """
    Test Zlib compression algorithm.

    Args:
        level: zlib level of compression.

    """
    technique = ZlibCompressor()
    original_bytes = pickle.dumps("LUIS PERUANO UUUUUUUUUUUUUUUUUUUUUUUUUU!!!!!!! Y HECTOR NO HACE NADA")
    compressed_bytes = technique.apply_strategy(original_bytes, level=level)
    assert len(original_bytes) > len(compressed_bytes), "compression resulted in more bytes than the original model"
    decompressed_bytes = technique.reverse_strategy(compressed_bytes)
    assert decompressed_bytes == original_bytes


@pytest.mark.parametrize("preset", [1, 5, 9])
def test_lzma(preset: int):
    """
    Test LZMA compression algorithm.

    Args:
        preset: LZMA level of compression.

    """
    technique = LZMACompressor()
    original_bytes = pickle.dumps("ABC " * 1000)
    compressed_bytes = technique.apply_strategy(original_bytes, preset=preset)
    assert len(original_bytes) > len(compressed_bytes), "compression resulted in more bytes than the original model"
    decompressed_bytes = technique.reverse_strategy(compressed_bytes)
    assert decompressed_bytes == original_bytes


###
# Manager tests
###


@pytest.fixture
def compression_manager() -> CompressionManager:
    """Fixture to create a new compression manager instance."""
    return CompressionManager()


def test_manager_multiple_byte_compressors(compression_manager: CompressionManager):
    """Test that only one byte compressor is allowed."""
    techniques = {"zlib": {"level": 5}, "lzma": {"preset": 5}}
    with pytest.raises(ValueError):
        _ = compression_manager.apply([np.random.randn(10, 10) for i in range(3)], {"dummy": "info"}, techniques)


def test_manager_unknown_strategy(compression_manager: CompressionManager):
    """Test that an unknown strategy raises an error."""
    techniques: dict[str, dict[str, Any]] = {"unknown": {}}
    with pytest.raises(ValueError):
        _ = compression_manager.apply([np.random.randn(10, 10) for i in range(3)], {"dummy": "info"}, techniques)


def test_manager_no_techniques(compression_manager: CompressionManager):
    """Test that an empty dictionary of techniques raises an error."""
    original_params = [np.random.randn(10, 10) for i in range(3)]
    original_add_info: dict[str, dict[str, Any]] = {}
    compressed_data = compression_manager.apply(original_params, original_add_info, {})
    deserialized_data = pickle.loads(compressed_data)
    assert deserialized_data["byte_compressor"] is None
    assert np.array_equal(pickle.loads(deserialized_data["bytes"])["params"], original_params)
    assert pickle.loads(deserialized_data["bytes"])["additional_info"]["applied_techniques"] == []
    decompressed_params, decompressed_add_info = compression_manager.reverse(compressed_data)
    assert np.array_equal(original_params, decompressed_params)
    assert decompressed_add_info == original_add_info


def test_manager_only_compressor(compression_manager: CompressionManager):
    """Test only compressor (loseless)."""
    original_params = [np.random.randn(10, 10) for i in range(3)]

    techniques: dict[str, dict[str, Any]] = {"zlib": {"level": 5}}
    additional_info: dict[str, dict[str, Any]] = {}
    compressed_data = compression_manager.apply(original_params, additional_info, techniques)
    deserialized_data = pickle.loads(compressed_data)
    assert deserialized_data["byte_compressor"] == "zlib"

    decompressed_bytes = zlib.decompress(deserialized_data["bytes"])
    assert "params" in pickle.loads(decompressed_bytes)
    assert "additional_info" in pickle.loads(decompressed_bytes)
    assert "applied_techniques" in pickle.loads(decompressed_bytes)["additional_info"]
    assert len(pickle.loads(decompressed_bytes)["additional_info"]["applied_techniques"]) == 0
    decompressed_params, _ = compression_manager.reverse(compressed_data)
    assert np.allclose(original_params[0], decompressed_params[0], atol=1e-2)


def test_manager_multiple_techniques(compression_manager: CompressionManager):
    """Test the manager with multiple techniques."""
    original_params = [np.random.randn(10, 10) for i in range(3)]
    techniques: dict[str, dict[str, Any]] = {
        "topk": {"k": 0.5},
        "zlib": {"level": 5},
        "low_rank": {"threshold": 0.7},
    }
    additional_info: dict[str, dict[str, Any]] = {}
    compressed_data = compression_manager.apply(original_params, additional_info, techniques)
    deserialized_data = pickle.loads(compressed_data)
    assert deserialized_data["byte_compressor"] == "zlib"

    decompressed_bytes = zlib.decompress(deserialized_data["bytes"])
    assert "params" in pickle.loads(decompressed_bytes)
    assert "additional_info" in pickle.loads(decompressed_bytes)
    assert "applied_techniques" in pickle.loads(decompressed_bytes)["additional_info"]
    assert len(pickle.loads(decompressed_bytes)["additional_info"]["applied_techniques"]) == 2
    assert pickle.loads(decompressed_bytes)["additional_info"]["applied_techniques"][0][0] == "topk"
    assert pickle.loads(decompressed_bytes)["additional_info"]["applied_techniques"][1][0] == "low_rank"

    decompressed_params, decompressed_info = compression_manager.reverse(compressed_data)
    for orig, decomp in zip(original_params, decompressed_params, strict=True):
        assert orig.shape == decomp.shape
    assert len(decompressed_params) == len(original_params)


def test_additional_info_preservation(compression_manager: CompressionManager):
    """Test that techniques don't remove additional info from other processes."""
    original_params = [np.random.randn(10, 10)]
    additional_info = {"test_key": "test_value"}
    registry = compression_manager.get_registry()

    assert registry, "The compression registry should not be empty."
    for technique_name in registry:
        try:
            compressed_data = compression_manager.apply(original_params, additional_info, {technique_name: {}})
        except ImportError:
            continue
        _, decompressed_info = compression_manager.reverse(compressed_data)
        assert decompressed_info == additional_info, f"Additional info lost for technique '{technique_name}'"
