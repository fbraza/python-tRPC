import numpy as np
import pandas as pd
import pytest

from pytrpc import serialiser


def test_dataframe_roundtrip():
    """Test DataFrame can be serialized and deserialized without loss"""
    original_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    pa_serialized = serialiser.to_arrow(original_df, False)
    pa_deserialized = serialiser.to_pandas(pa_serialized, format="arrow")

    pd.testing.assert_frame_equal(original_df, pa_deserialized)

    pq_serialized = serialiser.to_parquet(original_df, False)
    pq_deserialized = serialiser.to_pandas(pq_serialized, format="parquet")

    pd.testing.assert_frame_equal(original_df, pq_deserialized)


def test_large_dataframe():
    """Test performance with large DataFrames"""
    # Create a moderately large DataFrame (not too large for CI)
    large_df = pd.DataFrame(
        {
            "col1": range(10000),
            "col2": [f"string_{i}" for i in range(10000)],
            "col3": np.random.randn(10000),
        }
    )

    pa_serialized = serialiser.to_arrow(large_df, False)
    pa_deserialized = serialiser.to_pandas(pa_serialized, format="arrow")

    pd.testing.assert_frame_equal(large_df, pa_deserialized)

    pq_serialized = serialiser.to_parquet(large_df, False)
    pq_deserialized = serialiser.to_pandas(pq_serialized, format="parquet")

    pd.testing.assert_frame_equal(large_df, pq_deserialized)


def test_different_dtypes():
    """Test various pandas dtypes are preserved"""
    df = pd.DataFrame(
        {
            "integers": [1, 2, 3, 4],
            "floats": [1.1, 2.2, 3.3, 4.4],
            "strings": ["a", "b", "c", "d"],
            "booleans": [True, False, True, False],
            "dates": pd.date_range("2023-01-01", periods=4),
        }
    )

    pa_serialized = serialiser.to_arrow(df, False)
    pa_deserialized = serialiser.to_pandas(pa_serialized, format="arrow")

    pd.testing.assert_frame_equal(df, pa_deserialized)

    # Check dtypes are preserved
    for col in df.columns:
        assert df[col].dtype == pa_deserialized[col].dtype, (
            f"Dtype mismatch for column {col}"
        )


def test_nullable_types():
    """Test DataFrames with null values are handled correctly"""
    df = pd.DataFrame(
        {
            "integers_with_na": [1, 2, None, 4],
            "floats_with_na": [1.1, None, 3.3, 4.4],
            "strings_with_na": ["a", None, "c", "d"],
            "booleans_with_na": [True, None, True, False],
        }
    )

    pa_serialized = serialiser.to_arrow(df, False)
    pa_deserialized = serialiser.to_pandas(pa_serialized, format="arrow")

    pd.testing.assert_frame_equal(df, pa_deserialized)

    pq_serialized = serialiser.to_parquet(df, False)
    pq_deserialized = serialiser.to_pandas(pq_serialized, format="parquet")

    pd.testing.assert_frame_equal(df, pq_deserialized)


def test_categorical_data():
    """Test categorical data is preserved"""
    df = pd.DataFrame(
        {
            "categories": pd.Categorical(["cat1", "cat2", "cat1", "cat3"]),
            "ordered_cats": pd.Categorical(
                ["low", "medium", "high", "medium"],
                categories=["low", "medium", "high"],
                ordered=True,
            ),
        }
    )

    pa_serialized = serialiser.to_arrow(df, False)
    pa_deserialized = serialiser.to_pandas(pa_serialized, format="arrow")

    pd.testing.assert_frame_equal(df, pa_deserialized)

    pq_serialized = serialiser.to_parquet(df, False)
    pq_deserialized = serialiser.to_pandas(pq_serialized, format="parquet")

    pd.testing.assert_frame_equal(df, pq_deserialized)


def test_invalid_format():
    """Test error handling for invalid format"""
    df = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(serialiser.EncodingError):
        compressed = serialiser.to_parquet(
            df, preserve_index=False, compression="snappy"
        )
        serialiser.to_pandas(compressed, format="invalid format")


def test_mixed_type_columns():
    """Test handling of mixed-type object columns"""
    df = pd.DataFrame({"mixed": [1, "string", 3.14, True, None]})

    serialized = serialiser.to_arrow(df, False)
    deserialized = serialiser.to_pandas(serialized, format="arrow")

    # Should have same number of rows
    assert len(df) == len(deserialized)
    assert list(df.columns) == list(deserialized.columns)
