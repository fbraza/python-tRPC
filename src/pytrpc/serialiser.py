from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class EncodingError(Exception):
    """Raised when serialization/deserialization fails"""

    pass


def prepare_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Handle edge cases for Arrow compatibility"""
    df_clean = df.copy()

    # Handle mixed-type object columns by converting to string
    for col in df_clean.select_dtypes(include=["object"]).columns:
        # Check if column has mixed types (excluding None/NaN)
        non_null_values: pd.Series[Any] = df_clean[col].dropna()
        if len(non_null_values) > 0:
            types = {type(value) for value in non_null_values}
            if len(types) > 1:
                # Mixed types detected, convert to string
                df_clean[col] = df_clean[col].astype(str)

    # Handle nullable integers for better Arrow compatibility
    for col in df_clean.select_dtypes(include=["int64"]).columns:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].astype("Int64")

    return df_clean


def to_arrow(df: pd.DataFrame, preserve_index: bool) -> bytes:
    """Encode Arrow table as IPC stream"""
    clean = prepare_dataframe_dtypes(df=df)
    table = pa.Table.from_pandas(clean, preserve_index=preserve_index)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def to_parquet(
    df: pd.DataFrame, preserve_index: bool, compression: str = "snappy"
) -> bytes:
    """Encode Arrow table as Parquet"""
    clean = prepare_dataframe_dtypes(df=df)
    table = pa.Table.from_pandas(clean, preserve_index=preserve_index)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression=compression)
    return sink.getvalue().to_pybytes()


def to_pandas(data: bytes, format: str = "arrow") -> pd.DataFrame:
    """Deserialize bytes back to DataFrame"""
    try:
        buf = pa.py_buffer(data)

        if format == "arrow":
            with pa.ipc.open_stream(buf) as reader:
                table = reader.read_all()
            return table.to_pandas()

        elif format == "parquet":
            table = pq.read_table(buf)
            return table.to_pandas()

        else:
            raise ValueError(f"Unsupported format: {format}")

    except Exception as e:
        raise EncodingError(f"Failed to deserialize data: {e}") from e
