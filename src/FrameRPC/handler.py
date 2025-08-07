from router import DataFrameRouter
from serialiser import to_arrow, to_pandas, to_parquet


def request(
    router: DataFrameRouter,
    procedure_name: str,
    request_bytes: bytes,
    content_type: str,
) -> bytes:
    """Simple handler: bytes in, bytes out with format detection"""

    # Detect input format from Content-Type
    format = ""
    if content_type == "application/vnd.apache.arrow.stream":
        format = "arrow"
    elif content_type == "application/vnd.apache.parquet":
        format = "parquet"
    else:
        raise ValueError

    input_df = to_pandas(request_bytes, format=format)

    result_df = router.call_procedure(procedure_name, input_df)

    return (
        to_arrow(result_df, preserve_index=False)
        if format == "arrow"
        else to_parquet(result_df, preserve_index=False)
    )
