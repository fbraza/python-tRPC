import pandas as pd

from pytrpc.router import DataFrameRouter


def test_calling_procedure_on_df():
    router = DataFrameRouter()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    @router.procedure()
    def add_ten_to_numerical_cols(df: pd.DataFrame) -> pd.DataFrame:
        num_cols = [col for col in df.columns if df[col].dtype == int]
        df[num_cols] += 10
        return df

    result_df = router.call_procedure(name="add_ten_to_numerical_cols", df=df)

    pd.testing.assert_frame_equal(
        result_df, pd.DataFrame({"a": [11, 12, 13], "b": ["x", "y", "z"]})
    )


def test_complex_dataframe_operations():
    """Test router with complex DataFrame operations"""
    router = DataFrameRouter()

    @router.procedure(None, None, "Statistical summary")
    def calculate_stats(df: pd.DataFrame) -> pd.DataFrame:
        return df.describe()

    input_df = pd.DataFrame({"values": [1, 2, 3, 4, 5], "scores": [10, 20, 30, 40, 50]})

    result = router.call_procedure("calculate_stats", input_df)

    # Should return describe() output
    assert "count" in result.index
    assert "mean" in result.index
    assert len(result.columns) == 2  # values and scores columns
