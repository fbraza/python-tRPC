from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import pandas as pd
from pandantic import Pandantic
from pydantic import BaseModel

Pydantic = TypeVar("Pydantic", bound=BaseModel)


@dataclass
class ProcedureInfo:
    """Metadata about a registered procedure"""

    name: str
    func: Callable[[pd.DataFrame], pd.DataFrame]
    input_schema: type[Pydantic] | None = None  # type: ignore
    output_schema: type[Pydantic] | None = None  # type: ignore
    description: str | None = None


class DataFrameRouter:
    """Registry for DataFrame procedures"""

    def __init__(self):
        self._procedures: dict[str, ProcedureInfo] = {}

    def procedure(
        self,
        input_schema: type[Pydantic] | None = None,
        output_schema: type[Pydantic] | None = None,
        description: str | None = None,
    ):
        """Decorator to register DataFrame procedures"""

        def decorator(func):
            # Store procedure info
            self._procedures[func.__name__] = ProcedureInfo(
                name=func.__name__,
                func=func,
                input_schema=input_schema,
                output_schema=output_schema,
                description=description,
            )
            # Return original function unchanged
            return func

        return decorator

    def call_procedure(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Call a registered procedure"""
        if name not in self._procedures:
            raise ValueError(f"Procedure '{name}' not found")

        proc_info = self._procedures[name]

        if proc_info.input_schema:
            validator = Pandantic(schema=proc_info.input_schema)
            validator.validate(dataframe=df, errors="raise")

        processed_df = proc_info.func(df)
        if proc_info.output_schema:
            validator = Pandantic(schema=proc_info.output_schema)
            validator.validate(dataframe=df, errors="raise")

        return processed_df

    def list_procedures(self) -> list[str]:
        """List all procedure names"""
        return list(self._procedures.keys())
