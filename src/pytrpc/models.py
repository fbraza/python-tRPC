from typing import Any

import pydantic


def is_pydantic(obj: Any):
    return issubclass(obj, pydantic.BaseModel)
