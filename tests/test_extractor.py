import inspect
from typing import get_type_hints

from pydantic import BaseModel

from pytrpc import extractor


class User(BaseModel):
    id: int
    name: str
    email: str


def test_extract_simple_function():
    def get_user(user_id: int) -> User: ...  # type: ignore

    func_sig = inspect.signature(get_user)
    func_typ = get_type_hints(get_user)

    schema = extractor.schemas(sig=func_sig, hints=func_typ)

    assert schema["input"]["properties"]["user_id"]["type"] == "integer"
    assert schema["output"]["$ref"] == "#/defs/User"
    assert "User" in schema["$defs"]


def test_extract_optional_params():
    def list_users(limit: int = 10, offset: int = 0) -> list[User]: ...  # type: ignore

    func_sig = inspect.signature(list_users)
    func_typ = get_type_hints(list_users)

    schema = extractor.schemas(sig=func_sig, hints=func_typ)

    assert schema["input"]["properties"]["limit"]["default"] == 10
    assert schema["input"]["required"] == []  # No required params
