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

    assert schema == {
        "input": {
            "properties": {
                "limit": {"type": "integer", "default": 10},
                "offset": {"type": "integer", "default": 0},
            },
            "required": [],
        },
        "output": {
            "type": "array",
            "items": {"type": "pydantic", "$ref": "#/defs/User"},
        },
        "$defs": {
            "User": {
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["id", "name", "email"],
            }
        },
    }


def test_extract_optional_params():
    def list_users(limit: int = 10, offset: int = 0) -> list[User]: ...  # type: ignore

    func_sig = inspect.signature(list_users)
    func_typ = get_type_hints(list_users)

    schema = extractor.schemas(sig=func_sig, hints=func_typ)

    assert schema == {
        "input": {
            "properties": {
                "limit": {"type": "integer", "default": 10},
                "offset": {"type": "integer", "default": 0},
            },
            "required": [],
        },
        "output": {
            "type": "array",
            "items": {"type": "pydantic", "$ref": "#/defs/User"},
        },
        "$defs": {
            "User": {
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["id", "name", "email"],
            }
        },
    }


def test_extract_with_list_interger_as_output():
    def get_id_of_user_with_name_starting_by(pattern: str) -> list[int]: ...  # type: ignore

    func_sig = inspect.signature(get_id_of_user_with_name_starting_by)
    func_typ = get_type_hints(get_id_of_user_with_name_starting_by)

    schema = extractor.schemas(sig=func_sig, hints=func_typ)

    assert schema == {
        "input": {
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
        "output": {"type": "array", "items": {"type": "integer"}},
        "$defs": {},
    }
