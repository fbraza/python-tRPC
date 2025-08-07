import inspect
from typing import get_type_hints

from pydantic import BaseModel
from msgspec import Struct

from pytrpc import extractor


class User(BaseModel):
    id: int
    name: str
    email: str


class Car(Struct):
    color: str
    engine: str


def test_extract_simple_function():
    def get_user(user_id: int) -> User: ...  # type: ignore

    func_sig = inspect.signature(get_user)
    func_typ = get_type_hints(get_user)

    schema = extractor.schemas(sig=func_sig, hints=func_typ)

    assert schema == {
        "input": {
            "properties": {"user_id": {"type": "integer"}},
            "required": ["user_id"],
        },
        "output": {"$ref": "#/defs/User"},
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


def test_extract_with_list_msgspec_and_pydantic_as_output():
    def get_id_of_user_and_cars(pattern: str) -> list[User, Car]: ...  # type: ignore

    func_sig = inspect.signature(get_id_of_user_and_cars)
    func_typ = get_type_hints(get_id_of_user_and_cars)

    schema = extractor.schemas(sig=func_sig, hints=func_typ)

    assert schema == {
        "input": {
            "properties": {
                "pattern": {"type": "string"}
            },
            "required": ["pattern"],
        },
        "output": {
            "type": "array",
            "items": [
                {"type": "pydantic", "$ref": "#/defs/User"},
                {"type": "msgspec", "$ref": "#/defs/Car"},
            ]
        },
        "$defs": [
            {
                "Car": {
                    "properties": {
                        "color": {"type": "string"},
                        "engine": {"type": "string"},
                    },
                    "required": ["color", "engine"],
                }
            },
            {
                "User": {
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["id", "name", "email"],
                }
            }
        ],
    }
