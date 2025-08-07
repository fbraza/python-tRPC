import dataclasses

import attrs
import msgspec
import pydantic

from pytrpc import models


class User(pydantic.BaseModel):
    name: str
    age: int


@dataclasses.dataclass
class Car:
    model: str
    color: str


class Cat(msgspec.Struct):
    name: str
    age: int


@attrs.define
class Dog:
    name: str
    age: int


def test_check_if_right_model():
    assert models.__is_pydantic(User)
    assert models.__is_msgspec(Cat)
    assert models.__is_dataclass(Car)
    assert models.__is_attrs(Dog)

    assert not models.__is_dataclass(Dog)
    assert not models.__is_dataclass(Cat)
    assert not models.__is_dataclass(User)

    assert not models.__is_pydantic(Dog)
    assert not models.__is_pydantic(Cat)
    assert not models.__is_pydantic(Car)

    assert not models.__is_msgspec(Dog)
    assert not models.__is_msgspec(User)
    assert not models.__is_msgspec(Car)

    assert not models.__is_attrs(Cat)
    assert not models.__is_attrs(User)
    assert not models.__is_attrs(Car)
