def inspect(cls) -> str | None:
    """
    Collect and name the model
    """
    model = None
    predicates = {
        __is_pydantic: "pydantic",
        __is_msgspec: "msgspec",
        __is_dataclass: "dataclass",
        __is_attrs: "attrs",
    }

    for p in predicates:
        if p(cls):
            model = predicates[p]

    return model


def __is_pydantic(cls):
    """
    Check for if cls is pydantic V1 or V2
    """
    return (
        getattr(cls, "model_fields", None) is not None
        or getattr(cls, "__fields__", None) is not None
    )


def __is_msgspec(cls):
    """
    Check for if cls is msgspec
    """
    return getattr(cls, "__struct_fields__", None) is not None


def __is_dataclass(cls):
    """Check if type is a dataclass"""
    return getattr(cls, "__dataclass_fields__", None) is not None


def __is_attrs(cls):
    """Check if type is an attrs class"""
    return getattr(cls, "__attrs_attrs__", None) is not None
