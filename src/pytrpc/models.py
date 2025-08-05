def is_pydantic(cls):
    """
    Check for if cls is pydantic V1 or V2
    """
    return (
        getattr(cls, "model_fields", None) is not None
        or getattr(cls, "__fields__", None) is not None
    )


def is_msgspec(cls):
    """
    Check for if cls is msgspec
    """
    return getattr(cls, "__struct_fields__", None) is not None


def is_dataclass(cls):
    """Check if type is a dataclass"""
    return getattr(cls, "__dataclass_fields__", None) is not None


def is_attrs(cls):
    """Check if type is an attrs class"""
    return getattr(cls, "__attrs_attrs__", None) is not None
