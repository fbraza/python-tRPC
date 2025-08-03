import inspect
from typing import Any, get_type_hints

TYPE_MAPPING = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
    Any: "any",
}


def schemas(sig: inspect.Signature, hints: dict[str, Any]):
    """Example output:
    {
        "input": {
            "type": "object",
            "properties": {
                "user_id": {"type": "integer", "default": false},
                "user_name": {"type": "string"},
                "is_subscribed": {"type": "boolean", "default": "false"}
            },
            "required": ["user_id"]
        },
        "output": {
            "$ref": "#/defs/User"
            "$ref": "#/defs/User/Nested"
        },
        "$defs": {
            "User": { ... Pydantic schema ... }
        }
    }
    """
    input_properties, input_required = __collect_schemas(sig=sig, hints=hints)
    output_properties, output_required = __collect_schemas(
        sig=inspect.signature(hints["return"]), hints=get_type_hints(hints["return"])
    )

    schema = {
        "input": {
            "type": "object",
            "properties": input_properties,
            "required": input_required,
        },
        "output": {"$ref": f"#/defs/{hints['return'].__name__}"},
        "$defs": {
            hints["return"].__name__: {
                "properties": output_properties,
                "required": output_required,
            },
        },
    }

    print(schema)
    return schema


def __collect_schemas(sig: inspect.Signature, hints: dict[str, Any]):
    properties, required = {}, []

    for name, parameter in sig.parameters.items():
        properties[name] = {"type": TYPE_MAPPING[hints.get(name, Any)]}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            properties[name]["default"] = parameter.default

    return properties, required
