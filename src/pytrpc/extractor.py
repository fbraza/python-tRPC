import inspect
from typing import Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel

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
    properties, required = collect_properties_and_required(sig=sig, hints=hints)
    output, defs = collect_output_types(
        sig=inspect.signature(hints["return"]), return_type=hints["return"]
    )
    return {
        "input": {
            "properties": properties,
            "required": required,
        },
        "output": output,
        "$defs": defs,
    }


def is_pydantic(obj: Any):
    return issubclass(obj, BaseModel)


def collect_properties_and_required(sig: inspect.Signature, hints: dict[str, Any]):
    """
    Function that inspect an object and collect its types to populate properties
    and required schemas entries.
    """
    properties, required = {}, []

    for name, parameter in sig.parameters.items():
        properties[name] = {"type": TYPE_MAPPING[hints.get(name, Any)]}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            properties[name]["default"] = parameter.default

    return properties, required


def collect_output_types(sig: inspect.Signature, return_type: Any):
    """
    Function that specifically inspect the returned type. Presumably should be
    a data model (pydantic, msgspec, dataclasses, attrs). The populate the output,
    defs, with properties and required fields of the schema.
    """
    output: dict[str, Any] = {}
    model: list[Any] = []

    if is_pydantic(return_type):
        model.append(return_type)
        output["$ref"] = f"#/defs/{return_type.__name__}"

    if get_origin(return_type) is list:
        output["type"] = "array"
        has_model = is_pydantic(get_args(return_type)[0])
        obj_type = get_args(return_type)[0]
        if has_model:
            model.append(obj_type)
            output["items"] = {
                "type": "pydantic",
                "$ref": f"#/defs/{obj_type.__name__}",
            }
        else:
            output["items"] = {"type": f"{TYPE_MAPPING[return_type]}"}

    defs: dict[str, Any] = {}
    if model:
        model_to_inspect = model.pop()
        model_sig = inspect.signature(model_to_inspect)
        model_typ = get_type_hints(model_to_inspect)
        properties, required = collect_properties_and_required(
            sig=model_sig, hints=model_typ
        )
        defs[model_to_inspect.__name__] = {
            "properties": properties,
            "required": required,
        }

    return output, defs
