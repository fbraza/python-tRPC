import inspect
from typing import Any, get_args, get_origin, get_type_hints

import models

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
    output, defs = collect_type_refs_and_defs(return_type=hints["return"])
    return {
        "input": {
            "properties": properties,
            "required": required,
        },
        "output": output,
        "$defs": defs,
    }


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


def collect_type_refs_and_defs(return_type: Any):
    """
    Function that specifically inspect the returned type. Presumably should be
    a data model (pydantic, msgspec, dataclasses, attrs). Then populate the output,
    defs, with properties and required fields of the schema.
    """
    output: dict[str, Any] = {}
    model_list: list[Any] = []

    if models.inspect(return_type) is not None:
        model_list.append(return_type)
        output["$ref"] = f"#/defs/{return_type.__name__}"
    elif get_origin(return_type) is list:
        __collect_list_items_types(output, model_list, return_type)
    else:
        output["items"] = {"type": f"{TYPE_MAPPING[return_type]}"}

    defs = __collect_object_definitions(model_list)

    return output, defs


def __collect_list_items_types(
    output: dict[str, Any], model_list: list[Any], return_type: Any
):
    output["type"] = "array"
    generic = get_args(return_type)

    if len(generic) == 1:
        # check if type is pydantic class
        obj = generic[0]
        which_model = models.inspect(obj)
        if which_model is not None:
            model_list.append(obj)
            output["items"] = {
                "type": f"{which_model}",
                "$ref": f"#/defs/{obj.__name__}",
            }
        # if no pydantic model we just map the std types
        else:
            output["items"] = {"type": f"{TYPE_MAPPING[obj]}"}
    elif len(generic) > 1:
        output["items"] = []
        for obj in generic:
            which_model = models.inspect(obj)
            if which_model is not None:
                model_list.append(obj)
                output["items"].append(
                    {
                        "type": f"{which_model}",
                        "$ref": f"#/defs/{obj.__name__}",
                    }
                )
            else:
                output["items"].append({"type": f"{TYPE_MAPPING[obj]}"})


def __collect_object_definitions(model_list: list[Any]) -> dict[str, Any] | list[dict[str, Any]]:
    """
    From a list of models (pydantic, msgspec, dataclasses), generate their definitions.
    """
    defs: list[dict[str, Any]] = []

    while model_list:
        model_to_inspect = model_list.pop()
        model_sig = inspect.signature(model_to_inspect)
        model_typ = get_type_hints(model_to_inspect)
        properties, required = collect_properties_and_required(
            sig=model_sig, hints=model_typ
        )
        defs.append ({ model_to_inspect.__name__: {
                "properties": properties,
                "required": required,
            }
        })

    if len(defs) == 0:
        return {}
    elif len(defs) == 1:
        return defs.pop()
    else:
        return defs
