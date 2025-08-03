import inspect
from collections.abc import Callable
from typing import get_type_hints

import extractor


class Router:
    def __init__(self):
        self.proc = {}

    def query(self, name: str | None = None):
        """
        Minimal query decorator that:
        1. Registers the function
        2. Extracts type information
        3. Stores schema for client consumption
        """

        def decorator(func: Callable) -> Callable:
            proc_name = name or func.__name__

            # Extract type information
            type_hints = get_type_hints(func)
            sig = inspect.signature(func)

            self.proc[proc_name] = {
                "type": "query",
                "handler": func,
                "signature": sig,
                "type_hints": type_hints,
                "schema": extractor.schemas(sig=sig, hints=type_hints),
            }

            return func

        return decorator
