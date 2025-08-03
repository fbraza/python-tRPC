# pytrpc MVP: Minimal Viable Prototype

## Overview

This document outlines a step-by-step approach to building a minimal working prototype of pytrpc with:
- Single `@router.query()` decorator
- Pydantic model support only (for MVP)
- Basic type extraction and schema generation
- Simple HTTP transport (GET requests)
- Dynamic client with type information

## Architecture Breakdown

### 1. Core Router Implementation

```python
# pytrpc/router.py

from typing import Callable, Dict, Any, TypeVar, get_type_hints
import inspect
from pydantic import BaseModel

class Router:
    def __init__(self):
        self.procedures: Dict[str, Dict[str, Any]] = {}

    def query(self, name: str = None):
        """
        Minimal query decorator that:
        1. Registers the function
        2. Extracts type information
        3. Stores schema for client consumption
        """
        def decorator(func: Callable) -> Callable:
            procedure_name = name or func.__name__

            # Extract type information
            type_hints = get_type_hints(func)
            sig = inspect.signature(func)

            self.procedures[procedure_name] = {
                "type": "query",
                "handler": func,
                "signature": sig,
                "type_hints": type_hints,
                "schema": self._extract_schema(func)
            }

            return func
        return decorator

    def _extract_schema(self, func: Callable) -> dict:
        """
        TODO: Implement schema extraction
        Should return JSON Schema compatible format
        """
        pass
```

### 2. Type Extraction Strategy

```python
# pytrpc/extraction.py

from typing import Any, Type, get_type_hints, get_origin, get_args
from pydantic import BaseModel
import inspect

class TypeExtractor:
    """
    Handles conversion of Python types to JSON Schema
    """

    @staticmethod
    def extract_function_schema(func: Callable) -> dict:
        """
        Extracts input/output schema from function signature

        Example output:
        {
            "input": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"},
                    "include_deleted": {"type": "boolean", "default": false}
                },
                "required": ["user_id"]
            },
            "output": {
                "$ref": "#/definitions/User"
            },
            "definitions": {
                "User": { ... Pydantic schema ... }
            }
        }
        """
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        input_schema = TypeExtractor._extract_parameters(sig, type_hints)
        output_schema = TypeExtractor._extract_return_type(type_hints.get('return'))

        return {
            "input": input_schema,
            "output": output_schema,
            "definitions": TypeExtractor._collect_definitions(sig, type_hints)
        }

    @staticmethod
    def _extract_parameters(sig: inspect.Signature, hints: dict) -> dict:
        """Extract parameter schema"""
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            param_type = hints.get(param_name, Any)
            properties[param_name] = TypeExtractor._python_type_to_json_schema(param_type)

            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            else:
                properties[param_name]["default"] = param.default

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    @staticmethod
    def _python_type_to_json_schema(python_type: Type) -> dict:
        """Convert Python type to JSON Schema"""
        # Basic types
        if python_type == int:
            return {"type": "integer"}
        elif python_type == str:
            return {"type": "string"}
        elif python_type == bool:
            return {"type": "boolean"}
        elif python_type == float:
            return {"type": "number"}

        # Pydantic models
        if isinstance(python_type, type) and issubclass(python_type, BaseModel):
            return {"$ref": f"#/definitions/{python_type.__name__}"}

        # TODO: Handle List, Dict, Optional, etc.
        return {"type": "any"}
```

### 3. Server Adapter (FastAPI Example)

```python
# pytrpc/adapters/fastapi.py

from fastapi import FastAPI, Query
from pytrpc.router import Router
import json

class FastAPIAdapter:
    def __init__(self, router: Router, path_prefix: str = "/trpc"):
        self.router = router
        self.path_prefix = path_prefix

    def attach(self, app: FastAPI):
        """Attach router to FastAPI app"""

        # Schema endpoint
        @app.get(f"{self.path_prefix}/schema")
        async def get_schema():
            """Return complete API schema for client"""
            schema = {
                "procedures": {}
            }

            for name, procedure in self.router.procedures.items():
                schema["procedures"][name] = {
                    "type": procedure["type"],
                    "schema": procedure["schema"]
                }

            return schema

        # Register each procedure
        for name, procedure in self.router.procedures.items():
            self._register_procedure(app, name, procedure)

    def _register_procedure(self, app: FastAPI, name: str, procedure: dict):
        """Register a single procedure as HTTP endpoint"""
        handler = procedure["handler"]
        sig = procedure["signature"]

        # Create dynamic endpoint
        path = f"{self.path_prefix}/{name}"

        # For MVP, we'll use query parameters for all inputs
        async def endpoint(**kwargs):
            # TODO: Add validation against schema
            result = await handler(**kwargs) if inspect.iscoroutinefunction(handler) else handler(**kwargs)

            # TODO: Serialize Pydantic models properly
            return {"result": result}

        # Copy signature for FastAPI
        endpoint.__signature__ = sig

        app.get(path)(endpoint)
```

### 4. Client Implementation

```python
# pytrpc/client.py

import httpx
from typing import Any, Dict

class PyTRPCClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self._schema = None
        self._client = httpx.AsyncClient()

    async def _fetch_schema(self):
        """Fetch and cache schema from server"""
        response = await self._client.get(f"{self.base_url}/schema")
        self._schema = response.json()

        # Dynamically create methods
        for name, procedure in self._schema["procedures"].items():
            setattr(self, name, self._create_method(name, procedure))

    def _create_method(self, name: str, procedure: dict):
        """Create a dynamic method for a procedure"""
        async def method(**kwargs):
            # For MVP, send as query parameters
            response = await self._client.get(
                f"{self.base_url}/{name}",
                params=kwargs
            )

            data = response.json()
            return data["result"]

        # TODO: Add proper type annotations
        return method

    async def __aenter__(self):
        await self._fetch_schema()
        return self

    async def __aexit__(self, *args):
        await self._client.aclose()
```

## Testing Strategy

### 1. Unit Tests Structure

```python
# tests/test_extraction.py

import pytest
from pydantic import BaseModel
from pytrpc.extraction import TypeExtractor

class User(BaseModel):
    id: int
    name: str
    email: str

def test_extract_simple_function():
    def get_user(user_id: int) -> User:
        pass

    schema = TypeExtractor.extract_function_schema(get_user)

    assert schema["input"]["properties"]["user_id"]["type"] == "integer"
    assert schema["output"]["$ref"] == "#/definitions/User"
    assert "User" in schema["definitions"]

def test_extract_optional_params():
    def list_users(limit: int = 10, offset: int = 0) -> list[User]:
        pass

    schema = TypeExtractor.extract_function_schema(list_users)

    assert schema["input"]["properties"]["limit"]["default"] == 10
    assert schema["input"]["required"] == []  # No required params
```

### 2. Integration Tests

```python
# tests/test_integration.py

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pytrpc import Router, FastAPIAdapter, PyTRPCClient

class User(BaseModel):
    id: int
    name: str
    email: str

@pytest.fixture
def app():
    # Setup
    app = FastAPI()
    router = Router()

    @router.query()
    def get_user(user_id: int) -> User:
        return User(id=user_id, name="John Doe", email="john@example.com")

    @router.query()
    def list_users(limit: int = 10) -> list[User]:
        return [
            User(id=1, name="John", email="john@example.com"),
            User(id=2, name="Jane", email="jane@example.com")
        ][:limit]

    adapter = FastAPIAdapter(router)
    adapter.attach(app)

    return app

@pytest.fixture
def test_client(app):
    return TestClient(app)

async def test_client_server_interaction(test_client):
    """Test full client-server flow"""
    base_url = "http://testserver/trpc"

    async with PyTRPCClient(base_url) as client:
        # Test single user fetch
        user = await client.get_user(user_id=123)
        assert user["id"] == 123
        assert user["name"] == "John Doe"

        # Test list with params
        users = await client.list_users(limit=1)
        assert len(users) == 1

def test_schema_endpoint(test_client):
    """Test schema generation"""
    response = test_client.get("/trpc/schema")
    schema = response.json()

    assert "procedures" in schema
    assert "get_user" in schema["procedures"]
    assert schema["procedures"]["get_user"]["type"] == "query"
```

### 3. Type Safety Tests

```python
# tests/test_type_safety.py

def test_client_type_stubs():
    """
    Test that generated type stubs work correctly
    This would be tested with mypy/pyright in CI
    """
    # Example of what we want to achieve:
    # client: PyTRPCClient
    # user = await client.get_user(user_id=123)  # Should know return type is User
    # reveal_type(user)  # Should show: User
    pass
```

## Implementation Steps

### Phase 1: Core Components (Week 1)
1. Implement basic Router with @query decorator
2. Create TypeExtractor for simple types + Pydantic
3. Write comprehensive unit tests

### Phase 2: Server Integration (Week 2)
1. Build FastAPIAdapter
2. Implement schema endpoint
3. Add request/response handling

### Phase 3: Client Implementation (Week 3)
1. Create dynamic client with schema fetching
2. Add method generation
3. Implement basic error handling

### Phase 4: Type Safety (Week 4)
1. Generate .pyi stub files
2. Add mypy tests
3. Improve IDE integration

## Key Design Decisions

### 1. Schema Format
Use JSON Schema with $ref for model definitions. This is standard and well-supported.

### 2. Transport Protocol
Start with simple HTTP GET for queries. This simplifies the MVP and allows easy testing.

### 3. Serialization
Leverage Pydantic's built-in `.dict()` and `.parse_obj()` for serialization.

### 4. Error Handling
Return structured errors:
```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid input",
        "details": {...}
    }
}
```

## Next Steps After MVP

1. Add @mutation decorator (POST requests)
2. Support more type hints (Optional, Union, List, Dict)
3. Add middleware support
4. Implement batching
5. Add WebSocket support for subscriptions
6. Support other model libraries (msgspec, attrs, dataclasses)

## Success Criteria

The MVP is successful when:
1. ✅ Server can expose typed procedures via decorator
2. ✅ Client can dynamically call procedures with correct types
3. ✅ Full type information flows from server to client
4. ✅ IDE autocomplete works in client code
5. ✅ Tests pass with 100% coverage of core functionality
