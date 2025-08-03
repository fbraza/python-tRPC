# Python tRPC Specification Document

## Project: pytrpc - Type-Safe RPC for Python Services

### Executive Summary

Create a Python library that enables type-safe, zero-code-generation RPC between Python services, inspired by TypeScript's tRPC. The library should automatically propagate type information from server to client, providing IDE autocomplete, type checking, and runtime validation without manual schema definitions or code generation steps. Unlike existing solutions, pytrpc will be model-agnostic, supporting Pydantic, msgspec, cattrs, and dataclasses, allowing teams to choose the serialization library that best fits their performance and feature requirements.

### Core Design Principles

1. **Zero Code Generation**: Types should flow from server to client without any build step
2. **Runtime Type Safety**: Leverage Python's type hints for validation at runtime
3. **Developer Experience First**: Full IDE support with autocomplete and type checking
4. **Framework Agnostic**: Should work with FastAPI, Flask, Django, or standalone
5. **Model Library Agnostic**: Support multiple serialization libraries (Pydantic, msgspec, cattrs, dataclasses)
6. **Progressive Enhancement**: Can be added incrementally to existing projects

## Architecture Overview

```python
# Server Side Architecture

"""
1. Router defines procedures (queries/mutations)
2. Type Extractor analyzes function signatures and model types
3. Model Adapters handle different serialization libraries (Pydantic, msgspec, cattrs, dataclasses)
4. Schema Endpoint exposes type information as JSON
5. HTTP Adapter handles transport (FastAPI/Flask/etc)
"""

# Client Side Architecture
"""
1. Schema Fetcher retrieves type info from server
2. Dynamic Proxy creates typed methods at runtime
3. Type Stub Generator creates .pyi files for IDE support
4. Request Builder handles serialization/transport
"""
```

## Core Components Specification

### 1. Router System

```python
from typing import TypeVar, Generic, Callable, Any, Protocol

class Router:
    """
    Central router that collects all procedures and their type information.
    """

    def __init__(self, model_adapter=None):
        self.procedures = {}
        self.models = {}  # Model registry (works with any model library)
        self.model_adapter = model_adapter or self._detect_model_adapter()

    def query(self, name: str = None):
        """
        Decorator for read-only operations.
        Should be used for data fetching without side effects.
        """
        def decorator(func: Callable) -> Callable:
            procedure_name = name or func.__name__
            self.procedures[procedure_name] = {
                "type": "query",
                "handler": func,
                "input_schema": extract_input_schema(func),
                "output_schema": extract_output_schema(func),
                "metadata": extract_metadata(func)
            }
            return func
        return decorator

    def mutation(self, name: str = None):
        """
        Decorator for operations with side effects.
        Should be used for create/update/delete operations.
        """
        # Similar to query but with type="mutation"
        pass

    def subscription(self, name: str = None):
        """
        Decorator for real-time subscriptions (WebSocket).
        Returns async generator or observable pattern.
        """
        pass

    def merge(self, *routers: 'Router', prefix: str = None):
        """
        Combine multiple routers for modular API design.
        """
        pass
```

### 2. Model Adapter System

```python
from typing import Protocol, Any, Type, Dict
from abc import ABC, abstractmethod

class ModelAdapter(Protocol):
    """
    Protocol for model library adapters.
    """

    def is_model_type(self, typ: Type) -> bool:
        """Check if a type is a model from this library."""
        ...

    def model_to_json_schema(self, model_type: Type) -> dict:
        """Convert model to JSON Schema format."""
        ...

    def validate_data(self, data: Any, model_type: Type) -> Any:
        """Validate and deserialize data against model."""
        ...

    def serialize_model(self, instance: Any) -> dict:
        """Serialize model instance to dict."""
        ...

class PydanticAdapter:
    """Adapter for Pydantic models."""

    def is_model_type(self, typ: Type) -> bool:
        try:
            from pydantic import BaseModel
            return issubclass(typ, BaseModel)
        except:
            return False

    def model_to_json_schema(self, model_type: Type) -> dict:
        return model_type.schema()

class MsgspecAdapter:
    """Adapter for msgspec Struct types."""

    def is_model_type(self, typ: Type) -> bool:
        try:
            import msgspec
            return msgspec.structs.is_struct(typ)
        except:
            return False

    def model_to_json_schema(self, model_type: Type) -> dict:
        # Convert msgspec struct to JSON Schema
        import msgspec
        return msgspec.json.schema(model_type)

class CattrsAdapter:
    """Adapter for cattrs with dataclasses/attrs."""

    def is_model_type(self, typ: Type) -> bool:
        import dataclasses
        import attr
        return dataclasses.is_dataclass(typ) or attr.has(typ)

class DataclassAdapter:
    """Adapter for Python dataclasses."""

    def is_model_type(self, typ: Type) -> bool:
        import dataclasses
        return dataclasses.is_dataclass(typ)
```

### 3. Type Extraction System

```python
import inspect
from typing import get_type_hints, get_origin, get_args

class TypeExtractor:
    """
    Extracts Python type information and converts to JSON-serializable schema.
    Works with any model library through adapters.
    """

    def __init__(self, model_adapters: list[ModelAdapter]):
        self.model_adapters = model_adapters

    def extract_function_schema(self, func: Callable) -> dict:
        """
        Extract complete type information from a function.

        Returns:
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
                "type": "object",
                "ref": "#/definitions/User"  # Reference to model
            }
        }
        """
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Handle different input patterns:
        # 1. Individual parameters: func(id: int, name: str)
        # 2. Single model: func(data: UserInput)
        # 3. Mixed: func(id: int, data: UserUpdate)

        return {
            "input": self._extract_input_params(signature, type_hints),
            "output": self._extract_return_type(type_hints.get('return')),
            "async": inspect.iscoroutinefunction(func),
            "deprecated": extract_deprecation_info(func)
        }

    def _extract_type_schema(self, typ: Type) -> dict:
        """
        Convert any type to JSON Schema, checking model adapters first.
        """
        # Check if it's a model type
        for adapter in self.model_adapters:
            if adapter.is_model_type(typ):
                return adapter.model_to_json_schema(typ)

        # Fall back to standard type extraction
        return extract_standard_type_schema(typ)
```

### 4. Client Implementation

```python
from typing import TypeVar, Generic, Any, Optional
import httpx

T = TypeVar('T')

class PyTRPCClient:
    """
    Dynamic client that fetches schema and creates typed interface.
    """

    def __init__(
        self,
        url: str,
        headers: dict = None,
        fetch_schema_on_init: bool = True,
        cache_schema: bool = True,
        timeout: int = 30
    ):
        self.base_url = url
        self.headers = headers or {}
        self._schema = None
        self._procedures = {}

        if fetch_schema_on_init:
            self._fetch_and_build_schema()

    def _fetch_and_build_schema(self):
        """
        Fetch schema from server and build procedure proxies.
        """
        response = httpx.get(f"{self.base_url}/trpc/schema")
        self._schema = response.json()

        # Dynamically create methods
        for name, procedure in self._schema["procedures"].items():
            if procedure["type"] == "query":
                setattr(self, name, self._create_query_method(name, procedure))
            elif procedure["type"] == "mutation":
                setattr(self, name, self._create_mutation_method(name, procedure))

    def _create_query_method(self, name: str, schema: dict):
        """
        Create a typed method for a query procedure.
        """
        async def query_method(**kwargs):
            # Validate inputs against schema
            validated = self._validate_input(kwargs, schema["input"])

            # Make HTTP request
            response = await self._http_client.get(
                f"{self.base_url}/trpc/{name}",
                params=validated
            )

            # Validate and parse output
            return self._parse_output(response.json(), schema["output"])

        # Attach type hints for IDE support
        query_method.__annotations__ = self._build_annotations(schema)
        return query_method

class TypedProcedureProxy(Generic[T]):
    """
    Provides typed wrapper for better IDE support.
    """
    async def __call__(self, **kwargs) -> T:
        pass
```

### 5. Server Adapters

```python
# FastAPI Adapter
from fastapi import FastAPI, Request
from pytrpc import Router

class FastAPIAdapter:
    """
    Integrates PyTRPC router with FastAPI application.
    """

    def __init__(self, router: Router, path: str = "/trpc"):
        self.router = router
        self.path = path

    def apply(self, app: FastAPI):
        """
        Register all routes with FastAPI.
        """
        # Add schema endpoint
        @app.get(f"{self.path}/schema")
        async def get_schema():
            return self.router.get_schema()

        # Add procedure endpoints
        for name, procedure in self.router.procedures.items():
            if procedure["type"] == "query":
                self._register_query(app, name, procedure)
            elif procedure["type"] == "mutation":
                self._register_mutation(app, name, procedure)

    def _register_query(self, app: FastAPI, name: str, procedure: dict):
        """
        Register GET endpoint for query.
        """
        handler = procedure["handler"]

        @app.get(f"{self.path}/{name}")
        async def query_endpoint(request: Request):
            # Extract and validate params
            params = dict(request.query_params)
            validated = validate_against_schema(params, procedure["input_schema"])

            # Call original handler
            result = await handler(**validated) if asyncio.iscoroutinefunction(handler) else handler(**validated)

            # Serialize response
            return {"result": result}

# Flask Adapter
class FlaskAdapter:
    """Similar implementation for Flask"""
    pass

# Django Adapter
class DjangoAdapter:
    """Similar implementation for Django"""
    pass
```

### 6. Type Stub Generation

```python
class StubGenerator:
    """
    Generates .pyi files for static type checking support.
    """

    @staticmethod
    def generate_client_stubs(schema: dict, output_path: str):
        """
        Generate Python stub files from server schema.

        Example output:
        # generated_client.pyi
        from typing import Optional, List
        from datetime import datetime
        from .models import User, Post

        class APIClient:
            async def get_user(self, user_id: int) -> User: ...
            async def list_posts(
                self,
                author_id: Optional[int] = None,
                limit: int = 10,
                offset: int = 0
            ) -> List[Post]: ...
            async def create_post(self, title: str, content: str) -> Post: ...
        """
        stub_content = generate_stub_from_schema(schema)
        write_stub_file(output_path, stub_content)
```

## Usage Examples

### Example 1: Basic User Service (Multiple Model Libraries)

```python
# Example with Pydantic
from pytrpc import Router, create_app
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = Router()  # Auto-detects Pydantic

class User(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True

class CreateUserInput(BaseModel):
    username: str
    email: str
    password: str

# Example with msgspec (much faster)
import msgspec
from pytrpc import Router, MsgspecAdapter

router = Router(model_adapter=MsgspecAdapter())

class User(msgspec.Struct):
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True

class CreateUserInput(msgspec.Struct):
    username: str
    email: str
    password: str

# Example with dataclasses
from dataclasses import dataclass
from pytrpc import Router, DataclassAdapter

router = Router(model_adapter=DataclassAdapter())

@dataclass
class User:
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True

@dataclass
class CreateUserInput:
    username: str
    email: str
    password: str

# The router code remains the same for all model libraries!
@router.query()
async def get_user(user_id: int) -> User:
    # Fetch from database
    return User(
        id=user_id,
        username="john_doe",
        email="john@example.com",
        created_at=datetime.now()
    )

@router.query()
async def list_users(
    limit: int = 10,
    offset: int = 0,
    is_active: Optional[bool] = None
) -> List[User]:
    # Query database with filters
    return [...]

@router.mutation()
async def create_user(data: CreateUserInput) -> User:
    # Validate, hash password, save to DB
    return User(...)

# Create FastAPI app with tRPC
from fastapi import FastAPI
app = FastAPI()
router.attach_to_fastapi(app, path="/api/trpc")

# client.py
from pytrpc import Client

# Client automatically gets all type information!
api = Client("http://localhost:8000/api/trpc")

# Full IDE autocomplete and type checking
user = await api.get_user(user_id=123)  # ✅ Returns User type
print(user.username)  # ✅ IDE knows this exists

users = await api.list_users(limit=20, is_active=True)  # ✅ Returns List[User]
for user in users:
    print(user.email)  # ✅ Type checked

new_user = await api.create_user(data={
    "username": "jane_doe",
    "email": "jane@example.com",
    "password": "secure123"
})  # ✅ Input validated against CreateUserInput
```

### Example 2: Nested Routers (Modular APIs)

```python
# routers/auth.py
auth_router = Router()

@auth_router.mutation()
async def login(email: str, password: str) -> Token:
    return Token(access_token="...", refresh_token="...")

@auth_router.mutation()
async def logout(token: str) -> bool:
    return True

# routers/posts.py
posts_router = Router()

@posts_router.query()
async def get_post(post_id: int) -> Post:
    return Post(...)

# main.py
app_router = Router()
app_router.merge(auth_router, prefix="auth")
app_router.merge(posts_router, prefix="posts")

# Client usage
client = Client("http://api.example.com")
token = await client.auth.login(email="...", password="...")  # Namespaced!
post = await client.posts.get_post(post_id=1)
```

## Implementation Phases

### Phase 1: Core Functionality (MVP)
- Basic router with query/mutation decorators
- Type extraction for simple types (int, str, bool, list, dict)
- Model adapter system with Pydantic and dataclasses support
- FastAPI adapter
- Basic client with runtime type checking
- Schema endpoint

### Phase 2: Enhanced Type Support
- Complex types (Union, Optional, Literal, Enum)
- Additional model library adapters (msgspec, cattrs)
- Nested models across different libraries
- Custom validators per model library
- File uploads
- Datetime/UUID handling
- Recursive types

### Phase 3: Developer Experience
- Type stub generation (.pyi files)
- IDE plugins (VS Code, PyCharm)
- Better error messages with type hints
- Development mode with hot reload
- Debug mode with request/response logging

### Phase 4: Advanced Features
- WebSocket subscriptions
- Batch queries
- Request caching
- Middleware support (auth, logging, rate limiting)
- Schema versioning
- OpenAPI generation

### Phase 5: Production Features
- Connection pooling
- Retry logic with exponential backoff
- Circuit breakers
- Distributed tracing support
- Metrics integration
- Schema evolution strategies

## Technical Challenges & Solutions

### Challenge 1: Runtime Type Information
Python's type hints are not available at runtime by default.

**Solution:**
```python
# Use typing.get_type_hints() with eval_str=True
# Store type info in function attributes
# Use typing_extensions for backward compatibility
```

### Challenge 2: Multiple Model Library Support
Each library has different APIs for schema generation and validation.

**Solution:**
```python
# Adapter pattern with common protocol
# Feature detection for library capabilities
# Graceful fallback when features not supported
# Unified error handling across libraries
```

### Challenge 3: Circular Dependencies
Models may have circular references across different libraries.

**Solution:**
```python
# Use forward references universally
# Implement lazy loading for model schemas
# Use JSON Schema $ref for circular references
# Library-specific handling (update_forward_refs for Pydantic)
```

### Challenge 4: Generic Types
Handling Generic[T] and TypeVar in type extraction.

**Solution:**
```python
# Track TypeVar bindings during extraction
# Generate specialized schemas for each concrete type
# Use __orig_class__ for runtime generic info
```

### Challenge 5: Performance
Schema fetching and validation overhead.

**Solution:**
```python
# Cache schemas client-side
# Use msgspec for faster serialization (when available)
# Compile validators with Cython
# Lazy load procedures on first use
# Choose optimal model library based on performance needs
```

## Open Design Questions

1. **How should we handle authentication/authorization?**
   - Middleware pattern?
   - Built-in decorators?
   - Context propagation?

2. **Should we support GraphQL-like field selection?**
   ```python
   # Only fetch specific fields?
   user = await api.get_user(user_id=1, select=["id", "username"])
   ```

3. **How to handle streaming responses?**
   ```python
   # Server-sent events? Async generators?
   async for update in api.watch_changes():
       print(update)
   ```

4. **File upload strategy?**
   - Multipart form data?
   - Base64 encoding?
   - Separate upload endpoint?

5. **Error handling philosophy?**
   - Custom exception types?
   - Error codes?
   - Structured error responses?

6. **Model library mixing?**
   - Should we allow mixing different model libraries in one router?
   - How to handle conversions between libraries?
   - Performance implications of adapter pattern?

## Compatibility Requirements

- Python 3.8+ (for TypedDict and modern typing features)
- Should work with:
  - FastAPI 0.68+
  - Flask 2.0+
  - Django 3.2+
  - Starlette
  - aiohttp
- Type checkers:
  - mypy 0.900+
  - pyright/pylance
  - pyre

## Success Metrics

1. **Performance**: <5ms overhead per request
2. **Type Coverage**: 100% of Python's typing module
3. **Developer Experience**: Full IDE autocomplete in VS Code/PyCharm
4. **Adoption**: 1000+ GitHub stars in first year
5. **Documentation**: 100% API coverage with examples

## Code Style & Project Structure

```
pytrpc/
├── src/
│   ├── pytrpc/
│   │   ├── __init__.py
│   │   ├── router.py           # Core router implementation
│   │   ├── client.py           # Client implementation
│   │   ├── extraction.py       # Type extraction utilities
│   │   ├── validation.py       # Runtime validation
│   │   ├── schema.py           # Schema generation
│   │   ├── model_adapters/     # Model library adapters
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # Base adapter protocol
│   │   │   ├── pydantic.py     # Pydantic adapter
│   │   │   ├── msgspec.py      # msgspec adapter
│   │   │   ├── cattrs.py       # cattrs adapter
│   │   │   └── dataclasses.py  # dataclasses adapter
│   │   ├── server_adapters/    # Web framework adapters
│   │   │   ├── fastapi.py
│   │   │   ├── flask.py
│   │   │   └── django.py
│   │   └── utils/
│   │       ├── stubs.py        # Stub file generation
│   │       └── types.py        # Type utilities
├── tests/
├── examples/
├── docs/
└── benchmarks/
```

## Additional Considerations

### Security
- Input sanitization and validation
- Rate limiting support
- CORS configuration
- Authentication token handling
- SQL injection prevention in generated queries

### Monitoring & Observability
- OpenTelemetry integration
- Prometheus metrics
- Structured logging
- Request ID propagation
- Performance profiling hooks

### Testing Strategy
- Unit tests for type extraction
- Integration tests with real servers
- Property-based testing for validators
- Load testing for performance validation
- Type checker regression tests

### Documentation Requirements
- Getting started guide
- API reference
- Model library adapter guide (choosing the right one)
- Migration guides from REST/GraphQL
- Best practices guide
- Common patterns cookbook
- Performance comparison between model libraries
- Troubleshooting guide

### Community & Ecosystem
- Discord/Slack community
- GitHub discussions
- Example projects repository
- Plugin system for extensions
- Corporate sponsor friendly license (MIT/Apache 2.0)

This specification provides a comprehensive foundation for building pytrpc. It can be iteratively refined as implementation progresses and design decisions are made.
