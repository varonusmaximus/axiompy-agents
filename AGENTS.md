# AxiomPy Cursor Rules
# These rules enforce patterns and practices used throughout the axiompy codebase.

## Project Overview
AxiomPy is a comprehensive library of reusable utilities for Python applications including:
- I/O utilities (HTTP, Database, Object Storage, File, JSON-RPC)
- Data engineering (Pandas/Spark adapters, pipelines, streaming)
- Servers (MCP, JSON-RPC, Flask/FastAPI)
- Validation, logging, decorators, and error handling

## Code Style & Formatting

### General
- Use Python 3.10+ compatible code
- Format with `black` and lint with `ruff`
- Maximum line length: 100 characters
- Use type hints for ALL function parameters and return types
- Use `from __future__ import annotations` for forward references when needed
- **Prefer `match/case` over `if/elif` chains** for type/value dispatch

### Imports
- Group imports: stdlib, third-party, local (separated by blank lines)
- Use absolute imports from `axiompy.*`
- Import specific items, not entire modules

```python
# Good
from axiompy.io.http import HTTPClientFactory, RetryConfig
from axiompy.validators import ensure_not_empty, ensure_url

# Bad
from axiompy.io import *
import axiompy.io.http as http
```

## Design Patterns

### 1. Factory Pattern (REQUIRED for instantiable classes)
All major classes must have a corresponding Factory class for creation.

**Enum-Based Type Selection (REQUIRED for factories with multiple implementations)**

When a factory supports multiple implementations (e.g., different backends, providers, or sources), use **enums** for type selection instead of convenience methods like `create_for_X()`:

```python
# Bad - Convenience methods for each type
class MyServiceFactory:
    @staticmethod
    def create_for_postgres() -> MyService: ...  # ❌ Don't do this
    
    @staticmethod
    def create_for_mysql() -> MyService: ...     # ❌ Don't do this

# Good - Enum-based type selection (consistent with DatabaseFactory, ServerFactory)
from enum import Enum

class ServiceType(str, Enum):
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"

class MyServiceFactory:
    """Factory for creating MyService instances."""
    
    @staticmethod
    def create(
        service_type: ServiceType,
        settings: MyServiceSettings,
    ) -> MyService:
        """
        Create a MyService instance.
        
        Args:
            service_type: Type of service to create
            settings: Configuration for the service
            
        Returns:
            Configured MyService instance
        """
        match service_type:
            case ServiceType.POSTGRES:
                return PostgresService(settings)
            case ServiceType.MYSQL:
                return MySQLService(settings)
            case ServiceType.SQLITE:
                return SQLiteService(settings)
            case _:
                raise ValueError(f"Unknown service type: {service_type}")
    
    @staticmethod
    def create_mock() -> MockMyService:
        """Create a mock instance for testing."""
        return MockMyService()
```

**Why enum-based selection:**
- Consistent with existing axiompy factories (DatabaseFactory, ServerFactory, ObjectStorageFactory)
- IDE autocompletion shows all available types
- Single entry point makes it easy to add new types
- Type-safe - compiler catches invalid types

**Examples from axiompy:**
```python
# Database
db = DatabaseFactory.create(DatabaseType.POSTGRES, settings)

# Server
server = ServerFactory.create(ServerType.FASTAPI, settings)

# Object Storage
storage = ObjectStorageFactory.create(ObjectStorageType.S3, settings)
```

**Sub-Factories for Adapter Modules (REQUIRED for modules with multiple adapters)**

When a module has multiple adapter implementations (e.g., different embedders, vector stores, or providers), each adapter category should have its own Factory in its `__init__.py`. The main service factory uses these sub-factories rather than inline helper functions.

```python
# Bad - Helper functions scattered in main factory
class RAGServiceFactory:
    @staticmethod
    def create(...) -> RAGService:
        embedder = _create_embedder(embedder_type, settings)  # ❌ Helper function
        vector_store = _create_vector_store(store_type, settings)  # ❌ Helper function
        ...

def _create_embedder(embedder_type, settings):  # ❌ Module-level helper
    match embedder_type:
        case EmbedderType.OPENAI: ...
        case EmbedderType.FASTEMBED: ...

# Good - Sub-factories in each adapter module
# In adapters/embedders/__init__.py:
class EmbedderFactory:
    """Factory for creating Embedder instances."""
    
    @staticmethod
    def create(
        embedder_type: EmbedderType,
        settings: EmbedderSettings,
    ) -> Embedder:
        match embedder_type:
            case EmbedderType.OPENAI:
                return OpenAIEmbedder(...)
            case EmbedderType.FASTEMBED:
                return FastEmbedEmbedder(...)
    
    @staticmethod
    def create_mock(dimension: int = 384) -> MockEmbedder:
        return MockEmbedder(dimension=dimension)

# In adapters/vector_stores/__init__.py:
class VectorStoreFactory:
    """Factory for creating VectorStore instances."""
    
    @staticmethod
    def create(
        store_type: VectorStoreType,
        settings: VectorStoreSettings,
    ) -> VectorStore:
        match store_type:
            case VectorStoreType.MEMORY:
                return InMemoryVectorStore()
            case VectorStoreType.CHROMA:
                return ChromaVectorStore(...)

# Main factory uses sub-factories:
class RAGServiceFactory:
    @staticmethod
    def create(...) -> RAGService:
        embedder = EmbedderFactory.create(embedder_type, embedder_settings)
        vector_store = VectorStoreFactory.create(store_type, store_settings)
        ...
```

**Why sub-factories:**
- Keeps adapter creation logic with the adapters (single responsibility)
- Each factory is independently testable
- Main factory stays clean and focused on composition
- New adapter types only require changes in one place
- Consistent with axiompy patterns (each module owns its factory)

**Examples from axiompy:**
```python
# RAG module uses sub-factories
from axiompy.agents.rag import (
    RAGServiceFactory,      # Main factory
    EmbedderFactory,        # Sub-factory for embedders
    VectorStoreFactory,     # Sub-factory for vector stores
    ChunkerFactory,         # Sub-factory for chunkers
)

# Create embedder directly
embedder = EmbedderFactory.create(
    EmbedderType.FASTEMBED,
    EmbedderSettings(model="BAAI/bge-small-en-v1.5")
)

# Or let RAGServiceFactory compose everything
rag = RAGServiceFactory.create(
    embedder_type=EmbedderType.FASTEMBED,
    vector_store_type=VectorStoreType.MEMORY,
    llm_provider="ollama",
)
```

**Explicit Dependency Injection (REQUIRED)**

Factories must require all dependencies to be explicitly passed in. Do NOT create `create_from_env()` methods that hide configuration behind environment variables.

```python
# Bad - Hidden configuration from environment
class MyServiceFactory:
    @staticmethod
    def create_from_env() -> MyService:
        """Creates service from environment variables."""
        api_key = os.environ.get("API_KEY")  # Hidden dependency!
        return MyService(api_key=api_key)

# Good - All dependencies explicit
class MyServiceFactory:
    @staticmethod
    def create(
        api_key: str,
        timeout: int = 30,
    ) -> MyService:
        """All configuration must be passed explicitly."""
        return MyService(api_key=api_key, timeout=timeout)

# If needed, create a SEPARATE utility to load config from env
# This keeps the factory clean and testable
def load_config_from_env() -> dict:
    """Utility to load config from environment (not a factory method)."""
    return {
        "api_key": os.environ["API_KEY"],
        "timeout": int(os.environ.get("TIMEOUT", "30")),
    }

# Usage - configuration is visible at call site
config = load_config_from_env()
service = MyServiceFactory.create(**config)
```

**Why explicit injection**:
- Makes dependencies visible at the call site
- Easier to test (no environment mocking needed)
- Configuration errors surface immediately
- Clear contract for what the factory needs

### 2. Settings Dataclass Pattern (REQUIRED for configuration)
Use dataclasses with validation in `__post_init__`. Settings are passed explicitly to factories.

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from axiompy.validators import ensure_url, ensure_in_range

@dataclass
class MyServiceSettings:
    """
    Configuration for MyService.
    
    Attributes:
        url: Service endpoint URL
        timeout_secs: Request timeout in seconds (default: 30)
        extra_params: Additional parameters
    """
    url: str
    timeout_secs: int = 30
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate settings after initialization."""
        # Let validators throw directly - don't catch and re-raise
        ensure_url(self.url, "Invalid service URL format")
        ensure_in_range(self.timeout_secs, 1, 3600, "Timeout must be 1-3600 seconds")
```

**Factories accept Settings objects (REQUIRED)**

Factories must accept Settings dataclasses as parameters. This keeps configuration explicit and testable:

```python
class MyServiceFactory:
    @staticmethod
    def create(
        service_type: ServiceType,
        settings: MyServiceSettings,  # Settings passed explicitly
        optional_settings: Optional[OtherSettings] = None,
    ) -> MyService:
        """Create service with explicit settings."""
        return MyService(
            url=settings.url,
            timeout=settings.timeout_secs,
        )

# Usage - configuration is visible at call site
settings = MyServiceSettings(url="http://localhost:8080", timeout_secs=60)
service = MyServiceFactory.create(ServiceType.DEFAULT, settings)
```

**Examples from axiompy:**
```python
# Database
db_settings = DatabaseSettings(host="localhost", port=5432)
db = DatabaseFactory.create(DatabaseType.POSTGRES, db_settings)

# RAG
embedder_settings = EmbedderSettings(model="all-MiniLM-L6-v2", cache_dir="./models")
rag = RAGServiceFactory.create(
    embedder_type=EmbedderType.SENTENCE_TRANSFORMERS,
    embedder_settings=embedder_settings,
    ...
)
```

### 3. Fluent API Pattern (for configuration methods)
Methods that configure an object should return `self` for chaining:

```python
def add_header(self, key: str, value: str) -> "MyClient":
    """
    Add a header.
    
    Args:
        key: Header name
        value: Header value
        
    Returns:
        Self for method chaining
    """
    self._headers[key] = value
    return self

# Usage enables chaining:
client = (
    MyClientFactory.create("http://example.com")
    .add_header("X-Custom", "value")
    .bearer_token("token")
)
```

### 4. Error Hierarchy Pattern
Create exception hierarchies with a base error class:

```python
class MyServiceError(Exception):
    """Base exception for MyService errors."""
    pass

class MyServiceConnectionError(MyServiceError):
    """Connection failure."""
    pass

class MyServiceValidationError(MyServiceError):
    """Validation failure."""
    pass
```

### 5. Composition Over Inheritance
Prefer composition (wrapping) over inheritance:

```python
# Good - Composition
class JSONRPCClient:
    def __init__(self, settings):
        self._http_client = HTTPClientFactory.create(...)  # Has-a relationship
    
    def bearer_token(self, token: str) -> "JSONRPCClient":
        self._http_client.bearer_token(token)  # Delegates
        return self

# Bad - Inheritance (unless truly is-a relationship)
class JSONRPCClient(HTTPClient):  # JSONRPCClient is NOT an HTTPClient
    pass
```

### 6. Mock Classes for Testing
Provide mock implementations in the same module:

```python
class MockMyService(MyService):
    """Mock implementation for unit testing."""
    
    def __init__(self):
        self.calls: List[Tuple[str, Any]] = []
        self._responses: Dict[str, Any] = {}
    
    def set_response(self, method: str, result: Any) -> "MockMyService":
        """Set predefined response."""
        self._responses[method] = result
        return self
    
    def reset(self) -> None:
        """Reset recorded calls."""
        self.calls.clear()
```

### 7. HTTPClient Pattern (REQUIRED for external HTTP calls)
Use `axiompy.io.http.HTTPClientFactory` for all HTTP requests instead of raw `requests`:

```python
# Bad - Using raw requests
import requests

def call_api(url: str, token: str) -> dict:
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={"data": "value"},
        timeout=30,
    )
    return response.json()

# Good - Using axiompy HTTPClient
from axiompy.io.http import HTTPClientFactory

def call_api(url: str, token: str) -> dict:
    client = (
        HTTPClientFactory.create(timeout_secs=30)
        .bearer_token(token)
        .add_header("Content-Type", "application/json")
    )
    response = client.post(url, json={"data": "value"})
    return response.json()
```

**Benefits**:
- Consistent retry logic and error handling
- Built-in logging for debugging
- Fluent API for authentication and headers
- Automatic session management

**When to use raw requests**:
- Streaming responses (`iter_lines`) - HTTPClient doesn't support streaming yet
- WebSocket connections
- Very low-level HTTP control needed

### 8. Match/Case Pattern (REQUIRED for type/value dispatch)
Use `match/case` instead of `if/elif` chains for dispatching on types or values:

```python
# Bad - Long if/elif chain
def process(event_type: str) -> None:
    if event_type == "click":
        handle_click()
    elif event_type == "scroll":
        handle_scroll()
    elif event_type == "submit":
        handle_submit()
    else:
        handle_unknown()

# Good - Match/case is clearer
def process(event_type: str) -> None:
    match event_type:
        case "click":
            handle_click()
        case "scroll":
            handle_scroll()
        case "submit":
            handle_submit()
        case _:
            handle_unknown()

# Good - Match/case with pattern guards
def get_verdict(files_reviewed: int, score: int) -> str:
    match (files_reviewed, score):
        case (0, _):
            return "Review Failed"
        case (_, s) if s >= 90:
            return "Excellent"
        case (_, s) if s >= 70:
            return "Good"
        case _:
            return "Needs Work"
```

**Benefits**:
- More readable than `if/elif` chains
- Exhaustiveness checking with `case _:`
- Pattern guards for complex conditions
- Destructuring for tuple/object matching

## Design Principles

### 8. Rule of Three (WARNING)
Don't create abstractions until you have at least 3 concrete use cases.

**Why**: Premature abstraction leads to wrong abstractions. Wait until patterns emerge naturally from real usage.

```python
# Bad - Abstracting too early (only one implementation exists)
class AbstractNotificationSender(ABC):
    """We might need this someday..."""
    @abstractmethod
    def send(self, message: str) -> None: ...

class EmailNotificationSender(AbstractNotificationSender):
    def send(self, message: str) -> None: ...

# Good - Wait for 3 use cases before abstracting
# After you have:
#   1. EmailSender
#   2. SMSSender  
#   3. PushNotificationSender
# NOW create the abstraction:
class NotificationSender(Protocol):
    def send(self, message: str) -> None: ...
```

**When to abstract**:
- You have 3+ concrete implementations
- The pattern has proven itself in production
- The abstraction simplifies, not complicates

### 9. SOLID Principles

#### Single Responsibility Principle (SRP) (WARNING)
A class should have only one reason to change.

```python
# Bad - Multiple responsibilities
class UserService:
    def create_user(self, name: str) -> User: ...
    def send_welcome_email(self, user: User) -> None: ...  # Email is separate concern
    def generate_report(self, user: User) -> str: ...      # Reporting is separate concern

# Good - Single responsibility per class
class UserService:
    def create_user(self, name: str) -> User: ...

class EmailService:
    def send_welcome_email(self, user: User) -> None: ...

class ReportService:
    def generate_user_report(self, user: User) -> str: ...
```

#### Open/Closed Principle (OCP) (INFO)
Open for extension, closed for modification.

```python
# Bad - Must modify class to add new behavior
class PaymentProcessor:
    def process(self, payment_type: str, amount: float) -> bool:
        if payment_type == "credit":
            return self._process_credit(amount)
        elif payment_type == "debit":
            return self._process_debit(amount)
        # Must add elif for each new type!

# Good - Extend without modifying
class PaymentProcessor(Protocol):
    def process(self, amount: float) -> bool: ...

class CreditPaymentProcessor:
    def process(self, amount: float) -> bool: ...

class DebitPaymentProcessor:
    def process(self, amount: float) -> bool: ...
```

#### Liskov Substitution Principle (LSP) (WARNING)
Subtypes must be substitutable for their base types without breaking behavior.

```python
# Bad - Subtype changes expected behavior
class Rectangle:
    def set_width(self, width: int) -> None:
        self._width = width
    
    def set_height(self, height: int) -> None:
        self._height = height

class Square(Rectangle):  # Violates LSP!
    def set_width(self, width: int) -> None:
        self._width = width
        self._height = width  # Unexpected side effect
    
    def set_height(self, height: int) -> None:
        self._height = height
        self._width = height  # Unexpected side effect

# Good - Don't inherit if behavior differs
class Shape(Protocol):
    def area(self) -> float: ...

class Rectangle:
    def __init__(self, width: float, height: float): ...
    def area(self) -> float: return self._width * self._height

class Square:
    def __init__(self, side: float): ...
    def area(self) -> float: return self._side ** 2
```

#### Interface Segregation Principle (ISP) (INFO)
Prefer small, focused interfaces over large ones.

```python
# Bad - Fat interface forces unnecessary implementations
class Worker(ABC):
    @abstractmethod
    def work(self) -> None: ...
    @abstractmethod
    def eat(self) -> None: ...
    @abstractmethod
    def sleep(self) -> None: ...

class Robot(Worker):  # Robots don't eat or sleep!
    def work(self) -> None: ...
    def eat(self) -> None: pass  # Forced to implement
    def sleep(self) -> None: pass  # Forced to implement

# Good - Segregated interfaces
class Workable(Protocol):
    def work(self) -> None: ...

class Eatable(Protocol):
    def eat(self) -> None: ...

class Robot:  # Only implements what it needs
    def work(self) -> None: ...
```

#### Dependency Inversion Principle (DIP) (WARNING)

> *"Depend on abstractions, not on concretions."*
> — Robert C. Martin (Uncle Bob)

This is the core principle that enables testability, flexibility, and maintainability. High-level modules should not depend on low-level modules; both should depend on abstractions.

```python
# Bad - Depends on concrete class
class OrderService:
    def __init__(self):
        self.db = PostgresDatabase()  # Concrete dependency
        self.emailer = SMTPEmailer()  # Concrete dependency

# Good - Depends on abstractions (use Factory pattern)
class OrderService:
    def __init__(self, db: Database, emailer: Emailer):
        self.db = db
        self.emailer = emailer

# Even better - Use axiompy Factory pattern
order_service = OrderServiceFactory.create(
    db=DatabaseFactory.create(DatabaseType.POSTGRES, settings),
    emailer=EmailerFactory.create(EmailerType.SMTP, settings),
)
```

## Anti-Patterns

Patterns to **avoid**. The code review agent flags these as violations.

### God Class (ERROR)
A class that does too much and knows too much.

**Indicators**:
- More than 500 lines
- More than 10 public methods
- Depends on many other classes
- Name includes "Manager", "Handler", "Processor", "Utils" (sometimes)

```python
# Bad - God class doing everything
class ApplicationManager:
    def handle_user_login(self) -> None: ...
    def process_payment(self) -> None: ...
    def send_notification(self) -> None: ...
    def generate_report(self) -> None: ...
    def update_inventory(self) -> None: ...
    def calculate_shipping(self) -> None: ...
    def apply_discount(self) -> None: ...
    def validate_address(self) -> None: ...
    # ... 50 more methods

# Good - Separated responsibilities
class AuthService: ...
class PaymentService: ...
class NotificationService: ...
class ReportService: ...
class InventoryService: ...
```

**How to fix**: Extract Method, Extract Class, apply SRP.

### Speculative Generality (WARNING)
Creating abstractions "just in case" they're needed later. Violates Rule of Three.

```python
# Bad - YAGNI violation (You Aren't Gonna Need It)
class AbstractFutureProofHandler(ABC):
    """We might need multiple handlers someday..."""
    @abstractmethod
    def handle(self) -> None: ...

class ConcreteHandler(AbstractFutureProofHandler):
    def handle(self) -> None: ...
    
# Only one implementation exists - the abstraction adds no value!

# Good - Build what you need now
class Handler:
    """Handles the actual current requirement."""
    def handle(self) -> None: ...

# Create abstraction later when you have 3+ implementations
```

**How to fix**: Remove unused abstractions. Follow Rule of Three.

### Inappropriate Intimacy (WARNING)
Classes that are too dependent on each other's internals.

```python
# Bad - Class reaches into another's private details
class Order:
    def calculate_total(self) -> float:
        # Reaching into Customer's internals
        if self.customer._loyalty_points > 1000:  # Accessing private!
            discount = self.customer._calculate_internal_discount()
        return self._subtotal - discount

# Good - Ask, don't tell
class Order:
    def calculate_total(self) -> float:
        discount = self.customer.get_discount()  # Public method
        return self._subtotal - discount

class Customer:
    def get_discount(self) -> float:
        """Encapsulates discount logic."""
        if self._loyalty_points > 1000:
            return self._calculate_internal_discount()
        return 0.0
```

**How to fix**: Encapsulate internals, use public interfaces.

### Singleton (WARNING)
Using the Singleton pattern to ensure only one instance exists. Often misused as a global variable in disguise.

**Why it's a problem**:
- Hidden global state (same issues as global variables)
- Makes unit testing difficult (can't substitute instances)
- Creates tight coupling across the codebase
- Thread-safety issues in concurrent code

```python
# Bad - Classic Singleton anti-pattern
class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def query(self, sql: str) -> list:
        ...

# Usage - hidden global state!
db = DatabaseConnection()  # Always returns same instance

# Bad - Module-level singleton
_db_instance = None

def get_db():
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection()
    return _db_instance

# Good - Use dependency injection with Factory pattern
class DatabaseConnectionFactory:
    @staticmethod
    def create(settings: DatabaseSettings) -> DatabaseConnection:
        return DatabaseConnection(settings)

class MyService:
    def __init__(self, db: DatabaseConnection):  # Injected dependency
        self._db = db

# In application setup (composition root)
db = DatabaseConnectionFactory.create(settings)
service = MyService(db)

# Good - If you need shared state, use app.state or explicit container
from axiompy.servers import ServerFactory, ServerType, ServerSettings

server = ServerFactory.create(ServerType.FASTAPI, ServerSettings())
app = server.get_app()
app.state.db = DatabaseConnectionFactory.create(settings)  # Explicit, testable
```

**When Singleton might be acceptable**:
- Logging (stateless, no side effects)
- Configuration (read-only after initialization)
- Hardware resource access (but prefer Factory + injection)

**How to fix**: Use Factory pattern + dependency injection. Pass dependencies explicitly.

## Code Smells

Symptoms that may indicate deeper problems. Not always wrong, but worth reviewing.

### Long Method (WARNING)
Methods exceeding 50 lines are harder to understand, test, and maintain.

```python
# Bad - Method doing too much
def process_order(self, order: Order) -> None:
    # Validate order (20 lines)
    # Calculate totals (15 lines)
    # Apply discounts (15 lines)
    # Update inventory (10 lines)
    # Send notifications (10 lines)
    # ... 70+ lines total

# Good - Extract methods
def process_order(self, order: Order) -> None:
    self._validate_order(order)
    total = self._calculate_total(order)
    total = self._apply_discounts(order, total)
    self._update_inventory(order)
    self._send_notifications(order)
```

**How to fix**: Extract Method, Compose Method.

### Deep Nesting (WARNING)
More than 4 levels of indentation indicates complex logic that's hard to follow.

```python
# Bad - Deep nesting (5+ levels)
def process(self, data: list) -> None:
    if data:
        for item in data:
            if item.valid:
                for sub in item.children:
                    if sub.active:
                        if sub.ready:  # 5 levels deep!
                            self._handle(sub)

# Good - Early returns and extracted methods
def process(self, data: list) -> None:
    if not data:
        return
    
    for item in data:
        self._process_item(item)

def _process_item(self, item: Item) -> None:
    if not item.valid:
        return
    
    for sub in item.children:
        self._process_subitem(sub)

def _process_subitem(self, sub: SubItem) -> None:
    if sub.active and sub.ready:
        self._handle(sub)
```

**How to fix**: Guard clauses (early returns), Extract Method.

### Magic Numbers (WARNING)
Hardcoded numeric values without explanation.

```python
# Bad - What do these numbers mean?
if retry_count > 3:
    if timeout > 30:
        if response.status_code == 429:
            time.sleep(60)

# Good - Named constants explain intent
MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECS = 30
RATE_LIMIT_STATUS = 429
RATE_LIMIT_BACKOFF_SECS = 60

if retry_count > MAX_RETRIES:
    if timeout > DEFAULT_TIMEOUT_SECS:
        if response.status_code == RATE_LIMIT_STATUS:
            time.sleep(RATE_LIMIT_BACKOFF_SECS)
```

**How to fix**: Replace Magic Number with Named Constant.

### Copy-Paste Code (ERROR)
Duplicated code blocks that should be extracted into reusable functions.

```python
# Bad - Same code in multiple places
class UserService:
    def create_user(self, data: dict) -> User:
        # Validation logic (10 lines)
        if not data.get("email"):
            raise ValueError("Email required")
        if not data.get("name"):
            raise ValueError("Name required")
        # ... more validation
        
class AdminService:
    def create_admin(self, data: dict) -> Admin:
        # Same validation logic copy-pasted!
        if not data.get("email"):
            raise ValueError("Email required")
        if not data.get("name"):
            raise ValueError("Name required")
        # ... same validation

# Good - Extract common code
def validate_user_data(data: dict) -> None:
    """Reusable validation."""
    ensure_not_empty(data.get("email"), "Email required")
    ensure_not_empty(data.get("name"), "Name required")

class UserService:
    def create_user(self, data: dict) -> User:
        validate_user_data(data)
        # ...

class AdminService:
    def create_admin(self, data: dict) -> Admin:
        validate_user_data(data)
        # ...
```

**How to fix**: Extract Method, Extract Class, create utilities.

### Dead Code (WARNING)
Unreachable code, unused variables, commented-out code.

```python
# Bad - Dead code cluttering the codebase
def process(self, data: dict) -> None:
    result = self._transform(data)
    # old_result = self._old_transform(data)  # Commented out code
    # if False:  # Dead branch
    #     self._never_called()
    unused_variable = "this is never used"
    return result

# Good - Remove dead code (it's in version control if needed)
def process(self, data: dict) -> None:
    result = self._transform(data)
    return result
```

**How to fix**: Delete it. Use version control to recover if needed.

### Global Variables (WARNING)
Using `global` keyword or module-level mutable state that can be modified.

**Why it's a problem**:
- Makes code harder to test (hidden dependencies)
- Creates implicit coupling between functions
- Can cause race conditions in concurrent code
- Makes behavior unpredictable

```python
# Bad - Global mutable state
_cache = {}  # Module-level mutable state
_app = None  # Global variable

def get_app():
    global _app
    if _app is None:
        _app = create_app()
    return _app

def get_cached(key: str) -> Any:
    global _cache
    return _cache.get(key)

# Good - Use dependency injection or class-based approach
class AppFactory:
    """Factory with instance-level state."""
    
    @staticmethod
    def create() -> App:
        return App()

class CacheService:
    """Service with encapsulated state."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    def get(self, key: str) -> Any:
        return self._cache.get(key)

# Good - Use FastAPI/Flask app.state for request-scoped state
app.state.cache = CacheService()

# Good - Use axiompy.servers.ServerFactory
from axiompy.servers import ServerFactory, ServerType, ServerSettings

server = ServerFactory.create(ServerType.FASTAPI, ServerSettings())
```

**How to fix**: 
- Use dependency injection
- Use class-based encapsulation
- Use framework-provided state management (app.state)
- Use factories (ServerFactory, etc.)

### Hardcoded Credentials (CRITICAL)
Secrets, API keys, passwords in source code. **Security risk!**

```python
# CRITICAL - Never do this!
api_key = "sk-1234567890abcdef"
password = "admin123"
database_url = "postgres://user:password@localhost/db"

# Good - Use environment variables
import os
api_key = os.environ["API_KEY"]

# Better - Use axiompy secrets manager
from axiompy.secrets import SecretsFactory

secrets = SecretsFactory.create_from_env()
api_key = secrets.get("API_KEY")
database_url = secrets.get("DATABASE_URL")
```

**How to fix**: Use environment variables or secrets manager. **Never commit secrets.**

### Missing Error Handling (ERROR)
Bare except clauses, swallowed exceptions, missing error handling.

```python
# Bad - Swallowing exceptions hides bugs
try:
    result = risky_operation()
except:  # Bare except catches everything including KeyboardInterrupt!
    pass  # Silently swallowed - bugs will be hidden

# Bad - Catching too broadly
try:
    result = api_call()
except Exception:
    return None  # What went wrong? We'll never know

# Good - Specific exception handling with logging
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

try:
    result = api_call()
except ConnectionError as e:
    logger.error(f"API connection failed: {e}")
    raise
except ValidationError as e:
    logger.warning(f"Invalid input: {e}")
    return None
```

**How to fix**: Catch specific exceptions, log errors, re-raise or handle appropriately.

### Primitive Obsession (WARNING)
Using primitive types (str, int, float) instead of small domain objects.

```python
# Bad - Primitives everywhere
def create_user(
    email: str,           # Just a string?
    phone: str,           # What format?
    zip_code: str,        # Any validation?
    age: int,             # Can be negative?
) -> User:
    # Validation scattered everywhere it's used
    if "@" not in email:
        raise ValueError("Invalid email")
    if len(phone) != 10:
        raise ValueError("Invalid phone")
    ...

# Good - Domain objects encapsulate validation
@dataclass
class Email:
    value: str
    
    def __post_init__(self):
        if "@" not in self.value or "." not in self.value:
            raise ValueError(f"Invalid email: {self.value}")

@dataclass
class PhoneNumber:
    value: str
    
    def __post_init__(self):
        digits = "".join(c for c in self.value if c.isdigit())
        if len(digits) != 10:
            raise ValueError(f"Invalid phone: {self.value}")
        self.value = digits

def create_user(email: Email, phone: PhoneNumber, ...) -> User:
    # No validation needed - types guarantee validity
    ...
```

**How to fix**: Create small value objects for domain concepts. Validation happens once at construction.

### Feature Envy (WARNING)
A method that uses another class's data more than its own.

```python
# Bad - Order method is "envious" of Customer's data
class Order:
    def calculate_discount(self) -> float:
        # Uses customer's data extensively
        if self.customer.loyalty_tier == "gold":
            if self.customer.years_as_member > 5:
                if self.customer.total_purchases > 10000:
                    return 0.20
                return 0.15
            return 0.10
        elif self.customer.loyalty_tier == "silver":
            return 0.05
        return 0.0

# Good - Move method to the class whose data it uses
class Customer:
    def get_discount_rate(self) -> float:
        """Customer knows its own discount rules."""
        if self.loyalty_tier == "gold":
            if self.years_as_member > 5:
                if self.total_purchases > 10000:
                    return 0.20
                return 0.15
            return 0.10
        elif self.loyalty_tier == "silver":
            return 0.05
        return 0.0

class Order:
    def calculate_discount(self) -> float:
        return self.subtotal * self.customer.get_discount_rate()
```

**How to fix**: Move the method to the class whose data it uses most.

### Data Clumps (INFO)
Groups of data that appear together repeatedly across the codebase.

```python
# Bad - Same parameters appear together everywhere
def send_email(recipient_name: str, recipient_email: str, recipient_phone: str):
    ...

def create_invoice(customer_name: str, customer_email: str, customer_phone: str):
    ...

def schedule_delivery(contact_name: str, contact_email: str, contact_phone: str):
    ...

# Good - Extract a class for the data clump
@dataclass
class ContactInfo:
    name: str
    email: str
    phone: str

def send_email(recipient: ContactInfo):
    ...

def create_invoice(customer: ContactInfo):
    ...

def schedule_delivery(contact: ContactInfo):
    ...
```

**How to fix**: Extract a dataclass for data that travels together.

### Long Parameter List (WARNING)
Functions with too many parameters (>5) are hard to use and understand.

```python
# Bad - Too many parameters
def create_order(
    customer_id: int,
    customer_name: str,
    product_id: int,
    product_name: str,
    quantity: int,
    unit_price: float,
    discount_percent: float,
    tax_rate: float,
    shipping_method: str,
    shipping_address: str,
    billing_address: str,
    notes: str,
) -> Order:
    ...

# Good - Group related parameters into objects
@dataclass
class CustomerInfo:
    id: int
    name: str

@dataclass
class ProductInfo:
    id: int
    name: str
    unit_price: float

@dataclass
class OrderDetails:
    quantity: int
    discount_percent: float = 0.0
    notes: str = ""

@dataclass
class ShippingInfo:
    method: str
    address: str
    billing_address: str

def create_order(
    customer: CustomerInfo,
    product: ProductInfo,
    details: OrderDetails,
    shipping: ShippingInfo,
) -> Order:
    ...
```

**How to fix**: 
- Group related parameters into dataclasses
- Use Settings/Config objects
- Consider Builder pattern for complex construction

### Kwargs Anti-Pattern (WARNING)
Using `**kwargs` hides parameters and breaks IDE autocompletion/type checking.

```python
# Bad - Hidden parameters, no type hints
def create_service(
    source_type: str = "filesystem",
    **kwargs,  # What parameters are valid? No way to know!
) -> Service:
    host = kwargs.get("host", "localhost")  # Hidden parameter
    port = kwargs.get("port", 8080)          # Hidden parameter
    timeout = kwargs.get("timeout", 30)      # Hidden parameter
    ...

# Using it - no IDE help, easy to typo
service = create_service("filesystem", hoost="x")  # Typo not caught!

# Good - Explicit parameters with types
def create_service(
    source_type: str = "filesystem",
    host: str = "localhost",
    port: int = 8080,
    timeout: int = 30,
) -> Service:
    ...

# Using it - IDE shows all options, typos caught
service = create_service("filesystem", host="x")

# Better - Use a Settings object
@dataclass
class ServiceSettings:
    host: str = "localhost"
    port: int = 8080
    timeout: int = 30

def create_service(
    source_type: str = "filesystem",
    settings: Optional[ServiceSettings] = None,
) -> Service:
    settings = settings or ServiceSettings()
    ...
```

**Why it's bad**:
- No IDE autocompletion for valid parameters
- No type checking on parameter values
- Easy to typo parameter names without errors
- Documentation becomes essential (and often missing)

**How to fix**:
- Use explicit typed parameters
- Group related options into Settings/Config dataclasses
- Pass the settings object instead of individual values

### Shotgun Surgery (WARNING)
One logical change requires editing many different classes/files.

```python
# Bad - Adding a new payment type requires changes everywhere
# file: payment_processor.py
def process(payment_type: str) -> bool:
    if payment_type == "credit":
        ...
    elif payment_type == "debit":
        ...
    # Must add new elif here

# file: payment_validator.py
def validate(payment_type: str) -> bool:
    if payment_type == "credit":
        ...
    elif payment_type == "debit":
        ...
    # Must add new elif here too

# file: payment_logger.py
def log_payment(payment_type: str) -> None:
    if payment_type == "credit":
        ...
    elif payment_type == "debit":
        ...
    # And here...

# file: payment_report.py
# ... and so on in 10 more files

# Good - Encapsulate variation with polymorphism
class PaymentMethod(Protocol):
    def process(self) -> bool: ...
    def validate(self) -> bool: ...
    def log(self) -> None: ...
    def report_data(self) -> dict: ...

class CreditPayment:
    def process(self) -> bool: ...
    def validate(self) -> bool: ...
    def log(self) -> None: ...
    def report_data(self) -> dict: ...

class DebitPayment:
    def process(self) -> bool: ...
    def validate(self) -> bool: ...
    def log(self) -> None: ...
    def report_data(self) -> dict: ...

# Adding new payment type = ONE new class, no changes to existing code
class CryptoPayment:
    def process(self) -> bool: ...
    def validate(self) -> bool: ...
    def log(self) -> None: ...
    def report_data(self) -> dict: ...
```

**How to fix**: Apply Open/Closed Principle. Use polymorphism to encapsulate variation.

## Validation

### Use axiompy.validators for Input Validation
Always validate inputs at public API boundaries. **Let validators throw directly** - don't catch and re-raise:

```python
from axiompy.validators import (
    ensure_not_none,
    ensure_not_empty,
    ensure_url,
    ensure_in_range,
    ensure_type,
)

def my_method(self, url: str, timeout: int) -> None:
    # Let validators throw ValidationError directly
    # Don't catch and re-raise - it adds noise without value
    ensure_url(url, "Invalid URL format")
    ensure_in_range(timeout, 1, 3600, "Timeout out of range")
```

**Why let validators throw:**
- Catching an exception you're about to re-raise adds no value
- ValidationError already contains the message you need
- Callers can catch ValidationError at the appropriate level
- Cleaner, more readable code

## Decorators

### Use axiompy.decorators for Cross-Cutting Concerns

```python
from axiompy.decorators import LogExecutionTime, CatchAndLog, Retry
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

@LogExecutionTime(logger)
def timed_operation(self):
    """This method's execution time will be logged at DEBUG level."""
    pass

@Retry(logger, max_attempts=3, delay=1.0)
def flaky_operation(self):
    """This method will retry on failure."""
    pass

@CatchAndLog(logger, reraise=False, default_return=None)
def safe_operation(self):
    """This method catches exceptions and returns default."""
    pass
```

## Logging

### Use LoggerFactory for All Logging

```python
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

# In class __init__:
logger.info(f"MyService initialized for {settings.url}")

# Debug for internal details:
logger.debug(f"Processing request: {request_id}")

# Warning for recoverable issues:
logger.warning(f"Retrying after error: {e}")
```

## Documentation

### Docstring Format (Google Style)
Every public class, method, and function must have docstrings:

```python
def my_function(
    param1: str,
    param2: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Brief one-line description.
    
    Longer description if needed, explaining behavior,
    side effects, or important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: None)
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        ConnectionError: When connection fails
        
    Example:
        >>> result = my_function("test", param2=10)
        >>> print(result)
        {'status': 'ok'}
    """
```

### Module Docstrings
Every module should have a comprehensive docstring:

```python
"""
Brief module description.

Provides [main functionality] with support for:
    - Feature 1
    - Feature 2
    - Feature 3

Key Benefits:
    - Benefit 1
    - Benefit 2

Quick Example:
    >>> from axiompy.module import Factory
    >>> client = Factory.create(...)
    >>> result = client.do_something()

For comprehensive examples, see:
    - examples/module_examples.py
    - tests/test_module.py
"""
```

## Testing

### Test File Structure
- Test files: `tests/test_<module_name>.py`
- Use pytest with fixtures
- Aim for 80%+ coverage
- Group tests by class

```python
import pytest
from unittest.mock import Mock, patch

class TestMyServiceSettings:
    """Tests for MyServiceSettings configuration."""
    
    def test_valid_settings(self):
        """Test creating settings with valid parameters."""
        settings = MyServiceSettings(url="http://localhost", timeout_secs=30)
        assert settings.url == "http://localhost"
    
    def test_invalid_url_raises(self):
        """Test that invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid"):
            MyServiceSettings(url="not-a-url")

class TestMyService:
    """Tests for MyService."""
    
    @pytest.fixture
    def mock_dependency(self):
        """Create mock dependency."""
        with patch("axiompy.module.DependencyFactory.create") as mock:
            yield mock.return_value
    
    @pytest.fixture
    def service(self, mock_dependency):
        """Create service with mocked dependency."""
        return MyServiceFactory.create("http://localhost")
```

### Mock Client Testing Pattern

```python
def test_with_mock_client():
    """Test using mock client."""
    mock = MyClientFactory.create_mock()
    mock.set_response("method_name", {"result": "value"})
    
    result = mock.call("method_name", {"param": "test"})
    
    assert result == {"result": "value"}
    assert mock.calls == [("method_name", {"param": "test"})]
```

## File Organization

```
axiompy/
├── __init__.py
├── module_name/
│   ├── __init__.py      # Exports public API
│   ├── implementation.py # Main implementation
│   ├── types.py         # Type definitions, enums, dataclasses
│   ├── factory.py       # Factory classes (or in implementation.py)
│   └── README.md        # Module documentation
```

## README Documentation Standards

### Module README Structure (REQUIRED)
Every module must have a README.md following this structure:

```markdown
# Module Name

Brief one-line description of what the module does.

## Overview / Features

The `axiompy.module` provides [functionality] with support for:
- ✅ **Feature 1**: Brief description
- ✅ **Feature 2**: Brief description
- 🔐 **Feature 3**: Use emojis for visual organization

## Table of Contents (for long READMEs)

- [Quick Start](#quick-start)
- [Concepts](#concepts)
- [API Reference](#api-reference)
- [Examples](#examples)

## Installation (if has dependencies)

```bash
pip install axiompy[module]
```

## Quick Start

Show the simplest working example first:

```python
from axiompy.module import Factory

# 3-5 lines of working code
client = Factory.create("http://example.com")
result = client.do_something()
```

## Key Concepts / Architecture

Explain the design with diagrams if helpful:

```
┌─────────────────────┐
│   Component A       │
└─────────────────────┘
          ↓
┌─────────────────────┐
│   Component B       │
└─────────────────────┘
```

## API Reference

### Factory

| Method | Description | Returns |
|--------|-------------|---------|
| `create()` | Create instance | Client |
| `create_mock()` | Create mock | MockClient |

### Client Methods

Document each public method with signature and description.

## Examples

### Example 1: Common Use Case
```python
# Complete working example
```

### Example 2: Advanced Use Case
```python
# More complex example
```

## Error Handling

```python
from axiompy.module import ModuleError, ConnectionError

try:
    result = client.operation()
except ConnectionError as e:
    # Handle connection issues
except ModuleError as e:
    # Handle other errors
```

## Testing

```bash
pytest tests/test_module.py -v --cov=axiompy.module
```

## Best Practices

1. **Practice 1**: Explanation
2. **Practice 2**: Explanation
```

### README Content Patterns

#### 1. Before/After Comparisons (for transformative features)
```markdown
### Before: The Hard Way
```python
# 20 lines of boilerplate
```

### After: With AxiomPy
```python
# 3 lines of clean code
```
```

#### 2. Quick Reference Tables
```markdown
| Method | Description | Returns |
|--------|-------------|---------|
| `call()` | Make RPC call | Result |
| `notify()` | Fire-and-forget | None |
```

#### 3. Feature Checklists with Emojis
```markdown
- ✅ **Completed Feature**: Description
- 🚧 **In Progress**: Description  
- 📋 **Planned**: Description
```

#### 4. Architecture Diagrams (ASCII)
```markdown
```
User Request
     ↓
┌─────────────────┐
│   Controller    │
└─────────────────┘
     ↓
┌─────────────────┐
│    Service      │
└─────────────────┘
```
```

#### 5. Cross-References
```markdown
> **💡 Tip:** See [`axiompy/other/README.md`](../other/README.md) for related functionality.

For comprehensive examples, see:
- `examples/module_examples.py`
- `tests/test_module.py`
```

### Root README Updates

When adding new features, update root README.md:

1. **Features list** - Add bullet point with brief description
2. **Quick Start section** - Add minimal working example
3. **Coverage table** - Update test coverage metrics
4. **Test structure** - Add new test file

```markdown
### Key Metrics by Folder

| Folder/Module | Coverage | Status |
|---------------|----------|--------|
| `axiompy/io/jsonrpc.py` | 99.63% | ⭐⭐⭐ |

### Test Structure

- `test_module.py` - Module tests (N tests, XX% coverage)
```

## Common Patterns Checklist

When creating a new module or class, ensure:

### Required Patterns
- [ ] Factory class with `create()` and `create_mock()` static methods
- [ ] Settings dataclass with `__post_init__` validation
- [ ] Error hierarchy (BaseError -> SpecificErrors)
- [ ] Fluent API for configuration methods (return self)
- [ ] Mock implementation for testing
- [ ] Input validation using axiompy.validators
- [ ] Logging via LoggerFactory
- [ ] Comprehensive docstrings (Google style)
- [ ] Type hints on all parameters and returns
- [ ] Tests with 80%+ coverage
- [ ] README documentation

### Design Principles
- [ ] Rule of Three: Don't abstract until 3+ use cases exist
- [ ] Single Responsibility: Each class has one reason to change
- [ ] Dependency Inversion: Depend on abstractions (use Factories)
- [ ] Composition over Inheritance: Prefer has-a over is-a

### Avoid These Anti-Patterns
- [ ] No God classes (>500 lines, >10 methods)
- [ ] No speculative generality (unused abstractions)
- [ ] No inappropriate intimacy (accessing private members)
- [ ] No Singletons (use Factory + dependency injection)

### Watch for Code Smells
- [ ] No long methods (>50 lines)
- [ ] No deep nesting (>4 levels)
- [ ] No magic numbers (use named constants)
- [ ] No copy-paste code (extract to functions)
- [ ] No dead code (remove unused code)
- [ ] No global variables (use dependency injection)
- [ ] No hardcoded credentials (use env vars or secrets)
- [ ] No missing error handling (catch specific exceptions)
- [ ] No primitive obsession (use domain objects)
- [ ] No feature envy (move method to data owner)
- [ ] No data clumps (extract dataclass)
- [ ] No long parameter lists (>5 params, use objects)
- [ ] No shotgun surgery (encapsulate variation)

