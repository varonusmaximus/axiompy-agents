# AxiomPy Reasoning Module

**AI-Powered Data Intelligence with Natural Language Query Understanding**

Build production-grade AI services that understand natural language questions, generate SQL, validate queries, and provide intelligent insights—all with a provider-agnostic interface.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Core Components](#core-components)
5. [AI Providers](#ai-providers)
6. [Metadata System](#metadata-system)
7. [Query Agent](#query-agent)
8. [SQL Validation](#sql-validation)
9. [Best Practices](#best-practices)
10. [Examples](#examples)
11. [Troubleshooting](#troubleshooting)
12. [Migration Guide](#migration-guide)

---

## Overview

The `axiompy.reasoning` module enables you to build AI-powered data services that:

- ✅ **Understand natural language** - "What are the top 5 products?" → SQL
- ✅ **Route intelligently** - Automatically select the right dataset
- ✅ **Validate queries** - Prevent LLM hallucinations with schema validation
- ✅ **Generate insights** - Natural language explanations of results
- ✅ **Support multiple providers** - Ollama (local), OpenAI, Anthropic
- ✅ **Scale to production** - 80%+ test coverage, comprehensive error handling

### Why Use This Module?

**Traditional Approach** (200+ lines per service):
```python
# Custom LLM client
# Custom prompt engineering
# Custom SQL generation
# Custom validation
# Custom error handling
# Not reusable across projects
```

**With AxiomPy Reasoning** (25 lines):
```python
from axiompy.reasoning import AIClientFactory, QueryAgent, BaseDatasetService

# Define your dataset
class MyService(BaseDatasetService):
    # ... implement interface ...

# Create AI client and agent
ai = ReasoningFactory.create(ReasoningProvider.OLLAMA)
agent = QueryAgent(ai, {"mydata": MyService(db)})

# Ask questions!
result = agent.execute_query("Show me the top 10 records")
```

**Result:** 88% code reduction, production-ready patterns, reusable across projects.

---

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      User Question                          │
│              "What are the top 5 products?"                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                      QueryAgent                             │
│  (Orchestrates planning → SQL → validation → execution)    │
└─────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────┴──────────────────┐
        ↓                                      ↓
┌──────────────────┐                  ┌──────────────────┐
│   AIClient       │                  │ DatasetService   │
│  (LLM Provider)  │                  │  (Your Data)     │
└──────────────────┘                  └──────────────────┘
        ↓                                      ↓
┌──────────────────┐                  ┌──────────────────┐
│ Provider Config  │                  │ DatasetMetadata  │
│ (Ollama/OpenAI)  │                  │  (Schema Info)   │
└──────────────────┘                  └──────────────────┘
```

### Component Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (Your domain-specific services and business logic)        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                      Agent Layer                            │
│  QueryAgent: Planning, routing, orchestration              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    AI Client Layer                          │
│  AIClient: Provider-agnostic LLM interface                 │
│  Providers: Ollama, OpenAI, Anthropic                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   Service Layer                             │
│  BaseDatasetService: Abstract interface for datasets       │
│  DatasetMetadata: Self-describing schema                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  Database, API, File, Search Engine (via axiompy.io)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
# AxiomPy already includes the reasoning module
pip install git+https://github.com/varonusmaximus/axiompy.git

# For local LLM support (Ollama)
# macOS:
brew install ollama
ollama pull mistral

# Linux:
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
```

### 5-Minute Example

```python
from axiompy.io.database import DatabaseFactory, DatabaseType, DatabaseSettings
from axiompy.reasoning import (
    ReasoningFactory,
    ReasoningProvider,
    QueryAgent,
    BaseDatasetService,
    DatasetMetadata,
    ScopeMetadata,
    TableSchemaMetadata
)

# 1. Define your dataset service
class SalesService(BaseDatasetService):
    dataset_name = "sales"
    description = "Sales transactions database"
    
    def __init__(self, db):
        self.db = db
    
    def query(self, sql: str, limit: int = None) -> list[dict]:
        """Execute SQL query."""
        if limit and "LIMIT" not in sql.upper():
            sql = f"{sql.rstrip(';')} LIMIT {limit}"
        return self.db.execute(sql)
    
    def get_metadata(self) -> DatasetMetadata:
        """Return rich metadata for AI reasoning."""
        return DatasetMetadata(
            dataset="sales",
            description="Sales transactions with products and customers",
            scope=ScopeMetadata(
                geographic="United States",
                temporal="2020-2024",
                domain="E-commerce"
            ),
            schema={
                "orders": TableSchemaMetadata(
                    columns={
                        "order_id": "INTEGER PRIMARY KEY",
                        "customer_id": "INTEGER",
                        "product_id": "INTEGER",
                        "total_amount": "DECIMAL(10,2)",
                        "order_date": "DATE"
                    },
                    description="Customer orders",
                    row_count=1000000
                ),
                "products": TableSchemaMetadata(
                    columns={
                        "product_id": "INTEGER PRIMARY KEY",
                        "name": "TEXT",
                        "category": "TEXT",
                        "price": "DECIMAL(10,2)"
                    },
                    description="Product catalog"
                )
            },
            capabilities=[
                "revenue analysis",
                "product performance",
                "customer insights",
                "trend analysis"
            ],
            keywords={
                "revenue": ["sales", "income", "earnings"],
                "products": ["items", "goods", "merchandise"],
                "customers": ["buyers", "clients", "users"]
            }
        )
    
    def get_capabilities(self) -> list[str]:
        return ["revenue analysis", "product performance", "customer insights"]

# 2. Setup
db = DatabaseFactory.create(
    DatabaseType.SQLITE,
    DatabaseSettings(database="sales.db")
)
service = SalesService(db)

# 3. Create AI client (Ollama local LLM)
ai_client = ReasoningFactory.create(ReasoningProvider.OLLAMA, model="mistral")

# 4. Create query agent
agent = QueryAgent(
    ai_client=ai_client,
    datasets={"sales": service},
    enable_planning=True,
    enable_insights=True
)

# 5. Ask questions in natural language!
result = agent.execute_query("What are the top 5 products by revenue?")

print(f"📊 Generated SQL:\n{result['sql']}\n")
print(f"✅ Results ({len(result['results'])} rows):")
for i, row in enumerate(result['results'][:5], 1):
    print(f"  {i}. {row}")
print(f"\n💡 AI Insights:\n{result['insights']}")
```

**Output:**
```
📊 Generated SQL:
SELECT p.name, SUM(o.total_amount) AS revenue
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY p.name
ORDER BY revenue DESC
LIMIT 5

✅ Results (5 rows):
  1. {'name': 'Product_296', 'revenue': 3089981.76}
  2. {'name': 'Product_230', 'revenue': 3059421.34}
  3. {'name': 'Product_21', 'revenue': 3059310.93}
  4. {'name': 'Product_246', 'revenue': 3049129.02}
  5. {'name': 'Product_391', 'revenue': 3015368.84}

💡 AI Insights:
The top 5 products by revenue show strong performance across the catalog...
```

---

## Core Components

### 1. AIClient

Provider-agnostic AI client for LLM interactions.

**Features:**
- Unified interface for Ollama, OpenAI, Anthropic
- Uses `axiompy.io.HTTPClient` (no specialized LLM libraries)
- Built-in caching for expensive LLM calls
- Automatic retry with exponential backoff
- Type-safe with full type hints

**API:**

```python
from axiompy.reasoning import AIClient

class AIClient:
    """Provider-agnostic AI client."""
    
    def generate_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Generate completion for custom prompts."""
        ...
    
    def generate_sql_from_question(
        self,
        question: str,
        metadata: DatasetMetadata,
        examples: list[dict] = None
    ) -> str:
        """Generate SQL from natural language question."""
        ...
    
    def generate_insight(
        self,
        data: list[dict],
        question: str,
        metadata: DatasetMetadata
    ) -> str:
        """Generate AI insights from query results."""
        ...
```

**Usage:**

```python
from axiompy.reasoning import ReasoningFactory, ReasoningProvider

# Ollama (local)
ai = ReasoningFactory.create(
    ReasoningProvider.OLLAMA,
    model="mistral",
    endpoint="http://localhost:11434/api/generate"
)

# OpenAI
ai = ReasoningFactory.create(
    ReasoningProvider.OPENAI,
    api_key="sk-...",
    model="gpt-4"
)

# Anthropic
ai = ReasoningFactory.create(
    ReasoningProvider.ANTHROPIC,
    api_key="sk-ant-...",
    model="claude-3-opus"
)

# Generate SQL
sql = ai.generate_sql_from_question(
    question="What are the top 10 customers?",
    metadata=service.get_metadata()
)

# Generate insights
insights = ai.generate_insight(
    data=results,
    question="What are the top 10 customers?",
    metadata=service.get_metadata()
)
```

---

### 2. BaseDatasetService

Abstract interface for dataset services that can be queried by AI agents.

**Purpose:**
- Standard interface for AI-powered query routing
- Self-describing via metadata
- Backend-agnostic (database, API, file, search engine)

**Interface:**

```python
from abc import ABC, abstractmethod
from axiompy.reasoning import DatasetMetadata

class BaseDatasetService(ABC):
    """Abstract interface for dataset services."""
    
    dataset_name: str = "unknown"
    description: str = "No description provided"
    
    @abstractmethod
    def query(self, sql: str, limit: int | None = None) -> list[dict]:
        """Execute query and return results."""
        ...
    
    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """Get capabilities for AI agent routing."""
        ...
    
    @abstractmethod
    def get_metadata(self) -> DatasetMetadata:
        """Get rich metadata for AI reasoning and SQL generation."""
        ...
```

**Implementation Example:**

```python
from axiompy.reasoning import BaseDatasetService, DatasetMetadata

class MyService(BaseDatasetService):
    dataset_name = "mydata"
    description = "My dataset description"
    
    def __init__(self, db):
        self.db = db
    
    def query(self, sql: str, limit: int = None) -> list[dict]:
        # Add LIMIT if not present
        if limit and "LIMIT" not in sql.upper():
            sql = f"{sql.rstrip(';')} LIMIT {limit}"
        return self.db.execute(sql)
    
    def get_capabilities(self) -> list[str]:
        return ["analysis", "reporting", "insights"]
    
    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            dataset=self.dataset_name,
            description=self.description,
            # ... (see Metadata System section)
        )
```

---

### 3. QueryAgent

AI-powered agent for intelligent query routing and execution.

**Features:**
- Natural language understanding
- Automatic dataset selection
- SQL generation from questions
- Schema-based validation
- Insight generation
- Error recovery with retries

**API:**

```python
from axiompy.reasoning.agents import QueryAgent

class QueryAgent:
    """AI Agent that intelligently routes and executes queries."""
    
    def __init__(
        self,
        ai_client: AIClient,
        datasets: dict[str, BaseDatasetService],
        enable_planning: bool = True,
        enable_insights: bool = True,
        max_retries: int = 2
    ):
        """Initialize query agent."""
        ...
    
    def execute_query(self, question: str) -> dict[str, Any]:
        """
        Execute natural language query.
        
        Returns:
            {
                "question": str,
                "dataset": str,
                "sql": str,
                "results": list[dict],
                "insights": str,
                "metadata": dict
            }
        """
        ...
```

**Execution Flow:**

```
1. Planning (AI)
   ↓
   - Analyze question
   - Select dataset
   - Determine strategy

2. SQL Generation (AI)
   ↓
   - Use dataset metadata
   - Generate SQL query
   - Include examples if available

3. Validation
   ↓
   - Extract columns from SQL
   - Check columns exist in schema
   - Validate syntax

4. Execution
   ↓
   - Run SQL via dataset service
   - Apply row limits
   - Handle errors

5. Insights (AI)
   ↓
   - Analyze results
   - Generate natural language explanation
   - Provide recommendations
```

**Usage:**

```python
from axiompy.reasoning import ReasoningFactory, ReasoningProvider, QueryAgent

# Create AI client
ai = ReasoningFactory.create(ReasoningProvider.OLLAMA, model="mistral")

# Create agent with multiple datasets
agent = QueryAgent(
    ai_client=ai,
    datasets={
        "sales": sales_service,
        "inventory": inventory_service,
        "customers": customer_service
    },
    enable_planning=True,  # AI selects dataset
    enable_insights=True,  # Generate insights
    max_retries=2          # Retry on validation failure
)

# Execute query
result = agent.execute_query("Show me low-stock products")
# Agent automatically:
# 1. Selects "inventory" dataset
# 2. Generates appropriate SQL
# 3. Validates against inventory schema
# 4. Executes query
# 5. Generates insights
```

---

## AI Providers

### Provider Architecture

The reasoning module uses a provider pattern to support multiple LLM services:

```
AIClient (provider-agnostic)
    ↓
ProviderConfig (abstract)
    ↓
┌───────────────┬──────────────────┬─────────────────┐
│               │                  │                 │
Ollama      OpenAI           Anthropic         (Custom)
```

Each provider implements:
- `format_prompt()` - Convert domain prompt to provider format
- `build_payload()` - Build provider-specific API payload
- `parse_response()` - Extract text from provider response

### Supported Providers

#### 1. Ollama (Local LLM)

**Best for:** Development, privacy-sensitive data, cost control

**Setup:**
```bash
# macOS
brew install ollama
ollama pull mistral

# Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
```

**Usage:**
```python
from axiompy.reasoning import AIClientFactory

ai = AIClientFactory.create_ollama(
    model="mistral",  # or "llama2", "neural-chat", "codellama"
    endpoint="http://localhost:11434/api/generate"
)
```

**Recommended Models:**
- `mistral` - Best balance of speed and quality for SQL generation
- `llama2` - Good general-purpose model
- `neural-chat` - Fast, good for simple queries
- `codellama` - Specialized for code generation

#### 2. OpenAI

**Best for:** Production, highest quality, complex reasoning

**Setup:**
```bash
export OPENAI_API_KEY="sk-..."
```

**Usage:**
```python
ai = AIClientFactory.create_openai(
    api_key="sk-...",
    model="gpt-4",  # or "gpt-3.5-turbo"
    endpoint="https://api.openai.com/v1/chat/completions"
)
```

#### 3. Anthropic

**Best for:** Long context, detailed analysis

**Setup:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Usage:**
```python
ai = AIClientFactory.create_anthropic(
    api_key="sk-ant-...",
    model="claude-3-opus",  # or "claude-2"
    endpoint="https://api.anthropic.com/v1/complete"
)
```

### Custom Providers

Add your own provider by implementing `ProviderConfig`:

```python
from axiompy.reasoning.providers import ProviderConfig

class MyProviderConfig(ProviderConfig):
    @staticmethod
    def format_prompt(prompt_dict: dict[str, str]) -> dict:
        """Format prompt for your provider."""
        return {
            "system": prompt_dict.get("system", ""),
            "user": prompt_dict.get("user", "")
        }
    
    @staticmethod
    def build_payload(formatted_prompt, model, endpoint, **options) -> dict:
        """Build API payload."""
        return {
            "model": model,
            "messages": formatted_prompt,
            "temperature": options.get("temperature", 0.7),
            "max_tokens": options.get("max_tokens", 500)
        }
    
    @staticmethod
    def parse_response(response_json: dict) -> str:
        """Extract text from response."""
        return response_json["choices"][0]["message"]["content"]

# Register and use
ai = AIClient(
    provider="myprovider",
    model="mymodel",
    endpoint="https://api.myprovider.com/v1/generate",
    provider_config=MyProviderConfig
)
```

---

## Metadata System

### Overview

The metadata system enables **self-describing datasets** - datasets that provide rich schema information to AI agents for intelligent query generation.

### DatasetMetadata Structure

```python
from dataclasses import dataclass

@dataclass
class ScopeMetadata:
    """Geographic, temporal, and domain scope."""
    geographic: str  # Required: "United States", "Global", "California"
    temporal: str | None = None  # "2020-2024", "Last 30 days"
    domain: str | None = None  # "E-commerce", "Healthcare"
    important: str | None = None  # Key notes for AI

@dataclass
class TableSchemaMetadata:
    """Schema for a single table."""
    columns: dict[str, str]  # Required: {"col": "TYPE CONSTRAINTS"}
    description: str | None = None
    row_count: int | None = None
    indexes: list[str] | None = None

@dataclass
class ExampleMetadata:
    """Example query for few-shot learning."""
    question: str
    sql: str
    explanation: str | None = None

@dataclass
class DatasetMetadata:
    """Complete metadata for self-describing tools."""
    dataset: str  # Required: dataset name
    description: str  # Required: what this dataset contains
    scope: ScopeMetadata  # Required: geographic/temporal/domain scope
    schema: dict[str, TableSchemaMetadata]  # Required: table schemas
    
    # Optional but recommended
    capabilities: list[str] | None = None  # ["revenue analysis", "trends"]
    keywords: dict[str, list[str]] | None = None  # {"revenue": ["sales", "income"]}
    examples: list[ExampleMetadata] | None = None  # Few-shot examples
    constraints: list[str] | None = None  # ["No PII", "Aggregated only"]
    common_mistakes: dict[str, str] | None = None  # {"mistake": "correction"}
```

### Complete Metadata Example

```python
from axiompy.reasoning import (
    DatasetMetadata,
    ScopeMetadata,
    TableSchemaMetadata,
    ExampleMetadata
)

metadata = DatasetMetadata(
    dataset="ecommerce",
    description="E-commerce sales data with orders, products, and customers",
    
    scope=ScopeMetadata(
        geographic="United States",
        temporal="January 2020 - December 2024",
        domain="E-commerce / Retail",
        important="All monetary values in USD. Timestamps in UTC."
    ),
    
    schema={
        "orders": TableSchemaMetadata(
            columns={
                "order_id": "INTEGER PRIMARY KEY",
                "customer_id": "INTEGER NOT NULL",
                "product_id": "INTEGER NOT NULL",
                "quantity": "INTEGER NOT NULL",
                "total_amount": "DECIMAL(10,2) NOT NULL",
                "order_date": "TIMESTAMP NOT NULL"
            },
            description="Customer orders with product and pricing information",
            row_count=1000000,
            indexes=["customer_id", "product_id", "order_date"]
        ),
        "products": TableSchemaMetadata(
            columns={
                "product_id": "INTEGER PRIMARY KEY",
                "name": "TEXT NOT NULL",
                "category": "TEXT NOT NULL",
                "price": "DECIMAL(10,2) NOT NULL",
                "stock_quantity": "INTEGER NOT NULL"
            },
            description="Product catalog with pricing and inventory",
            row_count=500
        ),
        "customers": TableSchemaMetadata(
            columns={
                "customer_id": "INTEGER PRIMARY KEY",
                "name": "TEXT NOT NULL",
                "email": "TEXT UNIQUE NOT NULL",
                "join_date": "DATE NOT NULL"
            },
            description="Customer information",
            row_count=100000
        )
    },
    
    capabilities=[
        "revenue analysis",
        "product performance tracking",
        "customer behavior analysis",
        "inventory management",
        "sales trends over time"
    ],
    
    keywords={
        "revenue": ["sales", "income", "earnings", "money"],
        "products": ["items", "goods", "merchandise", "SKU"],
        "customers": ["buyers", "clients", "users", "shoppers"],
        "popular": ["top", "best", "most", "highest"]
    },
    
    examples=[
        ExampleMetadata(
            question="What are the top 5 products by revenue?",
            sql="""
                SELECT p.name, SUM(o.total_amount) AS revenue
                FROM orders o
                JOIN products p ON o.product_id = p.product_id
                GROUP BY p.name
                ORDER BY revenue DESC
                LIMIT 5
            """,
            explanation="Join orders with products, aggregate revenue, sort descending"
        ),
        ExampleMetadata(
            question="How many orders were placed last month?",
            sql="""
                SELECT COUNT(*) as order_count
                FROM orders
                WHERE order_date >= date('now', '-1 month')
            """,
            explanation="Use date functions to filter recent orders"
        )
    ],
    
    constraints=[
        "Do not return customer email addresses",
        "Always use aggregated data for customer analysis",
        "Maximum 1000 rows per query"
    ],
    
    common_mistakes={
        "Using 'price' from orders": "Use 'total_amount' instead - price is in products table",
        "Forgetting JOIN": "Always JOIN orders with products for product names",
        "Missing GROUP BY": "When using aggregates like SUM(), include GROUP BY"
    }
)
```

### Metadata Helpers

```python
from axiompy.reasoning.metadata_helpers import (
    extract_all_columns,
    format_schema_for_llm,
    extract_columns_from_sql,
    validate_columns_exist
)

# Extract all columns from metadata
columns = extract_all_columns(metadata)
# {'order_id', 'customer_id', 'product_id', 'name', 'email', ...}

# Format schema for LLM consumption
schema_text = format_schema_for_llm(metadata)
# """
# Dataset: ecommerce
# Description: E-commerce sales data...
# 
# Tables:
# - orders (1000000 rows)
#   - order_id: INTEGER PRIMARY KEY
#   - customer_id: INTEGER NOT NULL
#   ...
# """

# Extract columns from SQL
sql = "SELECT name, price FROM products WHERE category = 'Electronics'"
cols = extract_columns_from_sql(sql)
# ['name', 'price', 'category']

# Validate columns exist
validate_columns_exist(cols, metadata)
# Raises ValidationError if columns don't exist
```

---

## Query Agent

### Refactoring & Architecture

QueryAgent has been refactored from **628 lines to 353 lines** (43% reduction) by extracting validation, error feedback, and SQL generation into composable components.

**Before (Monolithic):**
```python
class QueryAgent:
    def execute_query(self, question):
        # 628 lines of:
        # - Planning logic
        # - SQL generation
        # - Inline validation (200+ lines)
        # - Manual retry loops (80+ lines)
        # - Error feedback strings (50+ lines)
        # - Execution
        # - Insights
```

**After (Composable):**
```python
class QueryAgent:
    def __init__(self, ai_client, datasets, **options):
        # Compose from focused components
        self.validation_pipeline = ValidationPipeline.default(...)
        self.feedback_generator = ErrorFeedbackGenerator()
        self.sql_generator = SQLGenerator(
            ai_client,
            self.validation_pipeline,
            self.feedback_generator
        )
    
    def execute_query(self, question):
        # 353 lines - clean orchestration
        dataset = self._plan_query(question)
        sql = self.sql_generator.generate(question, metadata, db)
        results = dataset_service.query(sql)
        insights = self._generate_insights(question, results)
        return {results, sql, insights}
```

**Benefits:**
- ✅ **43% code reduction** - Simpler, more maintainable
- ✅ **Better testability** - Each component tested independently
- ✅ **Reusable** - Components work standalone
- ✅ **Extensible** - Add validators without touching QueryAgent
- ✅ **Type-safe** - Protocol-based interfaces with enums

**Component Breakdown:**

| Component | Lines | Responsibility |
|-----------|-------|----------------|
| `QueryAgent` | 353 | High-level orchestration |
| `ValidationPipeline` | ~220 | Composable validation chain |
| `ErrorFeedbackGenerator` | ~90 | Contextual error messages |
| `SQLGenerator` | ~210 | Generation + validation + retry |

### Model Recommendations

For optimal performance, use these models with Ollama:

| Model | Size | Speed | Best For | RAM |
|-------|------|-------|----------|-----|
| **qwen2.5-coder:1.5b** ⭐ | 1.5B | 2-5s | SQL generation (default) | ~1GB |
| **deepseek-coder:1.3b** | 1.3B | 2-5s | Fast & efficient | ~1GB |
| **mistral** | 7B | 10-20s | Complex queries | ~4GB |
| **codellama** | 7B | 10-20s | Code-focused | ~4GB |

**Setup:**
```bash
# Install Ollama
brew install ollama  # macOS
# or
curl -fsSL https://ollama.com/install.sh | sh  # Linux

# Pull recommended model
ollama pull qwen2.5-coder:1.5b

# Use in QueryAgent
ai = ReasoningFactory.create(
    ReasoningProvider.OLLAMA,
    model="qwen2.5-coder:1.5b"
)
```

**Why qwen2.5-coder:1.5b?**
- ✅ **13 seconds** vs 30+ minutes for older models
- ✅ **Reliable** - Consistent, non-empty responses
- ✅ **Lightweight** - Only 1GB RAM required
- ✅ **Accurate** - Trained on code/SQL datasets

### Detailed Flow

```python
def execute_query(self, question: str) -> dict:
    """
    Execute natural language query through 5-step pipeline.
    """
    
    # Step 1: Planning (AI)
    # - Analyze question intent
    # - Select appropriate dataset
    # - Determine query strategy
    selected_dataset = self._plan_query(question)
    
    # Step 2: SQL Generation (AI)
    # - Use dataset metadata
    # - Apply few-shot examples
    # - Generate SQL query
    sql = self._generate_sql(question, metadata)
    
    # Step 3: Validation
    # - Extract columns from SQL
    # - Check columns exist in schema
    # - Validate syntax
    validation = self._validate_sql(sql, metadata)
    
    # Step 4: Execution
    # - Run SQL via dataset service
    # - Apply row limits
    # - Handle errors with retry
    results = dataset_service.query(sql, limit=1000)
    
    # Step 5: Insights (AI)
    # - Analyze results
    # - Generate natural language explanation
    # - Provide recommendations
    insights = self._generate_insights(question, results, metadata)
    
    return {
        "question": question,
        "dataset": selected_dataset,
        "sql": sql,
        "results": results,
        "insights": insights,
        "metadata": {...}
    }
```

### Configuration

```python
from axiompy.reasoning import QueryAgent

agent = QueryAgent(
    ai_client=ai,
    datasets={"sales": sales_service},
    
    # Planning
    enable_planning=True,  # Use AI for dataset selection
    
    # Insights
    enable_insights=True,  # Generate AI insights
    
    # Error handling
    max_retries=2,  # Retry SQL generation on validation failure
)
```

### Error Handling

The agent handles errors at each step:

**Planning Errors:**
- No matching dataset → Returns error with available datasets
- Ambiguous question → Asks for clarification

**SQL Generation Errors:**
- Invalid SQL → Retries with error feedback
- Timeout → Returns error after max retries

**Validation Errors:**
- Missing columns → Retries with schema reminder
- Syntax errors → Retries with syntax correction

**Execution Errors:**
- Database errors → Returns error with SQL for debugging
- Timeout → Returns partial results if available

**Example:**
```python
try:
    result = agent.execute_query("Show me the top products")
except ValueError as e:
    print(f"Planning failed: {e}")
except ConnectionError as e:
    print(f"Execution failed: {e}")
```

---

## SQL Validation

### Overview

The reasoning module includes comprehensive SQL validation to prevent LLM hallucinations and ensure generated queries are safe to execute. Validation happens in three stages before SQL reaches the database.

### Validation Pipeline Architecture

```
SQL Generation (AI)
       ↓
┌────────────────────────────────┐
│ 1. Syntax Validation           │
│    - Uses sqlparse library     │
│    - Checks SQL structure      │
│    - Validates keyword order   │
│    - Detects missing clauses   │
└────────────────────────────────┘
       ↓
┌────────────────────────────────┐
│ 2. Column Validation           │
│    - Checks columns exist      │
│    - Validates against schema  │
│    - Detects hallucinated cols │
└────────────────────────────────┘
       ↓
┌────────────────────────────────┐
│ 3. Database Dry-Run (EXPLAIN)  │
│    - Database validates query  │
│    - Catches semantic errors   │
│    - Validates table names     │
│    - No data accessed          │
└────────────────────────────────┘
       ↓
   Valid SQL Ready for Execution
```

### Validation Components

The validation system uses a **composable filter chain pattern** with Protocol-based validators:

#### ValidationPipeline

Orchestrates validation through a chain of validators:

```python
from axiompy.reasoning.agents import ValidationPipeline

# Create pipeline with default validators
pipeline = ValidationPipeline.default(
    enable_db_validation=True,
    dialect="sqlite"  # or "postgres", "mysql"
)

# Validate SQL
result = pipeline.validate(sql, metadata, db_connection)

if not result.valid:
    print(f"Errors: {result.errors}")
    print(f"Error type: {result.error_type}")  # SQLErrorType enum
```

#### Validator Protocol

All validators implement a common Protocol interface:

```python
from axiompy.validators import Validator, ValidationContext

class CustomValidator:
    """Custom validator following the Protocol"""
    
    def validate(self, context: ValidationContext) -> ValidationContext:
        # Add errors/warnings to context
        if some_condition:
            context.errors.append("Validation failed")
        return context

# Add to pipeline
pipeline.add_validator(CustomValidator())
```

#### Built-in Validators

1. **EmptySQLValidator**: Checks SQL is not empty
2. **SQLSyntaxValidator**: Uses sqlparse for syntax checking
3. **SQLColumnValidator**: Validates columns exist in schema
4. **SQLDatabaseValidator**: Dialect-specific validation via EXPLAIN

### Automatic Retry with Error Feedback

When validation fails, the system automatically retries with contextual error feedback:

```python
from axiompy.reasoning.agents import SQLGenerator

generator = SQLGenerator(
    ai_client=ai,
    validation_pipeline=pipeline,
    max_retries=2  # Total of 3 attempts
)

# Generate validates → retry on failure → return valid SQL
sql = generator.generate(question, metadata, db_connection)
```

**Retry Flow Example:**

```
Attempt 1: SELECT * LIMIT 10
❌ Validation: "LIMIT requires FROM clause"
    ↓
Attempt 2 (with feedback):
    Previous error: syntax_error
    Hint: "LIMIT requires FROM clause before it"
    ↓
    Generated: SELECT * FROM products LIMIT 10
    ✅ Validation passed!
```

### Error Type Classification

Errors are classified using the `SQLErrorType` enum for type-safe handling:

```python
from axiompy.validators import SQLErrorType

# Error types
SQLErrorType.EMPTY_SQL      # Empty response from LLM
SQLErrorType.SYNTAX_ERROR   # Invalid SQL syntax  
SQLErrorType.COLUMN_ERROR   # Column doesn't exist
SQLErrorType.DATABASE_ERROR # Database validation failed
SQLErrorType.GENERATION_ERROR # LLM API error
SQLErrorType.UNKNOWN        # Other errors
```

### Contextual Error Feedback

The `ErrorFeedbackGenerator` creates helpful, actionable feedback:

```python
from axiompy.reasoning.agents import ErrorFeedbackGenerator

feedback_gen = ErrorFeedbackGenerator()

feedback = feedback_gen.generate(
    error_type=SQLErrorType.SYNTAX_ERROR,
    previous_sql="SELECT * LIMIT 10",
    errors=["LIMIT requires FROM clause"],
    context={}
)
# Returns formatted feedback with hints for the LLM
```

### SQLValidator

Validates LLM-generated SQL to prevent hallucination errors.

**Features:**
- Column existence validation
- Table reference validation
- JOIN qualification checking
- Clear error messages for retry

**API:**

```python
from axiompy.reasoning.validators import SQLValidator

class SQLValidator:
    """Validate SQL queries against schema."""
    
    @staticmethod
    def extract_columns(sql: str) -> list[str]:
        """Extract column names from SQL query."""
        ...
    
    @staticmethod
    def validate_columns(
        sql: str,
        schema_columns: set[str],
        strict: bool = False
    ) -> ValidationResult:
        """
        Validate SQL references only columns that exist.
        
        Args:
            sql: SQL query to validate
            schema_columns: Set of valid column names
            strict: If True, fail on warnings
            
        Returns:
            ValidationResult with valid/invalid columns
        """
        ...
```

**Usage:**

```python
from axiompy.reasoning.validators import SQLValidator
from axiompy.reasoning.metadata_helpers import extract_all_columns

# Get valid columns from metadata
valid_columns = extract_all_columns(metadata)

# Validate SQL
result = SQLValidator.validate_columns(
    sql="SELECT name, price FROM products WHERE category = 'Electronics'",
    schema_columns=valid_columns,
    strict=False
)

if not result.is_valid:
    print(f"Invalid columns: {result.invalid_columns}")
    print(f"Suggestions: {result.suggestions}")
```

### Validation Rules

**1. Column Existence:**
```python
# ✅ Valid
"SELECT name, price FROM products"

# ❌ Invalid - 'cost' doesn't exist
"SELECT name, cost FROM products"
```

**2. JOIN Qualification:**
```python
# ✅ Valid - columns qualified
"SELECT p.name, o.total_amount FROM orders o JOIN products p ..."

# ⚠️  Warning - ambiguous 'name'
"SELECT name, total_amount FROM orders o JOIN products p ..."
```

**3. Alias Recognition:**
```python
# ✅ Valid - alias defined
"SELECT SUM(total_amount) AS revenue FROM orders ORDER BY revenue"

# ❌ Invalid - alias not defined
"SELECT total_amount FROM orders ORDER BY revenue"
```

---

## Best Practices

### 1. Metadata Design

**✅ DO:**
- Provide comprehensive schema information
- Include few-shot examples for common queries
- Add keywords for natural language matching
- Document constraints and common mistakes
- Keep descriptions clear and concise

**❌ DON'T:**
- Leave metadata incomplete
- Use vague descriptions
- Forget to update metadata when schema changes
- Include sensitive information in metadata

### 2. Provider Selection

**Ollama (Local):**
- ✅ Development and testing
- ✅ Privacy-sensitive data
- ✅ Cost control
- ❌ Complex reasoning (use GPT-4)

**OpenAI GPT-4:**
- ✅ Production workloads
- ✅ Complex queries
- ✅ High accuracy requirements
- ❌ Cost-sensitive applications

**OpenAI GPT-3.5:**
- ✅ Simple queries
- ✅ High volume
- ✅ Cost optimization
- ❌ Complex reasoning

### 3. Error Handling

```python
from axiompy.reasoning import QueryAgent

agent = QueryAgent(
    ai_client=ai,
    datasets={"sales": service},
    max_retries=2  # Retry on validation failure
)

try:
    result = agent.execute_query(question)
    
    # Check for warnings
    if result.get("warnings"):
        logger.warning(f"Query warnings: {result['warnings']}")
    
    # Process results
    process_results(result["results"])
    
except ValueError as e:
    # Planning or validation error
    logger.error(f"Query failed: {e}")
    return {"error": str(e)}
    
except ConnectionError as e:
    # Execution error
    logger.error(f"Database error: {e}")
    return {"error": "Database unavailable"}
```

### 4. Performance Optimization

**Caching:**
```python
# AIClient automatically caches LLM responses
# Clear cache if metadata changes
ai.clear_cache()
```

**Row Limits:**
```python
# Always apply row limits
def query(self, sql: str, limit: int = None) -> list[dict]:
    if limit and "LIMIT" not in sql.upper():
        sql = f"{sql.rstrip(';')} LIMIT {limit}"
    return self.db.execute(sql)
```

**Metadata Optimization:**
```python
# Cache metadata in service
class MyService(BaseDatasetService):
    def __init__(self, db):
        self.db = db
        self._metadata = None  # Cache
    
    def get_metadata(self) -> DatasetMetadata:
        if self._metadata is None:
            self._metadata = self._build_metadata()
        return self._metadata
```

### 5. Security

**SQL Injection Prevention:**
```python
# ✅ Use parameterized queries when possible
def query(self, sql: str, params: tuple = ()) -> list[dict]:
    return self.db.execute(sql, params)

# ✅ Validate SQL before execution
SQLValidator.validate_columns(sql, valid_columns)

# ✅ Apply row limits
if "LIMIT" not in sql.upper():
    sql = f"{sql} LIMIT 1000"
```

**Data Access Control:**
```python
# ✅ Document constraints in metadata
metadata = DatasetMetadata(
    ...,
    constraints=[
        "Do not return customer email addresses",
        "Always use aggregated data for customer analysis"
    ]
)

# ✅ Filter sensitive columns
def query(self, sql: str, limit: int = None) -> list[dict]:
    results = self.db.execute(sql)
    # Remove sensitive columns
    return [{k: v for k, v in row.items() if k not in SENSITIVE_COLUMNS}
            for row in results]
```

---

## Examples

### Complete Working Example

See `examples/ecommerce_ai/` for a full implementation:

```bash
cd examples/ecommerce_ai

# Generate 1M record dataset
python setup.py

# Run basic demo
python main.py

# Run interactive terminal demo
python interactive_demo.py

# Or use one-command setup
cd ../..
make mcp-example  # Sets up Ollama, generates data, runs tests
make mcp-interactive  # Launch interactive demo
```

### Example Structure

```
examples/ecommerce_ai/
├── README.md                          # Documentation
├── setup.py                           # Dataset generator
├── main.py                            # Basic demo
├── interactive_demo.py                # Interactive terminal demo
├── test_query.py                      # Automated tests
├── requirements.txt
├── ecommerce/
│   ├── __init__.py
│   ├── config/
│   │   └── settings.py                # Configuration
│   └── services/
│       └── ecommerce_service.py       # BaseDatasetService implementation
└── data/
    └── ecommerce.db                   # Generated SQLite database (100MB)
```

### Key Files

**ecommerce_service.py:**
```python
from axiompy.reasoning import BaseDatasetService, DatasetMetadata

class EcommerceService(BaseDatasetService):
    dataset_name = "ecommerce"
    description = "E-commerce sales data"
    
    def __init__(self, database):
        self.db = database
    
    def query(self, sql: str, limit: int = None) -> list[dict]:
        if limit and "LIMIT" not in sql.upper():
            sql = f"{sql.rstrip(';')} LIMIT {limit}"
        return self.db.execute(sql)
    
    def get_metadata(self) -> DatasetMetadata:
        # ... (see Metadata System section)
        ...
```

**main.py:**
```python
from axiompy.io.database import DatabaseFactory, DatabaseType, DatabaseSettings
from axiompy.reasoning import ReasoningFactory, ReasoningProvider, QueryAgent
from ecommerce.services.ecommerce_service import EcommerceService

# Setup
db = DatabaseFactory.create(DatabaseType.SQLITE, DatabaseSettings(database="data/ecommerce.db"))
service = EcommerceService(db)
ai = ReasoningFactory.create(ReasoningProvider.OLLAMA, model="mistral")
agent = QueryAgent(ai, {"ecommerce": service})

# Query
result = agent.execute_query("What are the top 5 products by revenue?")
print(f"SQL: {result['sql']}")
print(f"Results: {result['results']}")
```

---

## Troubleshooting

### Common Issues

#### 1. "Model not found" Error

**Problem:**
```
HTTP 404: {"error":"model 'mistral' not found"}
```

**Solution:**
```bash
# Pull the model
ollama pull mistral

# Verify
ollama list
```

#### 2. SQL Syntax Errors

**Problem:**
```
Execute failed: near "LIMIT": syntax error
```

**Solution:**
- Check for duplicate LIMIT clauses
- Ensure SQL is normalized (no newlines)
- Validate against schema

**Fixed in AxiomPy:**
```python
# Database automatically normalizes SQL
# EcommerceService checks for existing LIMIT
```

#### 3. Invalid Column Errors

**Problem:**
```
Execute failed: no such column: c.category
```

**Solution:**
- LLM hallucinated table alias
- Improve metadata with examples
- Add common mistakes to metadata

```python
metadata = DatasetMetadata(
    ...,
    common_mistakes={
        "Using alias 'c' for products": "Use 'p' or 'products' instead"
    },
    examples=[
        ExampleMetadata(
            question="Show products by category",
            sql="SELECT p.name, p.category FROM products p ..."
        )
    ]
)
```

#### 4. Ollama Connection Errors

**Problem:**
```
Connection refused: http://localhost:11434
```

**Solution:**
```bash
# Start Ollama service
ollama serve

# Or use system service
brew services start ollama  # macOS
systemctl start ollama      # Linux
```

#### 5. Slow Query Generation

**Problem:**
- LLM takes 20+ seconds to generate SQL

**Solution:**
- Use faster model (neural-chat vs mistral)
- Reduce metadata size
- Cache results
- Consider GPT-3.5-turbo for production

```python
# Fast local model
ai = ReasoningFactory.create(ReasoningProvider.OLLAMA, model="neural-chat")

# Or fast cloud model
ai = ReasoningFactory.create(ReasoningProvider.OPENAI, model="gpt-3.5-turbo")
```

---

## Migration Guide

### From MCP Server to AxiomPy Reasoning

If you're migrating from the original MCP server implementation:

**Before (MCP Server):**
```python
from services.reasoning.service import ReasoningService
from services.crime.service import CrimeService

reasoning = ReasoningService(provider="ollama", model="mistral")
crime_service = CrimeService(db)

# Custom query logic (200+ lines)
```

**After (AxiomPy):**
```python
from axiompy.reasoning import ReasoningFactory, ReasoningProvider, QueryAgent
from axiompy.reasoning import BaseDatasetService

# Implement BaseDatasetService interface
class CrimeService(BaseDatasetService):
    # ... implement interface ...

ai = ReasoningFactory.create(ReasoningProvider.OLLAMA, model="mistral")
agent = QueryAgent(ai, {"crime": CrimeService(db)})

# 25 lines total
```

**Migration Steps:**

1. **Update imports:**
   ```python
   # Old
   from services.reasoning.service import ReasoningService
   
   # New
   from axiompy.reasoning import ReasoningFactory, ReasoningProvider, AIClient
   ```

2. **Implement BaseDatasetService:**
   ```python
   # Old
   class CrimeService:
       def query(self, sql): ...
   
   # New
   from axiompy.reasoning import BaseDatasetService
   
   class CrimeService(BaseDatasetService):
       def query(self, sql, limit=None): ...
       def get_metadata(self): ...
       def get_capabilities(self): ...
   ```

3. **Use QueryAgent:**
   ```python
   # Old
   reasoning = ReasoningService(...)
   sql = reasoning.generate_sql(question, metadata)
   results = service.query(sql)
   
   # New
   agent = QueryAgent(ai, {"crime": service})
   result = agent.execute_query(question)
   # Returns: {"sql": ..., "results": ..., "insights": ...}
   ```

4. **Update metadata:**
   ```python
   # Old
   CRIME_METADATA = {...}  # Dict
   
   # New
   from axiompy.reasoning import DatasetMetadata
   
   def get_metadata(self) -> DatasetMetadata:
       return DatasetMetadata(...)  # Dataclass
   ```

---

## API Reference

### Module Structure

```
axiompy.reasoning/
├── __init__.py                    # Public API
├── client.py                      # AIClient
├── factory.py                     # AIClientFactory
├── base.py                        # BaseDatasetService
├── metadata.py                    # DatasetMetadata, ScopeMetadata, etc.
├── metadata_helpers.py            # Utility functions
├── prompts.py                     # DynamicPromptBuilder
├── validators.py                  # SQLValidator
├── agents/
│   ├── __init__.py
│   └── query.py                   # QueryAgent
└── providers/
    ├── __init__.py
    ├── base.py                    # ProviderConfig (ABC)
    ├── ollama.py                  # OllamaProviderConfig
    ├── openai.py                  # OpenAIProviderConfig
    └── anthropic.py               # AnthropicProviderConfig
```

### Public API

```python
from axiompy.reasoning import (
    # Core
    AIClient,
    ReasoningFactory,
    ReasoningProvider,
    QueryAgent,
    BaseDatasetService,
    
    # Metadata
    DatasetMetadata,
    ScopeMetadata,
    TableSchemaMetadata,
    ExampleMetadata,
    
    # Utilities
    SQLValidator,
    DynamicPromptBuilder,
    
    # Helpers
    extract_all_columns,
    format_schema_for_llm,
    extract_columns_from_sql,
    validate_columns_exist
)
```

---

## Performance Benchmarks

### Query Execution Times

**Ollama (local, mistral):**
- Planning: 5-10s
- SQL Generation: 10-20s
- Validation: <0.1s
- Execution: 0.1-1s
- Insights: 10-20s
- **Total: 25-50s**

**OpenAI (gpt-3.5-turbo):**
- Planning: 1-2s
- SQL Generation: 2-4s
- Validation: <0.1s
- Execution: 0.1-1s
- Insights: 2-4s
- **Total: 5-10s**

**OpenAI (gpt-4):**
- Planning: 2-5s
- SQL Generation: 5-10s
- Validation: <0.1s
- Execution: 0.1-1s
- Insights: 5-10s
- **Total: 12-25s**

### Optimization Tips

1. **Use caching** - AIClient caches LLM responses
2. **Disable planning** - If only one dataset
3. **Disable insights** - If not needed
4. **Use faster models** - neural-chat vs mistral
5. **Optimize metadata** - Smaller metadata = faster prompts

---

## Contributing

See main [AxiomPy README](../../README.md) for contribution guidelines.

**Reasoning Module Specific:**
- Add tests for new providers
- Document metadata best practices
- Include examples in docstrings
- Maintain 80%+ test coverage

---

## License

MIT License - see [LICENSE](../../LICENSE)

---

## Support

- **Issues:** [GitHub Issues](https://github.com/varonusmaximus/axiompy/issues)
- **Examples:** `examples/ecommerce_ai/`
- **Tests:** `tests/test_reasoning_*.py`

---

**Built with ❤️ for AI-powered data intelligence**

---

**Last Updated:** 2025-12-03

