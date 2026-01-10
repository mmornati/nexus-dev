"""Pytest configuration and fixtures for Nexus-Dev tests."""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def sample_python_code():
    """Sample Python code for chunker tests."""
    return '''"""Sample module docstring."""

import os
from typing import Optional


def greet(name: str) -> str:
    """Say hello to someone.
    
    Args:
        name: Person's name.
        
    Returns:
        Greeting message.
    """
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Calculator:
    """A simple calculator class."""
    
    def __init__(self, initial: int = 0):
        """Initialize calculator."""
        self.value = initial
    
    def add(self, x: int) -> int:
        """Add to current value."""
        self.value += x
        return self.value
    
    def subtract(self, x: int) -> int:
        """Subtract from current value."""
        self.value -= x
        return self.value


async def async_fetch(url: str) -> str:
    """Async function example."""
    return f"Fetched: {url}"
'''


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code for chunker tests."""
    return """// Sample JavaScript module

import { something } from 'module';

function greet(name) {
    return `Hello, ${name}!`;
}

const add = (a, b) => {
    return a + b;
};

class Calculator {
    constructor(initial = 0) {
        this.value = initial;
    }
    
    add(x) {
        this.value += x;
        return this.value;
    }
    
    subtract(x) {
        this.value -= x;
        return this.value;
    }
}

export { greet, add, Calculator };
"""


@pytest.fixture
def sample_typescript_code():
    """Sample TypeScript code for chunker tests."""
    return """// Sample TypeScript module

interface User {
    name: string;
    age: number;
}

function greet(user: User): string {
    return `Hello, ${user.name}!`;
}

const add = (a: number, b: number): number => {
    return a + b;
};

class Calculator {
    private value: number;
    
    constructor(initial: number = 0) {
        this.value = initial;
    }
    
    public add(x: number): number {
        this.value += x;
        return this.value;
    }
}

export { greet, add, Calculator };
"""


@pytest.fixture
def sample_java_code():
    """Sample Java code for chunker tests."""
    return """package com.example;

import java.util.List;

/**
 * A simple calculator class.
 */
public class Calculator {
    private int value;
    
    /**
     * Create a new calculator.
     * @param initial Initial value
     */
    public Calculator(int initial) {
        this.value = initial;
    }
    
    /**
     * Add to the current value.
     * @param x Value to add
     * @return New value
     */
    public int add(int x) {
        this.value += x;
        return this.value;
    }
    
    public int subtract(int x) {
        this.value -= x;
        return this.value;
    }
}
"""


@pytest.fixture
def sample_markdown():
    """Sample Markdown documentation for chunker tests."""
    return """# Project Documentation

This is the introduction to the project.

## Installation

Install the package using pip:

```bash
pip install mypackage
```

## Configuration

Configure the application by creating a config file.

### Database Settings

Set up database connection:

```json
{
    "host": "localhost",
    "port": 5432
}
```

### API Settings

Configure API endpoints.

## Usage

Here's how to use the library.

## Contributing

See CONTRIBUTING.md for guidelines.
"""


@pytest.fixture
def sample_rst():
    """Sample RST documentation for chunker tests."""
    return """Project Documentation
=====================

This is the introduction.

Installation
------------

Install using pip::

    pip install mypackage

Configuration
-------------

Set up the config file.

Usage
-----

Here's how to use it.
"""


@pytest.fixture
def nexus_config_dict():
    """Sample Nexus-Dev configuration dictionary."""
    return {
        "project_id": "test-project-123",
        "project_name": "Test Project",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "ollama_url": "http://localhost:11434",
        "db_path": "~/.nexus-dev/test-db",
        "include_patterns": ["**/*.py", "**/*.js"],
        "exclude_patterns": ["**/node_modules/**", "**/__pycache__/**"],
        "docs_folders": ["docs/", "README.md"],
    }
