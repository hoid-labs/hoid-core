from __future__ import annotations

from datetime import UTC, datetime

from hoid.core.tools import tool


@tool
def add_numbers(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
        a: First operand.
        b: Second operand.
    """
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers.

    Args:
        a: First operand.
        b: Second operand.
    """
    return a * b


@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtracts the second number from the first.

    Args:
        a: The minuend.
        b: The subtrahend.
    """
    return a - b


@tool
def divide_numbers(a: float, b: float) -> float:
    """Divides the first number by the second. Raises an error if divisor is zero.

    Args:
        a: The dividend.
        b: The divisor; must not be zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def get_current_datetime() -> str:
    "Return the current UTC date and time in ISO 8601 format."
    return datetime.now(UTC).isoformat()
