from __future__ import annotations

from typing import Any, Callable


class Tool:
    """Interface for tools executed by a tool-calling agent."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any],
        function: Callable[[Any], Any],
    ) -> None:
        # Validate the tool name
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if not name.strip():
            raise ValueError("name must not be empty.")

        # Validate the description
        if not isinstance(description, str):
            raise TypeError("description must be a string.")
        if not description.strip():
            raise ValueError("description must not be empty.")

        # Validate the input and output schemas
        if not isinstance(input_schema, dict):
            raise TypeError("input_schema must be a dictionary.")
        if not isinstance(output_schema, dict):
            raise TypeError("output_schema must be a dictionary.")

        # Validate that the function is callable
        if not callable(function):
            raise TypeError("function must be callable.")

        # Store the tool specification
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.function = function

    def run(
        self,
        tool_input: Any,
    ) -> Any:
        """Execute the configured function."""
        return self.function(tool_input)
