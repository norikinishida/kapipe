from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal, cast

from .. import utils
from ..llms.base import BaseLLM
from .tool import Tool


@dataclass
class AgentStep:
    """Record of one LLM decision and its resulting action."""

    llm_input: str
    llm_output: str
    action_type: Literal["use_tool", "finish"]
    thought: str
    tool_name: str | None = None
    tool_input: Any | None = None
    tool_output: Any | None = None
    final_answer: str | None = None


@dataclass
class AgentTrajectory:
    """Complete execution trajectory of one agent run."""

    initial_input: str
    agent_steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        agent_step: AgentStep,
    ) -> None:
        """Append one step and update the final answer when execution finishes."""

        # Validate that a new step is not provided after the final answer has been set
        if self.final_answer is not None:
            raise RuntimeError(
                "A step cannot be added after the final answer."
            )

        # Append the validated step to the execution history
        self.agent_steps.append(agent_step)

        # Synchronize the trajectory-level answer with the final step
        if agent_step.action_type == "finish":
            self.final_answer = agent_step.final_answer


class ToolCallingAgent:
    """Agent that repeatedly selects tools until the LLM returns a final answer."""

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[Tool],
        prompt_template_name_or_path: str = "tool_calling_agent_01",
        max_steps: int = 10,
        temperature: float = 0.0,
        fallback_answer: str = "NO ANSWER",
    ) -> None:
        # Validate max_steps
        if not isinstance(max_steps, int):
            raise TypeError("max_steps must be an integer.")
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero.")

        # Validate the temperature
        if not isinstance(temperature, int | float):
            raise TypeError("temperature must be a number.")

        # Validate the tool names
        tool_names = [tool.name for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Tool names must be unique.")

        # Store the externally initialized components
        self.llm = llm
        self.tools = {
            tool.name: tool
            for tool in tools
        }

        # Load the prompt template
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path=self.prompt_template_name_or_path,
            prompt_template_package_name="kapipe.agents.prompt_templates",
        )

        # Validate the prompt template
        if "{tool_specifications}" not in self.prompt_template:
            raise ValueError(
                "The prompt template must contain {tool_specifications}."
            )
        if "{initial_input}" not in self.prompt_template:
            raise ValueError(
                "The prompt template must contain {initial_input}."
            )
        if "{execution_trajectory}" not in self.prompt_template:
            raise ValueError(
                "The prompt template must contain {execution_trajectory}."
            )

        # Store the execution settings
        self.max_steps = max_steps
        self.temperature = float(temperature)
        self.fallback_answer = fallback_answer

    def infer(
        self,
        initial_input: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentTrajectory:
        """Run the agent until it returns an answer or reaches max_steps."""

        # Validate the initial input
        if not isinstance(initial_input, str):
            raise TypeError("initial_input must be a string.")
        if not initial_input.strip():
            raise ValueError("initial_input must not be empty.")

        # Validate the metadata
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None.")

        # Initialize a complete execution trajectory before the first LLM call
        trajectory: AgentTrajectory = AgentTrajectory(
            initial_input=initial_input,
            metadata={} if metadata is None else dict(metadata),
        )

        # Execute at most max_steps LLM decisions
        for _ in range(self.max_steps):
            # Build a prompt from the input, Tool specifications, and current trajectory
            llm_input = self._build_llm_input(trajectory=trajectory)

            # Ask the LLM to select a Tool or return a final answer
            llm_output = self.llm.generate(
                prompt=llm_input,
                temperature=self.temperature,
            )

            # Parse and validate the JSON action returned by the LLM
            action = self._parse_action(llm_output=llm_output)

            # Execute the selected Tool when more information is required
            if action["action_type"] == "use_tool":
                tool_name = action["tool_name"]
                tool_input = action["tool_input"]

                # Validate the selected tool
                if tool_name not in self.tools:
                    raise ValueError(
                        f"The LLM selected an unknown Tool: {tool_name!r}."
                    )

                # Resolve the selected Tool
                tool = self.tools[tool_name]

                # Execute the Tool with the input
                tool_output = tool.run(tool_input=tool_input)

                # Ensure that the Tool result can be included in later JSON prompts
                self._ensure_json_serializable(
                    value=tool_output,
                    value_name=f"output from Tool {tool_name!r}",
                )

                # Record the complete Tool-use step
                agent_step: AgentStep = AgentStep(
                    llm_input=llm_input,
                    llm_output=llm_output,
                    action_type=action["action_type"],
                    thought=action["thought"],
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=tool_output,
                )
                trajectory.add_step(agent_step=agent_step)
                
                # Continue with the Tool result available in the trajectory
                continue

            # Read the answer after the validated finish action
            final_answer = action["answer"]

            # Record the complete final-answer step
            agent_step: AgentStep = AgentStep(
                llm_input=llm_input,
                llm_output=llm_output,
                action_type=action["action_type"],
                thought=action["thought"],
                final_answer=final_answer,
            )
            trajectory.add_step(agent_step=agent_step)

            return trajectory

        # Return the configured fallback when the agent reaches the execution limit
        trajectory.final_answer = self.fallback_answer

        return trajectory

    def _build_llm_input(
        self,
        trajectory: AgentTrajectory,
    ) -> str:
        """Build an LLM prompt from the current execution trajectory."""

        # Convert Tool specifications into a JSON-compatible prompt representation
        tool_specifications = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
            }
            for tool in self.tools.values()
        ]
        tool_specifications = json.dumps(
            tool_specifications,
            ensure_ascii=False,
            indent=2,
        )

        # Convert the previous steps into a JSON-compatible prompt representation
        execution_trajectory = [
            {
                "step": step_i + 1,
                "thought": agent_step.thought,
                "action_type": agent_step.action_type,
                "tool_name": agent_step.tool_name,
                "tool_input": agent_step.tool_input,
                "tool_output": agent_step.tool_output,
                "final_answer": agent_step.final_answer,
            }
            for step_i, agent_step in enumerate(trajectory.agent_steps)
        ]
        execution_trajectory = json.dumps(
            execution_trajectory,
            ensure_ascii=False,
            indent=2,
        )

        # Insert the serialized Tool specifications and execution context
        llm_input = self.prompt_template.format(
            initial_input=trajectory.initial_input,
            tool_specifications=tool_specifications,
            execution_trajectory=execution_trajectory,
        )

        return llm_input

    def _parse_action(
        self,
        llm_output: str,
    ) -> dict[str, Any]:
        """Parse and validate one JSON action returned by the LLM."""

        # Validate the LLM output
        if not isinstance(llm_output, str):
            raise TypeError("llm_output must be a string.")

        # Parse the complete output
        try:
            parsed_output = json.loads(llm_output)
        except json.JSONDecodeError as error:
            raise ValueError(
                "The LLM output must be exactly one valid JSON object."
            ) from error

        # Validate the parsed output
        if not isinstance(parsed_output, dict):
            raise ValueError("The LLM output must be a JSON object.")
        if "thought" not in parsed_output:
            raise ValueError("The LLM output must contain thought.")
        if "action_type" not in parsed_output:
            raise ValueError("The LLM output must contain action_type.")

        # Validate the thought
        thought = parsed_output["thought"]
        if not isinstance(thought, str):
            raise TypeError("thought must be a string.")
        if not thought.strip():
            raise ValueError("thought must not be empty.")

        # Validate the tool-use action
        if parsed_output["action_type"] == "use_tool":
            expected_fields = {
                "thought",
                "action_type",
                "tool_name",
                "tool_input",
            }
            if set(parsed_output) != expected_fields:
                raise ValueError(
                    "A use_tool action must contain exactly thought, "
                    "action_type, tool_name, and tool_input."
                )

            # Validate the tool name
            tool_name = parsed_output["tool_name"]
            if not isinstance(tool_name, str):
                raise TypeError("tool_name must be a string.")
            if not tool_name.strip():
                raise ValueError("tool_name must not be empty.")

            return cast(dict[str, Any], parsed_output)

        # Validate the final-answer action
        if parsed_output["action_type"] == "finish":
            expected_fields = {
                "thought",
                "action_type",
                "answer",
            }
            if set(parsed_output) != expected_fields:
                raise ValueError(
                    "A finish action must contain exactly thought, "
                    "action_type, and answer."
                )

            # Validate the answer
            if not isinstance(parsed_output["answer"], str):
                raise TypeError("answer must be a string.")

            return cast(dict[str, Any], parsed_output)

        # Validate that the action type is supported
        raise ValueError(
            "action_type must be either 'use_tool' or 'finish'."
        )

    def _ensure_json_serializable(
        self,
        value: Any,
        value_name: str,
    ) -> None:
        """Ensure that a value can be included in subsequent JSON prompts."""

        # Validate that the value can be serialized to JSON
        try:
            json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"The {value_name} must be JSON-serializable."
            ) from error