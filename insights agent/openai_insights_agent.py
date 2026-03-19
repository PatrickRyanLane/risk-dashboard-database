#!/usr/bin/env python3
"""OpenAI Responses API starter for the AI insights assistant."""

import json
import os
import sys
from typing import Optional

from openai import OpenAI

from ai_insights_api import build_openai_tools, dispatch_tool, load_system_prompt
from query_planner import build_planner_context


def run_turn(
    client: OpenAI,
    model: str,
    tools: list[dict],
    user_message: str,
    previous_response_id: Optional[str] = None,
) -> tuple[str, str]:
    planner_context = build_planner_context(user_message)
    if planner_context:
        user_message = f"{planner_context}\n\nUser question:\n{user_message}"

    request_kwargs = dict(
        model=model,
        instructions=load_system_prompt(),
        input=user_message,
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=False,
    )
    if previous_response_id:
        request_kwargs["previous_response_id"] = previous_response_id

    response = client.responses.create(
        **request_kwargs,
    )

    while True:
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            return response.output_text, response.id

        tool_outputs = []
        for call in function_calls:
            arguments = json.loads(call.arguments or "{}")
            result = dispatch_tool(call.name, arguments)
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )

        response = client.responses.create(
            model=model,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=tools,
            parallel_tool_calls=False,
        )
    return response.output_text, response.id


def run_agent(user_message: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("OPENAI_MODEL", "gpt-5")
    tools = build_openai_tools()
    text, _ = run_turn(client, model, tools, user_message)
    return text


def chat_loop() -> int:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("OPENAI_MODEL", "gpt-5")
    tools = build_openai_tools()
    previous_response_id: Optional[str] = None

    print("OpenAI AI insights chat. Type 'exit' or 'quit' to end.")
    while True:
        try:
            user_message = input("> ").strip()
        except EOFError:
            print("")
            break
        except KeyboardInterrupt:
            print("")
            break

        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            break

        answer, previous_response_id = run_turn(
            client,
            model,
            tools,
            user_message,
            previous_response_id=previous_response_id,
        )
        print(answer)

    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"--chat", "-i"}:
        return chat_loop()
    print(run_agent(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
