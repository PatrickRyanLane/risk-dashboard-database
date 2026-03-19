#!/usr/bin/env python3
"""Gemini starter for the AI insights assistant."""

import os
import sys
from typing import List, Tuple

from google import genai
from google.genai import types

from ai_insights_api import (
    aggregate_crisis_patterns,
    aggregate_industry_durations,
    compare_entities,
    find_storylines,
    get_anomalies,
    get_evidence,
    get_narrative_tags,
    get_narrative_timeline,
    get_search_feature_items,
    get_search_feature_series,
    get_sector_baseline,
    get_trend_summary,
    load_system_prompt,
    resolve_entity,
    resolve_sector,
    screen_entities,
)
from query_planner import build_planner_context


def build_conversation_input(history: List[Tuple[str, str]], user_message: str) -> str:
    if not history:
        return user_message
    lines = ["Conversation so far:"]
    for role, text in history[-8:]:
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {text.strip()}")
    lines.append(f"User: {user_message}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def run_turn(client: genai.Client, model: str, user_message: str, history: List[Tuple[str, str]] | None = None) -> str:
    contents = build_conversation_input(history or [], user_message)
    planner_context = build_planner_context(user_message)
    if planner_context:
        contents = f"{planner_context}\n\n{contents}"
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=load_system_prompt(),
            temperature=0.2,
            tools=[
                resolve_entity,
                resolve_sector,
                screen_entities,
                get_sector_baseline,
                aggregate_crisis_patterns,
                aggregate_industry_durations,
                find_storylines,
                get_trend_summary,
                get_narrative_timeline,
                get_narrative_tags,
                get_search_feature_series,
                get_search_feature_items,
                compare_entities,
                get_anomalies,
                get_evidence,
            ],
        ),
    )
    return response.text or ""


def run_agent(user_message: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return run_turn(client, model, user_message)


def chat_loop() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    history: List[Tuple[str, str]] = []

    print("Gemini AI insights chat. Type 'exit' or 'quit' to end.")
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

        answer = run_turn(client, model, user_message, history)
        history.append(("user", user_message))
        history.append(("assistant", answer))
        print(answer)

    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"--chat", "-i"}:
        return chat_loop()
    print(run_agent(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
