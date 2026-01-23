import json
import os
import requests


def build_risk_prompt(entity_type: str, entity_name: str, title: str,
                      snippet: str = "", source: str = "", url: str = "") -> dict:
    system = (
        "You are a reputation risk analyst. "
        "Classify the headline as crisis_risk, routine_financial, neutral, or positive. "
        "Respond with compact JSON: {label, severity, reason}."
    )
    user = (
        f"Entity: {entity_type} = {entity_name}\n"
        f"Title: {title}\n"
        f"Snippet: {snippet}\n"
        f"Source: {source}\n"
        f"URL: {url}\n"
        "Return JSON only."
    )
    return {"system": system, "user": user}


def _parse_json_from_text(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def call_llm_openai(prompt: dict, api_key: str, model: str, timeout: int = 20) -> dict:
    if not api_key:
        return {}
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        return {}
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        return {}
    return _parse_json_from_text(content)


def call_llm_gemini(prompt: dict, api_key: str, model: str, timeout: int = 20) -> dict:
    if not api_key:
        return {}
    if not model:
        model = "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}
    merged = f"{prompt['system']}\n\n{prompt['user']}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": merged}]}],
        "generationConfig": {"temperature": 0.0},
    }
    resp = requests.post(url, params=params, json=payload, timeout=timeout)
    if resp.status_code != 200:
        return {}
    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return {}
    return _parse_json_from_text(text)


def call_llm_json(prompt: dict, api_key: str, model: str) -> dict:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "gemini":
        return call_llm_gemini(prompt, api_key, model)
    return call_llm_openai(prompt, api_key, model)
