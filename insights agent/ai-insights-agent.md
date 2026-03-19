## AI Insights Agent

Starter docs for wiring OpenAI or Gemini to the existing `/api/v1/insights/*`
endpoints.

## What The JSON Contract Is

The JSON contract in [ai-insights-tool-contract.json](/Users/plane/Documents/GitHub/risk-dashboard-database/insights%20agent/ai-insights-tool-contract.json)
is not a file you hand directly to the model.

Your application should:

1. Load the JSON contract.
2. Convert it into the provider's tool format.
3. Send those tool definitions in the API request.

For OpenAI, that means passing the tools to the Responses API.

For Gemini, that means passing function declarations or Python callables to the
Gemini SDK.

## Auth Modes

This starter supports two different hosted-auth paths:

- Cloud Run IAM
- IAP desktop OAuth

### IAP Desktop OAuth

Use this mode if the internal dashboard is protected by Identity-Aware Proxy
and you want to call it from a terminal agent.

Set:

- `AI_INSIGHTS_BASE_URL=https://your-service.run.app`
- `AI_INSIGHTS_AUTH_MODE=iap-desktop`
- `AI_INSIGHTS_IAP_CLIENT_ID=...`
- `AI_INSIGHTS_IAP_CLIENT_SECRET=...`

Then run the one-time sign-in flow:

```bash
python ai_insights_api.py init-iap-desktop-auth
```

That flow opens a browser, signs you in with the allowlisted Desktop OAuth
client, and stores a refresh token locally at:

`~/.config/ai-insights-agent/iap_desktop_auth.json`

After that, the agent will automatically mint fresh `id_token`s for requests to
the IAP-protected service.

Useful environment variables:

- `AI_INSIGHTS_AUTH_MODE`
- `AI_INSIGHTS_IAP_CLIENT_ID`
- `AI_INSIGHTS_IAP_CLIENT_SECRET`
- `AI_INSIGHTS_IAP_TOKEN_FILE`

### Cloud Run IAM

If your dashboard is protected by Cloud Run IAM instead of IAP, you do not
create a token in Cloud Run itself.

For that mode:

- if `AI_INSIGHTS_AUTH_BEARER` is set, the starter uses that token directly
- otherwise, if `AI_INSIGHTS_AUTO_AUTH=true`, the starter will try:
  - `gcloud auth print-identity-token --audiences=$AI_INSIGHTS_AUTH_AUDIENCE`
  - and fall back to `gcloud auth print-identity-token`

Useful environment variables:

- `AI_INSIGHTS_AUTH_BEARER`
- `AI_INSIGHTS_AUTO_AUTH`
- `AI_INSIGHTS_AUTH_AUDIENCE`

Requirements:

- `gcloud` installed locally
- `gcloud auth login` already completed
- your user has permission to invoke the Cloud Run service

## Prompt

The baseline system prompt lives in
[ai-insights-system-prompt.txt](/Users/plane/Documents/GitHub/risk-dashboard-database/insights%20agent/ai-insights-system-prompt.txt).

It is designed to preserve an important distinction:

- negative search visibility
- weak search control
- mixed cases where both are true

That nuance matters because some brands will show neutral finance or reference
pages that are weakly controlled without being in a true negative search
spillover state.

## Starter Scripts

Example engineering starter files live in:

- [ai_insights_api.py](/Users/plane/Documents/GitHub/risk-dashboard-database/insights%20agent/ai_insights_api.py)
- [query_planner.py](/Users/plane/Documents/GitHub/risk-dashboard-database/insights%20agent/query_planner.py)
- [openai_insights_agent.py](/Users/plane/Documents/GitHub/risk-dashboard-database/insights%20agent/openai_insights_agent.py)
- [gemini_insights_agent.py](/Users/plane/Documents/GitHub/risk-dashboard-database/insights%20agent/gemini_insights_agent.py)
- [ai_insights_requirements.txt](/Users/plane/Documents/GitHub/risk-dashboard-database/insights%20agent/ai_insights_requirements.txt)

These are intentionally not wired into production Flask routes. They are a
starter for local prototyping or for lifting into a separate internal service.

## Query Planner

The starter now includes a lightweight heuristic planner in
[query_planner.py](/Users/plane/Documents/GitHub/risk-dashboard-database/insights%20agent/query_planner.py).

It prefetches likely tool results for broad or nebulous prompts such as:

- "What brand was affected in the education sector?"
- "Which brands have the most negative Top Stories today?"
- "How long is the average crisis duration broken down by industry?"

This does not replace model tool calling. It gives the model a stronger starting
context so it is less likely to stall on ambiguity.

## Hosted IAP Setup

For your hosted internal Cloud Run service behind IAP, the shortest working
setup is:

```bash
export AI_INSIGHTS_BASE_URL="https://risk-dashboard-yelv2pxzuq-uw.a.run.app"
export AI_INSIGHTS_AUTH_MODE="iap-desktop"
export AI_INSIGHTS_IAP_CLIENT_ID="YOUR_DESKTOP_OAUTH_CLIENT_ID"
export AI_INSIGHTS_IAP_CLIENT_SECRET="YOUR_DESKTOP_OAUTH_CLIENT_SECRET"

python ai_insights_api.py init-iap-desktop-auth
python gemini_insights_agent.py "Is ARKO facing negative spillover or mostly a control problem?"
```

After the one-time browser login, the agent reuses the saved refresh token and
does not need `gcloud auth print-identity-token`.

## Recommended Flow

For entity-specific questions:

1. If the entity name might be abbreviated or non-canonical, call `resolve_entity`.
2. Call `get_trend_summary`.
3. If search nuance or anomalies warrant it, call `get_evidence`.
4. Answer with dates, counts, and citations.

For discovery questions:

1. Call `screen_entities` for ranking or leaderboard-style questions.
2. Call `screen_entities` with a `sector` filter when the user wants specific brands within one industry.
3. Call `resolve_sector` if the industry wording is fuzzy or abbreviated.
4. Call `get_sector_baseline` for peer or sector-normal questions.
5. Call `aggregate_crisis_patterns` for sector questions about common crisis types and average duration.
6. Call `aggregate_industry_durations` for all-industry duration breakdowns.
7. Call `find_storylines` for thought-leadership, article-idea, and editorial-angle prompts.
8. Call `get_anomalies` for anomaly-driven discovery.
9. Drill into `get_trend_summary`, `get_narrative_timeline`, `get_search_feature_series`, and `get_evidence` for the most relevant entities.

## Additional Tools

- `resolve_sector`
  - Maps fuzzy sector phrasing like `insurers` or `financial services` to the canonical sector label used by the dashboard.
- `get_sector_baseline`
  - Returns the peer baseline for a sector and metric, plus an optional entity-vs-sector comparison.
- `get_narrative_timeline`
  - Shows whether a crisis narrative is persistent, resurfacing, or shifting over time.
- `get_narrative_tags`
  - Gives the dominant visible narrative tags on a given date.
- `get_search_feature_series`
  - Breaks search visibility down by feature type over time.
- `get_search_feature_items`
  - Returns the actual URLs, titles, and snippets behind a feature spike.
- `compare_entities`
  - Compares two brands or CEOs across recent news, search, and control metrics.
- `find_storylines`
  - Returns ranked editorial storyline candidates for a rolling window or a real calendar period such as last quarter.

## Time Windows

- Rolling windows:
  - Phrases like `rolling 90 days` or `last 90 days` are treated as rolling windows ending on the latest available data date.
- Calendar windows:
  - Phrases like `last quarter`, `this quarter`, or `Q4 2025` are treated as real calendar ranges with explicit start and end dates.

## Good Defaults

- OpenAI:
  - model: `gpt-5`
  - tool loop: manual function-calling loop with the Responses API
- Gemini:
  - model: `gemini-2.5-flash`
  - tool loop: automatic function calling through the Python SDK

## Interactive Chat

Both starter CLIs now support a simple REPL for follow-up questions in one
terminal session:

```bash
python gemini_insights_agent.py
python openai_insights_agent.py --chat
```

That mode keeps conversational context for the current terminal session, so you
can ask a follow-up like "show me the evidence" after an initial brand question.
