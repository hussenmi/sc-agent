#!/bin/bash
# Smoke-test tool calling against a running OpenAI-compatible server (llama-server
# or vLLM). Sends one prompt with a single tool defined and checks the model
# returns a well-formed tool_call. This is the make-or-break for scagent — if it
# fails here, the agent won't work no matter how good the model is.
#
# Usage:
#   bash smoke_test_tools.sh                       # defaults: localhost:8001, model GLM-5.2
#   bash smoke_test_tools.sh http://localhost:8000/v1 Qwen2.5-Coder-32B-Instruct
#
# Args:
#   BASE_URL   OpenAI base url incl. /v1   (default http://localhost:8001/v1)
#   MODEL      served model name           (default GLM-5.2)

BASE_URL=${1:-"http://localhost:8001/v1"}
MODEL=${2:-"GLM-5.2"}

read -r -d '' PAYLOAD <<JSON
{
  "model": "$MODEL",
  "messages": [
    {"role": "user", "content": "What is the weather in Tokyo? Use the get_weather tool."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"}
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 0,
  "max_tokens": 256
}
JSON

echo "POST $BASE_URL/chat/completions  (model=$MODEL)"
echo ""

RESP=$(curl -s "$BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer none" \
  -d "$PAYLOAD")

if [[ -z "$RESP" ]]; then
  echo "FAIL: empty response — is the server up at $BASE_URL ?"
  exit 1
fi

# Pretty-print if python is around; otherwise dump raw.
# NB: pass RESP via env, not a stdin pipe — `python3 -` reads the program from
# stdin (the heredoc), which would collide with a piped response.
PY=$(command -v python3 || command -v python)
if [[ -n "$PY" ]]; then
  RESP="$RESP" "$PY" - <<'PYEOF'
import json, os, sys
try:
    r = json.loads(os.environ["RESP"])
except Exception as e:
    print("FAIL: response is not valid JSON:", e)
    sys.exit(1)

if "error" in r:
    print("FAIL: server returned an error:")
    print(json.dumps(r["error"], indent=2))
    sys.exit(1)

try:
    msg = r["choices"][0]["message"]
except (KeyError, IndexError):
    print("FAIL: no choices/message in response:")
    print(json.dumps(r, indent=2)[:2000])
    sys.exit(1)

calls = msg.get("tool_calls") or []
if not calls:
    print("FAIL: model did NOT emit a tool_call. Returned content instead:")
    print((msg.get("content") or "")[:1000])
    print("\n-> tool calling is not working through the chat template (--jinja).")
    sys.exit(1)

c = calls[0]["function"]
print("PASS: got tool_call")
print("  name:", c["name"])
print("  arguments:", c["arguments"])
try:
    args = json.loads(c["arguments"])
    ok = c["name"] == "get_weather" and "city" in args
    print("  parsed args:", args)
    print("\nRESULT:", "PASS — valid tool call" if ok else "WARN — call present but name/args unexpected")
    sys.exit(0 if ok else 2)
except Exception as e:
    print("\nWARN: arguments are not valid JSON:", e)
    sys.exit(2)
PYEOF
else
  echo "$RESP"
fi
