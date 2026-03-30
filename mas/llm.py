import json
from typing import Any, Dict, List, Optional


def run_llm(
    client,
    prompt: str,
    tools: Optional[List[Dict]] = None,
    tool_functions: Optional[Dict[str, Any]] = None,
    model: str = "gpt-5-mini",
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        tools=tools if tools else None,
        tool_choice="auto" if tools else None,
    )

    message = response.choices[0].message
    if not getattr(message, "tool_calls", None):
        return message.content

    if not tool_functions:
        return message.content + "\n\nNo tool functions provided to execute tool calls."

    tool_messages = []
    for tool_call in message.tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments or "{}")
        tool_fn = tool_functions.get(func_name)

        try:
            result = tool_fn(**args) if tool_fn else {"error": f"Tool '{func_name}' not implemented."}
        except Exception as exc:
            result = {"error": str(exc)}

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })

    followup_messages = [
        {"role": "system", "content": prompt},
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        },
        *tool_messages,
    ]

    final = client.chat.completions.create(model=model, messages=followup_messages)
    return final.choices[0].message.content
