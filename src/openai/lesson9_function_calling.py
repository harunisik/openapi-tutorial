from openai import OpenAI
import json

client = OpenAI()

def get_weather(city: str) -> str:
    # In a real implementation, this function would call a weather API.
    # Here, we return a mock response for demonstration purposes.
    return f"The current weather in {city} is sunny with a temperature of 75°F."

# function schema
weather_tool = {
    "name": "get_weather",
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        },
    },
}


response = client.responses.create(
    model="gpt-4o-mini",
    input="What's the weather like in New York City?",
    tools=[weather_tool],
    tool_choice="auto",
)

print(response.output_text)


# collect tool calls using attribute access
tool_calls = []
for item in response.output:
    if getattr(item, "type") == "function_call":
        tool_calls.append(item)

# execute the tool call (parse arguments if they're a JSON string)
result = None
for call in tool_calls:
    if getattr(call, "name") == "get_weather":
        args = call.arguments
        if isinstance(args, str):
            args = json.loads(args)
        if "city" in args:
            result = get_weather(**args)
        else:
            print("Error: 'city' argument is missing.")

# send the tool result back as a 'tool' message (don't include SDK objects directly)
if result is not None:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "user", "content": "What is the weather in Berlin?"},
            {
                "role": "tool",
                "tool_name": getattr(tool_calls[0], "name", "get_weather") if tool_calls else "get_weather",
                "content": result,
            },
        ],
    )

print(response.output_text)