import json
import os
import re
from datetime import datetime
from smolagents import LiteLLMModel


# Set Paramters:
local_model = LiteLLMModel(model_id="ollama_chat/qwen2.5:7b", temperature=0.2,
                      max_tokens=100, requests_per_minute=60)

# Define Tools
def calculate_expression(expression):
    """Calculator: Evaluate a mathematical expression"""
    safe_expr = re.sub(r'[^0-9+\-*/(). ]', '', expression)
    if safe_expr.strip() == "":
        return None
    try:
        return eval(safe_expr)
    except:
        return None

def get_weather(location):
    """Weather: Get weather information for a location"""
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 58°F",
        "tokyo": "Rainy, 65°F",
        "paris": "Partly cloudy, 68°F"
    }
    location_lower = location.lower()
    for city, weather in weather_data.items():
        if city in location_lower:
            return f"Weather in {city.title()}: {weather}"
    return f"Weather information for {location} is not available in simulation."

def get_date(location=None):
    """Get Date: Get the current date"""
    now = datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    return f"Today's date is: {date_str}"

def get_time(location=None):
    """Get Time: Get the current time"""
    now = datetime.now()
    time_str = now.strftime("%I:%M:%S %p")
    return f"The current time is: {time_str}"

# Agent Core Function:
def call_llm(user_input, conversation_history=[], model=local_model):
    """Single LLM call function"""
    try:
        # Build messages list with conversation history
        messages = []
        
        # Add previous conversation history
        messages.extend(conversation_history)
        
        # Add current user message
        messages.extend(user_input)
        # print("🤖 LLM call:")
        # for msg in messages:
        #     print(msg)
        response = model(messages)
        return response.content
    except Exception as e:
        print(f"Error calling local model: {e}")
        return None

def call_tool(tool_name, tool_input):
    """Execute a tool function based on tool name"""
    if tool_name == "calculator":
        print("🔧 ... ...Tool: calculator")
        result = calculate_expression(tool_input)
        return f"The result is: {result}" if result is not None else "I couldn't compute that."
    elif tool_name == "get_weather":
        print("🔧 ... ...Tool: get_weather")
        return get_weather(tool_input)
    elif tool_name == "get_date":
        print("🔧 ... ...Tool: get_date")
        return get_date(tool_input)
    elif tool_name == "get_time":
        print("🔧 ... ...Tool: get_time")
        return get_time(tool_input)
    else:
        return f"Unknown tool: {tool_name}"

def update_memory(conversation_history, user_input, response):
    """Update conversation history with user message and assistant response"""
    updated_history = conversation_history.copy()
    updated_history.append({
        "role": "user",
        "content": [{'type': 'text',"text": user_input}]
    })
    updated_history.append({
        "role": "assistant",
        "content": [{'type': 'text',"text": response}]
    })
    return updated_history

def query_claude(user_input, conversation_history=[]):
    # System message for tool selection and general conversation
    system_message = (
        "You're a helpful personal assistant. Based on the user's message, "
        "decide if you need to use a tool or respond directly.\n\n"
        "If you need a tool, respond ONLY with a JSON object:\n"
        "{ \"tool\": \"calculator\", \"input\": \"5 * (4 + 3)\" }\n"
        "or\n"
        "{ \"tool\": \"get_weather\", \"input\": \"New York\" }\n"
        "or\n"
        "{ \"tool\": \"get_date\", \"input\": \"\" }\n"
        "or\n"
        "{ \"tool\": \"get_time\", \"input\": \"\" }\n\n"
        "If no tool is needed, respond naturally with a helpful message (NOT JSON)."
    )
    messages=[
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [{"type": "text", "text": user_input}]}
    ]
    # Single LLM call with conversation history
    print("🤖 System call")
    content = call_llm(messages, conversation_history)
    if content is None:
        return "Error: Could not connect to the LLM.", conversation_history
    
    # Try to extract JSON from response (if a tool is needed)
    tool = None
    tool_input = None
    
    try:
        tool_call = json.loads(content)
        tool = tool_call.get("tool")
        tool_input = tool_call.get("input")
    except:
        # Try to extract JSON if it's wrapped in markdown or other text
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            try:
                tool_call = json.loads(json_match.group())
                tool = tool_call.get("tool")
                tool_input = tool_call.get("input")
            except:
                pass

    
    # Execute tools if a tool was selected
    if tool:
        response = call_tool(tool, tool_input)
    else:
        # No tool needed - return the LLM's natural language response
        response = content
    
    # Update conversation history with user message and assistant response
    updated_history = update_memory(conversation_history, user_input, response)
    
    return response, updated_history

print("Welcome! I'm your personal assistant. I can tell you the current date, time, and weather. I can also calculate mathematical expressions. Type 'quit' to stop.")
conversation_history = []
while True:
    user_input = input("👤 You: ")
    if user_input.lower() == "quit":
        print("Agent: Goodbye!")
        break
    response, conversation_history = query_claude(user_input, conversation_history)
    print("Agent:", response)
