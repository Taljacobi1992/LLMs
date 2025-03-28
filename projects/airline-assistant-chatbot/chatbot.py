# imports

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
MODEL = "gpt-4o-mini"
openai = OpenAI()

system_message = "You are a helpful assistant for an Airline. "
system_message += "Give short, straight to the point answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."


# First tool
ticket_prices = {"tel aviv": "$999", "london": "$799", "paris": "$899", "tokyo": "$1,400", "berlin": "$499", "new york": "$1,200"}

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")

# There's a particular dictionary structure that's required to describe our function:

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

# Second tool

operated_airline = {"tel aviv": "el-al", "london": "british airways", "paris": "air france", "tokyo": "japan airlines", "berlin": "lufthansa", "new york": "delta"}

def get_airline_name(destination_city):
    print(f"Tool get_airline_name called for {destination_city}")
    city = destination_city.lower()
    return operated_airline.get(city, "Unknown")

airline_function = {
    "name": "get_airline_name",
    "description": "Get the airline name of destination city only when asked directly. Call this whenever you need to know the airline name, for example when a customer asks 'which airline fly to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}



tools = [{"type": "function", "function": price_function}, {"type": "function", "function": airline_function}]


def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    
    return response.choices[0].message.content


def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    airline = get_airline_name(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city,"price": price, "airline": airline}),
        "tool_call_id": tool_call.id
    }
    return response, city

gr.ChatInterface(fn=chat, type="messages").launch()
