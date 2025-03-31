from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from openai import AsyncOpenAI
from termcolor import colored
import os

# Initialize OpenAI client
client = AsyncOpenAI()
print(colored("[LLM] OpenAI client initialized", "yellow"))

def get_llm(temperature=0):
    """Get the LLM instance with specified temperature."""
    print(colored(f"[LLM] Initializing ChatOpenAI with model=gpt-4o-mini, temperature={temperature}", "yellow"))
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

async def call_llm(messages, json_mode=False):
    """Generic function to make OpenAI API calls."""
    try:
        model = "gpt-4o"
        print(colored(f"\n[LLM] Calling {model} {'with JSON mode' if json_mode else ''}", "yellow"))
        
        # Log the full messages being sent
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            print(colored(f"\n[LLM] Message {i+1} ({role}):\n{content}", "yellow"))
        
        kwargs = {
            "model": model,
            "messages": messages
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        completion = await client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content
        
        print(colored(f"\n[LLM] Response received:\n{content}", "yellow"))
        return content
    except Exception as e:
        print(colored(f"[LLM] Error in LLM call: {str(e)}", "red"))
        raise 