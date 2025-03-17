# imports
from openai import OpenAI
from dotenv import load_dotenv
from IPython.display import Markdown, update_display
import ollama

# constants
MODEL_GPT = 'gpt-4o-mini'
MODEL_LLAMA = 'llama3.2

# set up environment
load_dotenv(
openai = OpenAI()

# Define our system prompt
system_prompt = "You are an tutor that takes a technical question, \
and responds with an explanation. \
Respond in streaming."

# here is the question; type over this to ask something new
question = """
Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""

def stream_answer(question):
    stream = openai.chat.completions.create(
        model=MODEL_GPT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role":"user", "content":question}
          ],
        stream=True
    )
    
    response = ""
    display_handle = display(Markdown(""), display_id=True)
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        response = response.replace("```","").replace("markdown", "")
        update_display(Markdown(response), display_id=display_handle.display_id)

# Get gpt-4o-mini to answer, with streaming
stream_answer(question)

my_question = input("Please enter your question:")

# Get Llama 3.2 to answer
response = ollama.chat(
    model=MODEL_LLAMA,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": my_question}
    ]
)
reply = response.get("message", {}).get("content", "No response received")

# Display output in Markdown format
display(Markdown(reply))
