##  Breast Cancer Support Agent

This is a compassionate conversational assistant designed to support women recently diagnosed with breast cancer. The system combines multiple specialized AI agents to provide emotionally sensitive, personalized guidance, and validate meaningful responses through agentic collaboration.



###  Key Features

Supportive Conversational Agent: Gently asks five key questions to understand the user's situation with empathy and patience.

Response Validation: Uses an AI-based validation agent to determine whether user responses are meaningful or vague, and prompts rephrasing when needed.

Personalized Follow-Ups: Once the basic profile is collected, a health coach agent provides deeply supportive follow-up questions.

Asynchronous Agent Execution: Powered by OpenAI's agents SDK for multi-agent orchestration.

Web Interface: Simple Gradio-powered chat interface for easy interaction.



###  Agents Used

ResponseValidator – Checks whether a user's answer is specific and meaningful.

BreastCancerSupportAgent – Asks the core set of 5 structured questions.

health_coach – Provides personalized, emotionally sensitive follow-up questions.

gen_coach – Offers a final reflective question based on all previous input.



###  Setup

1. Install packages from the requirements.txt
2. Add your API key by creating a file named .env in the same directory as the script and add: OPENAI_API_KEY=your-openai-api-key
3. Run final_v1.ipynb


## Please Note
In this project, gpt-4o-mini was used for all agents to reduce API usage costs while maintaining reasonable performance. Feel free to switch to a more advance model if needed.
