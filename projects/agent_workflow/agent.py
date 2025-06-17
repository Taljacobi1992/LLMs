from dotenv import load_dotenv
from typing import List, Tuple, Dict
from agents import Agent, Runner, trace
import gradio as gr

load_dotenv(override=True)

# Define the agent that validates if a response is meaningful
validation_agent = Agent(
    name="ResponseValidator",
    model="gpt-4o-mini",
    instructions=(
        "Your job is to check if a ###user's answer is meaningful and specific to the ###Question. "
        "Respond ONLY with 'yes' if it's acceptable, or 'no' if it's vague (e.g., 'not sure', 'maybe', 'I don't know')."
    )
)

# Define the support agent who asks the questions
support_agent = Agent(
    name="BreastCancerSupportAgent",
    model="gpt-4o-mini",
    instructions=(
        "You are a supportive, empathetic assistant for women recently diagnosed with breast cancer. "
        "You ask 5 specific questions in order. If a user gives a vague or unclear answer, gently ask again. "
        "Be emotionally sensitive and thank the user for each answer."
    )
)

# Define the support agent who asks the questions
gen_coach = Agent(
    name="gen_coach",
    model="gpt-4o-mini",
    instructions = """
                    You are a compassionate support agent assisting a woman recently diagnosed with breast cancer.
                    - Be kind, gentle, and present. You're here to walk beside them, not interrogate or fix them.
                    - Be emotionally sensitive"
                    """
)

health_coach = Agent(
    name="health_coach",
    model="gpt-4o-mini",
    instructions = """
        You are a compassionate support agent for women recently diagnosed with breast cancer.
        
        Your role is to provide emotional support, helpful guidance, and personalized questions to gently explore their current situation, needs, and feelings.
        
        Always prioritize empathy and emotional sensitivity. Be patient if a user is hesitant, vague, or doesn't answer right away—gently follow up with polite persistence and reassurance.
        
        You will:
        
        - Ask the user personalized, emotionally supportive questions about their diagnosis, treatment, and support system.
        - If provided with a condition profile (including age, diagnosis type, treatment stage, and next steps), generate 5 personalized, relevant follow-up questions.
        - Thank the user when they share something, and validate their emotions.
        - If the user deflects, rephrase the question kindly, showing understanding and giving them space.
        - Avoid medical advice, but encourage users to speak to their doctor for clinical guidance.
        - NEVER rush or overwhelm the user—take a slow, safe, and thoughtful approach to each conversation.
       
        End each interaction with one personalized follow-up question based on what the user shared. This helps deepen trust and support.
        
        Be kind, gentle, and present. You're here to walk beside them, not interrogate or fix them.
        """
)


# Questions with keys for storing answers
questions = [
    {"key": "name", "question": "Hi, I'm here to support you. May I ask your first name?"},
    {"key": "age", "question": "How old are you, if you don't mind sharing?"},
    {"key": "diagnosis", "question": "Could you tell me the type (must include TNBC,IDC,ILC,IBC etc..) of breast cancer you've been diagnosed with?"},
    {"key": "treatment", "question": "Are you currently receiving any treatment, tell me the details?"},
    {"key": "support_system", "question": "Do you have a support system—like friend, family, or a community—to help you through this?"}
]

# Session memory
session_state = {
    "current_question_index": 0,
    "profile_question_index": 0,
    "responses": {}
}

old_resp = []

def get_profile_qstn(msg,indx, responses):
    name = responses.get("name", "you").capitalize()
    treatment = responses.get("treatment", "your treatment")
    support = responses.get("support_system", "your support system")

    profile_questions = [
        f"{name}, would you like to share how you're feeling about starting {treatment}?",
        f"Have you been offered genetic counseling yet, or is that part of your next steps?",
        f"Do you feel informed about the typical side effects of {treatment} and how to manage them?",
        f"Are you feeling supported emotionally right now?",
        f"Is there anything you're anxious about in the next few weeks??"
    ]
    return str(msg) + profile_questions[indx]
    
# Main conversation function using agentic validation
async def agent_converse(message: str, history: List[Tuple[str, str]]) -> str:
    idx = session_state["current_question_index"]
    p_idx = session_state["profile_question_index"]

    if p_idx==5:
        GenQuestion = await Runner.run(gen_coach,message)
        
        return GenQuestion.final_output
    
    if idx >= len(questions):
        responses = session_state["responses"]

        if len(old_resp)==0:
            ProfileQuestion = await Runner.run(health_coach,get_profile_qstn({},p_idx,responses))
        else:
            ProfileQuestion = await Runner.run(health_coach,get_profile_qstn(old_resp[-1],p_idx,responses))
            
        p_idx = p_idx + 1
        session_state["profile_question_index"] = p_idx

        ### Memory ###
        qa_pair = {'User': message,'Agent': ProfileQuestion.final_output}
        old_resp.append(qa_pair)
        ### ### ######
        
        return ProfileQuestion.final_output
        # else:

    else:

        responses = session_state["responses"]
    
        # First time: ask the first question
        if len(history) == 0 and idx == 0:
            return questions[0]["question"]
    
        # Get current question/key
        key = questions[idx]["key"]
        question_text = questions[idx]["question"]
    
        # Validate the user response using the validation agent
        validation_result = await Runner.run(validation_agent,
            input=f"###Question: {question_text} \n ###User's answer: {message}\n. Is it meaningful and specific?"
        )
    
        if "yes" in validation_result.final_output.lower():
            responses[key] = message.strip()
            idx += 1
            session_state["current_question_index"] = idx
    
            if idx >= len(questions):
                return "Thank you for sharing. You're not alone in this. Let's continue with some personalized support."
    
            next_question = questions[idx]["question"]
            next_question = await Runner.run(support_agent,next_question)
            return next_question.final_output
    
        else:
            # Vague response: re-ask with empathy
            repeat_question = await Runner.run(support_agent,f"Seems invalid, You can take your time, If you're okay with it, {question_text}")
            return repeat_question.final_output

chat_interface = gr.ChatInterface(fn=agent_converse, title="Breast Cancer Support Agent")
chat_interface.launch()
