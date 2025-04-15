 imports
import os
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoProcessor, AutoModelForSpeechSeq2Seq
import gradio as gr

# Log in to HuggingFace
login(token=os.getenv("HF_TOKEN"))

# quantization setup
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# constants
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(LLAMA, use_auth_token=os.getenv("HF_TOKEN"), device_map="cuda:0", quantization_config=quant_config)


# load Speech model, read audio file and convert to text
def transcript_audio(audio_file):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(audio_file, return_timestamps=True)
    return result


# promopts and summarizing the text
def summarize_with_llama(transcript, context="Phone Call", language="English"):
    if context == "Phone Call":
        system_message = (
            "You are an assistant that produces minutes of phone calls from transcripts, "
            "with summary and key discussion points, in markdown."
        )
        user_prompt = (
            "Below is an extract transcript of a phone call. "
            "Please write minutes in markdown, including a summary, location, and discussion points."
        )
    elif context == "Meeting":
        system_message = (
            "You are an assistant that produces minutes of meetings from transcripts, "
            "with summary, key discussion points, takeaways, and action items with owners, in markdown."
        )
        user_prompt = (
            "Below is an extract transcript of a meeting. "
            "Please write minutes in markdown, including a summary with attendees, location, and date; "
            "discussion points; takeaways; and action items with owners."
        )
    else:
        raise ValueError(f"Unknown context: {context}")

    # Add language instruction
    if language.lower() == "hebrew":
        user_prompt += "\n\nWrite the entire generatated summary and points in Hebrew."
    elif language.lower() == "english":
        user_prompt += "\n\nWrite the entire summary in English."
    else:
        raise ValueError(f"Unsupported language: {language}")

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{system_message}\n\n{user_prompt}\n\nTranscript:\n{transcript}"}
    ]

    input_features = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    output_ids = model.generate(input_features, max_new_tokens=2000)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return summary


# combining the two functions
def transcribe_and_summarize(audio, choices, language):
    try:
        result = transcript_audio(audio)
        transcript = result["text"]
        return summarize_with_llama(transcript, context=choices, language=language)
    except Exception as e:
        return f"**Error:** {str(e)}"
        

# Gradio UI setup
demo = gr.Interface(
    fn=transcribe_and_summarize,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio"),
        gr.Radio(["Phone Call", "Meeting"], label="Select Audio Type"),
        gr.Radio(["English", "Hebrew"], label="Select Language Output")
    ],
    outputs=gr.Markdown(height=750),
    title="Audio Summarizer"
     demo.queue().launch()
)
