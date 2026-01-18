from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def generate_summary(text: str, model_name: str, max_new_tokens: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024, padding=True).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True,
        length_penalty=2.0,   # encourages shorter output vs copying
        min_new_tokens=20     # prevents ultra-short outputs
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


generate_summary("On a slow morning that felt longer than it should have, Mira woke up late, stared at the ceiling, and thought about nothing useful. She made coffee, added too much milk, changed her mind, and drank it anyway while scrolling past articles she didn’t read. On the way outside, she forgot why she had left, remembered, then doubted it again. At work, meetings stretched without conclusions, and explanations repeated themselves. By evening, she felt tired but oddly satisfied, convinced the day mattered even though she couldn’t explain why.", "facebook/bart-large-cnn", 100)
# from transformers import T5Tokenizer, T5ForConditionalGeneration

# model_name = 't5-small'
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)

# def summarize(text):
#     inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
#     summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# def process_video_and_summarize (file_path):
#   !ffmpeg -i "{file_path}" output.wav  #extract audio
#   audio = whisper.load_audio("output.wav")
#   audio = whisper.pad_or_trim(audio)

#   # make log-Mel spectrogram and move to the same device as the model
#   mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

#   # detect the spoken language
#   _, probs = model.detect_language(mel)
#   detected_language = max(probs, key=probs.get)
#   print(f"Detected language: {detected_language}")

#   # transcribe the audio
#   result = model.transcribe("output.wav", language=detected_language)

#   # store the recognized text
#   input_text = result["text"]
#   print(summarize(input_text))
