from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

def generate_summary(text : str, model_name: str) -> str:

    # model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    def summarize(text):
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # def process_video_and_summarize (file_path):
    #     !ffmpeg -i "{file_path}" output.wav  #extract audio
    #     audio = whisper.load_audio("output.wav")
    #     audio = whisper.pad_or_trim(audio)

    # # make log-Mel spectrogram and move to the same device as the model
    # mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # # detect the spoken language
    # _, probs = model.detect_language(mel)
    # detected_language = max(probs, key=probs.get)
    # print(f"Detected language: {detected_language}")

    # # transcribe the audio
    # result = model.transcribe("output.wav", language=detected_language)

    # # store the recognized text
    # input_text = result["text"]
    print(summarize(text))


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

generate_function("Snow White is a kind princess whose beauty makes her stepmother, the Evil Queen, jealous. When the Queen orders her killed, Snow White escapes into the forest and finds shelter with the Seven Dwarfs. The Queen later disguises herself and poisons Snow White with an apple, causing her to fall into a deep sleep. In the end, a prince awakens Snow White, good triumphs over jealousy, and Snow White begins a happy new life.", "t5-small")