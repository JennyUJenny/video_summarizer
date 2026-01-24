import whisper 
import subprocess

def extract_audio(video_path, output_path="output.wav"):
    command = [
        "ffmpeg",
        "-y", # Add -y flag to overwrite output file without asking
        "-i", video_path,
        output_path
    ]
    subprocess.run(command, check=True)

def preprocess_video(video_path, model):
    # # load the whisper model
    # model = whisper.load_model("turbo")
    
    #extract audio
    extract_audio(video_path)  
    audio = whisper.load_audio("output.wav")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language}")

    # transcribe the audio
    result = model.transcribe("output.wav", language=detected_language)

    # store the recognized text
    return result["text"]

