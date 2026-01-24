import whisper
import subprocess
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from model_t5 import generate_summary
from preprocessing import preprocess_video

class VideoSummarizer:
    def __init__(self, model_name):
        
        # load the whisper model
        self.model_whisper = whisper.load_model("turbo")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float).to(self.device)
        self.model.eval()

    def summarize(self, video_path: str, max_new_tokens: int) -> None:
        input = preprocess_video(video_path, model = self.model_whisper)
        summary = generate_summary(input, self.model, self.tokenizer, self.device, max_new_tokens) 
        return summary
