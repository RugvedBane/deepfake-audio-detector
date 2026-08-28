# here we will preprocess the data 
# this is the data the user will upload nd will be processed nd passed to model for inference 
import librosa
import numpy as np
import torch
import io

def preprocess_audio(audio_bytes: bytes) -> torch.Tensor:
    audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    mel = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()
    mel_resized = torch.nn.functional.interpolate(mel_tensor, size=(224, 224))
    return mel_resized.squeeze(0)