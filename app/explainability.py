import torch
from pytorch_grad_cam import GradCAM
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from groq import Groq
import os
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

def get_gradcam(model, tensor):
    target_layers = [model.features[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=tensor, targets=targets)

    return grayscale_cam[0]

def analyze_heatmap(grayscale_cam):

    low_freq = grayscale_cam[150:, :].mean()
    mid_freq = grayscale_cam[75:150, :].mean()
    high_freq = grayscale_cam[:75, :].mean()

    time_activation = grayscale_cam.mean(axis=0)
    peak_time_idx = np.argmax(time_activation)
    peak_time =  round(peak_time_idx / 224 * 10, 2)

    regions = []

    if high_freq > 0.4:
        regions.append('high frequency range (consonants/breathing patterns)')
    if mid_freq > 0.4:
        regions.append("mid frequency range (vowel formants)")
    if low_freq > 0.4:
        regions.append("low frequency range (pitch/fundamental frequency)")

    return {
        "regions": regions if regions else ["overall frequency spectrum"],
        "peak_timestamp": peak_time,
        "low_activation": round(float(low_freq), 3),
        "mid_activation": round(float(mid_freq), 3),
        "high_activation": round(float(high_freq), 3)
    }


def generate_explanation(prediction: str, confidence: float, heatmap_analysis: dict) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    regions = ", ".join(heatmap_analysis["regions"])
    peak_time = heatmap_analysis["peak_timestamp"]
    
    prompt = f"""
A deepfake audio detector analyzed an audio clip and determined it is {prediction} with {confidence}% confidence.

The analysis found suspicious patterns in: {regions}
Peak anomaly was detected at {peak_time} seconds into the audio.

Write 2-3 sentences explaining this result in simple, non-technical language for a regular person.
- If fake: explain what specific audio patterns gave it away
- If real: explain what natural characteristics were detected
Do not use words like "mel spectrogram", "neural network", or "model".
Be specific about the frequency regions mentioned.
"""
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        reasoning_effort="low" 
    )

    return response.choices[0].message.content
