import torch
import torch.nn as nn
import torchvision.models as models
from preprocess import preprocess_audio
from explainability import get_gradcam, analyze_heatmap, generate_explanation

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# we have again built the same architecture bc the .pth file stores only weight so we again need to built the architecture for inference
def built_model():
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.features[0][0] = nn.Conv2d(
        in_channels=1, 
        out_channels=32,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    )
    model.classifier[1] = nn.Linear(
        in_features=1280,
        out_features=1
    )

    return model

device = torch.device('cpu')
model = built_model()
model.load_state_dict(torch.load('../models/best_model.pth', map_location=device))
model.eval()

import time

def predict(audio_bytes: bytes):
    total_start = time.perf_counter()

    start = time.perf_counter()
    tensor = preprocess_audio(audio_bytes)
    print(f"PREPROCESS: {time.perf_counter() - start:.2f}s")

    tensor = tensor.unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
    print(f"MODEL INFERENCE: {time.perf_counter() - start:.2f}s")

    prediction = "fake" if prob >= 0.4 else "real"
    confidence = round(prob * 100, 2)

    start = time.perf_counter()
    heatmap = get_gradcam(model, tensor)
    print(f"GRADCAM: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    heatmap_analysis = analyze_heatmap(heatmap)
    print(f"HEATMAP ANALYSIS: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    explanation = generate_explanation(
        prediction,
        confidence,
        heatmap_analysis
    )
    print(f"GROQ: {time.perf_counter() - start:.2f}s")

    print(f"TOTAL: {time.perf_counter() - total_start:.2f}s")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "raw_score": round(prob, 4),
        "explanation": explanation,
        "heatmap_analysis": heatmap_analysis
    }