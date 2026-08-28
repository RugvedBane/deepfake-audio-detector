from fastapi import FastAPI, UploadFile, File
import time 
from model import predict
from database import init_db, log_prediction, get_stats
from youtube import extract_audio_from_youtube

app = FastAPI()
init_db()

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
async def predict_audio(
    file: UploadFile = File(...),
    input_method: str = 'upload'
):
    start = time.time()
    audio_bytes = await file.read()
    result = predict(audio_bytes)
    latency = (time.time() - start) * 1000



    log_prediction(
        result['prediction'],
        result['confidence'],
        latency,
        input_method
    )
    result['latency_ms'] = round(latency, 2)
    return result

@app.get('/status')
def status():
    return {'predictions': get_stats()}

@app.post("/predict-youtube")
async def predict_youtube(url: str):
    import time
    start = time.time()
    
    audio_bytes = extract_audio_from_youtube(url)
    result = predict(audio_bytes)
    latency = (time.time() - start) * 1000
    
    log_prediction(result["prediction"], result["confidence"], latency, "youtube")
    result["latency_ms"] = round(latency, 2)
    return result