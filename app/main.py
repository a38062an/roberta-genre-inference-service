from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.schemas import PlotRequest, GenreResponse
from app.service import inference_service

app = FastAPI(title="RoBERTa Genre Classifier", version="1.0")

# Mount static directory (optional, for future CSS/JS files)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('app/static/index.html')

@app.post("/predict", response_model=GenreResponse)
async def predict_genre(request: PlotRequest):
    try:
        if not request.plot_synopsis:
            raise HTTPException(status_code=400, detail="Plot synopsis cannot be empty")

        predictions, exec_time = inference_service.predict(request.plot_synopsis)
        
        return GenreResponse(
            movie_id=request.movie_id,
            genres=predictions,
            execution_time_ms=round(exec_time, 2)
        )
    except Exception as e:
        # Log this error in a real app
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(inference_service.model.device)}
