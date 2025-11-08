from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from routers import predict, analyze, allocate, recommend  # import routers

app = FastAPI(title="HeatX | Industrial Energy AI API", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-heatx-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router)
app.include_router(analyze.router)
app.include_router(allocate.router)
app.include_router(recommend.router)

@app.get("/")
async def root():
    return {
        "message": "⚡ HeatX Energy Forecasting API v2.0",
        "status": "running",
        "endpoints": ["/predict", "/analyze", "/recommend-method", "/allocate"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    preview = df.head(6).fillna('-').values.tolist()  # only first 6 rows for UI
    full_data = df.fillna('-').values.tolist()        # full dataset for modeling
    return {"preview": preview, "dataset": full_data}
