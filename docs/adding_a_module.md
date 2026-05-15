# Adding a new module

This is the whole point of the architecture: shipping a new medical capability should be a folder-sized change, not a refactor. Here's the recipe.

## Example: adding a "Hypertension Risk" module

### 1. Create the folder

```
backend/modules/hypertension/
├── __init__.py        # Exposes the BaseModule subclass
├── router.py          # FastAPI routes
├── service.py         # ML inference logic
├── schemas/
│   ├── __init__.py
│   └── hypertension.py
└── ml_models/         # Reserved for model artifacts (usually empty in repo)
```

### 2. Define request/response schemas (`schemas/hypertension.py`)

```python
from pydantic import BaseModel, Field

class HypertensionRequest(BaseModel):
    age: int = Field(ge=1, le=120)
    systolic_bp: float = Field(ge=60, le=250)
    diastolic_bp: float = Field(ge=30, le=150)
    bmi: float = Field(ge=10, le=70)
    cholesterol: float = Field(ge=80, le=400)
    smoker: bool
    family_history: bool

class HypertensionResponse(BaseModel):
    risk_score: float
    risk_level: str
    recommendations: list[str]
    disclaimer: str
```

### 3. Implement the service (`service.py`)

Mirror `manual_diagnosis/diabetes_service.py`:
- Load model in `.load()`
- Implement `.predict(req)`
- Log predictions to the `PredictionLog` table (audit trail)

### 4. Wire up the router (`router.py`)

```python
from fastapi import APIRouter
from backend.modules.hypertension.service import service
from backend.modules.hypertension.schemas.hypertension import (
    HypertensionRequest, HypertensionResponse
)

router = APIRouter()

@router.post("/predict", response_model=HypertensionResponse)
async def predict(req: HypertensionRequest):
    return service.predict(req)
```

### 5. Register the module (`__init__.py`)

```python
from fastapi import APIRouter

from backend.core.base_module import BaseModule
from backend.modules.hypertension.service import service
from backend.modules.hypertension.router import router

class HypertensionModule(BaseModule):
    name = "hypertension"
    version = "0.1.0"
    description = "Hypertension risk estimation from vitals & lifestyle."
    tags = ["diagnosis"]

    def get_router(self) -> APIRouter:
        return router

    def on_startup(self) -> None:
        service.load()
        self._ready = service.is_ready()
```

### 6. Restart the backend

That's it. The registry auto-discovers the module, mounts its router at `/api/v1/hypertension/*`, and lists it in `/api/v1/modules`.

### 7. (Optional) Add a frontend page

Drop a file in `frontend/pages/`:

```
frontend/pages/2_Hypertension_Risk.py
```

Streamlit auto-detects it and adds it to the sidebar.

## Training the model

If your model needs deep learning or a big dataset, write a Colab notebook in `ml_training/`:

```
ml_training/
└── train_hypertension.ipynb
```

The notebook should:
1. Pull data (from Kaggle / a public mirror)
2. Train with the free GPU
3. Export model (joblib for sklearn, ONNX for deep learning models)
4. Push to Hugging Face Hub:
   ```python
   from huggingface_hub import HfApi
   api = HfApi(token=HF_TOKEN)
   api.upload_file(
       path_or_fileobj="hypertension_model.onnx",
       path_in_repo="hypertension_model.onnx",
       repo_id="your-username/medimind-hypertension",
       repo_type="model",
   )
   ```
5. The service's `.load()` downloads from HF Hub (cached locally).

For small tabular models, training locally is fine — see `scripts/train_diabetes_model.py`.

## Checklist before merging a module

- [ ] Pydantic schemas with `Field(...)` descriptions for every field (these become API docs)
- [ ] Service returns a `disclaimer` field in every prediction response
- [ ] `predict()` logs to `PredictionLog` (input hashed, not raw)
- [ ] Unit tests in `backend/tests/test_<module>.py`
- [ ] At least one happy-path test and one input-validation test
- [ ] Docs: a short `docs/modules/<module>.md` explaining the model, dataset, metrics, and limitations
- [ ] No raw patient data committed to the repo
- [ ] Module's `health_check()` reports whether the model loaded
