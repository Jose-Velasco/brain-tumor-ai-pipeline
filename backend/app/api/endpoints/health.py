# use for any non-model required routes
from fastapi import APIRouter
from backend.app.schemas.report import ReportRequest

router = APIRouter()

@router.get("/", summary="Health check")
async def health(payload: ReportRequest):
    return {"status": "ok the health routes are working!! :)"}