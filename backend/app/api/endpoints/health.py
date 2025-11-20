# use for any non-model required routes
from fastapi import APIRouter

router = APIRouter()

@router.get("/", summary="Health check")
async def health():
    return {"status": "ok"}