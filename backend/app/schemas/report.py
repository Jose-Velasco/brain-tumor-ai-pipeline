from pydantic import BaseModel
from typing import Any

class ReportRequest(BaseModel):
    case_report: dict[str, Any]
    include_images: bool = False

class ReportResponse(BaseModel):
    text: str