from datetime import date
from pydantic import BaseModel, Field, model_validator


class PartialTripRequest(BaseModel):
    """What the LLM fills in. Everything optional; no guessing allowed."""
    origin: str | None = None
    destination: str | None = None
    duration_days: int | None = None
    travelers: int | None = None
    departure_date: date | None = None
    return_date: date | None = None
    dates_undecided: bool = False      # user said "no dates yet / flexible"


class TripRequest(BaseModel):
    """Strict model handed to the downstream tools."""
    origin: str
    destination: str
    duration_days: int = Field(gt=0)
    travelers: int = Field(gt=0)
    departure_date: date | None = None
    return_date: date | None = None

    @model_validator(mode="after")
    def check_dates(self):
        if self.departure_date and self.return_date:
            span = (self.return_date - self.departure_date).days
            if span <= 0:
                raise ValueError("The return date must be after the departure date.")
            if span != self.duration_days:
                raise ValueError(
                    f"Those dates span {span} days, but you said {self.duration_days} days."
                )
        return self