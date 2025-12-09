from pydantic import BaseModel, field_validator
from typing import Dict

class PlotRequest(BaseModel):
    movie_id: str
    plot_synopsis: str

    @field_validator('plot_synopsis')
    @classmethod
    def validate_word_count(cls, v: str) -> str:
        # A reasonable minimum for a synopsis to have meaning
        min_words = 20
        word_count = len(v.split())
        if word_count < min_words:
            raise ValueError(f"Plot synopsis must be at least {min_words} words long. Yours is {word_count}.")
        return v

class GenreResponse(BaseModel):
    movie_id: str
    genres: Dict[str, int]  # e.g., {"comedy": 1, "cult": 0...}
    execution_time_ms: float
