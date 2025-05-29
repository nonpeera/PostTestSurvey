from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class TextAnalysisRequest(BaseModel):
    text: str

class KeywordModel(BaseModel):
    word: str
    count: int
    avg_score: float

class SentimentDistribution(BaseModel):
    positive: int
    neutral: int
    negative: int

class InsightsModel(BaseModel):
    positive_aspects: List[str] = []
    negative_aspects: List[str] = []
    recommendations: List[str] = []

class AnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: str
    filename: str
    total_responses: int
    texts_analyzed: int
    sentiment_distribution: SentimentDistribution
    top_keywords: List[KeywordModel]
    detailed_results: List[Dict[str, Any]]
    insights: InsightsModel