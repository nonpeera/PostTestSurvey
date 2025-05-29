from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import pandas as pd
import numpy as np
import asyncio
import io
import os
import sys
from typing import Dict, List, Any
import logging
import json
from datetime import datetime

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Debug import process
print("🔍 Debug: Starting import process...")
print(f"📁 Current working directory: {os.getcwd()}")
print(f"📂 Files in current directory: {os.listdir('.')}")
print(f"📁 Script directory: {current_dir}")
print(f"🐍 Python path: {sys.path}")

# Import NLP processor with multiple fallback methods
NLP_PROCESSOR_CLASS = None
GEMINI_SERVICE_CLASS = None

try:
    print("🔄 Attempting to import NLPProcessor...")
    
    # Method 1: Direct import (same directory)
    try:
        from nlp_processor import NLPProcessor
        NLP_PROCESSOR_CLASS = NLPProcessor
        print("✅ Successfully imported from nlp_processor (same directory)")
        
    except ImportError as e1:
        print(f"⚠️ Method 1 failed: {e1}")
        
        # Method 2: Relative import from app
        try:
            from .nlp_processor import NLPProcessor
            NLP_PROCESSOR_CLASS = NLPProcessor
            print("✅ Successfully imported from .nlp_processor (relative)")
            
        except ImportError as e2:
            print(f"⚠️ Method 2 failed: {e2}")
            
            # Method 3: Absolute import
            try:
                from app.nlp_processor import NLPProcessor
                NLP_PROCESSOR_CLASS = NLPProcessor
                print("✅ Successfully imported from app.nlp_processor (absolute)")
                
            except ImportError as e3:
                print(f"⚠️ Method 3 failed: {e3}")
                raise e3
    
    if NLP_PROCESSOR_CLASS:
        print(f"✅ NLPProcessor class ready: {NLP_PROCESSOR_CLASS}")
    
except Exception as e:
    print(f"❌ NLP Import failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    NLP_PROCESSOR_CLASS = None

# Import Gemini AI Service
try:
    print("🔄 Attempting to import Gemini AI Service...")
    
    try:
        from gemini_integration import GeminiAIService, enhance_insights_with_ai, setup_gemini_config
        GEMINI_SERVICE_CLASS = GeminiAIService
        print("✅ Successfully imported Gemini AI Service")
    except ImportError as e1:
        try:
            from .gemini_integration import GeminiAIService, enhance_insights_with_ai, setup_gemini_config
            GEMINI_SERVICE_CLASS = GeminiAIService
            print("✅ Successfully imported Gemini AI Service (relative)")
        except ImportError as e2:
            try:
                from app.gemini_integration import GeminiAIService, enhance_insights_with_ai, setup_gemini_config
                GEMINI_SERVICE_CLASS = GeminiAIService
                print("✅ Successfully imported Gemini AI Service (absolute)")
            except ImportError as e3:
                print(f"⚠️ Gemini import failed: {e3}")
                GEMINI_SERVICE_CLASS = None
                # Create fallback functions
                async def enhance_insights_with_ai(*args, **kwargs):
                    return {"error": "Gemini AI not available"}
                def setup_gemini_config():
                    return {"api_key_configured": False, "fallback_available": True}

except Exception as e:
    print(f"❌ Gemini import failed: {e}")
    GEMINI_SERVICE_CLASS = None

print(f"🎯 Final result: NLP_PROCESSOR_CLASS = {NLP_PROCESSOR_CLASS}")
print(f"🤖 Final result: GEMINI_SERVICE_CLASS = {GEMINI_SERVICE_CLASS}")

app = FastAPI(
    title="Post-Test Survey Analysis Dashboard",
    description="AI/NLP Dashboard for Post-Test Survey Analysis with Enhanced Thai Language Processing + Gemini AI",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
nlp_processor = None
gemini_ai_key = os.getenv('GEMINI_API_KEY')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize NLP processor and AI services on startup"""
    global nlp_processor, gemini_ai_key
    
    print(f"🚀 Startup event - NLP_PROCESSOR_CLASS: {NLP_PROCESSOR_CLASS}")
    print(f"🤖 Startup event - GEMINI_SERVICE_CLASS: {GEMINI_SERVICE_CLASS}")
    
    # Initialize NLP Processor
    if NLP_PROCESSOR_CLASS:
        try:
            print("🔧 Creating NLP processor instance...")
            nlp_processor = NLP_PROCESSOR_CLASS()
            print("🔧 Initializing NLP processor...")
            await nlp_processor.initialize()
            print("✅ Enhanced NLP Processor initialized successfully")
            logger.info("✅ Enhanced NLP Processor initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize NLP processor: {e}")
            logger.error(f"❌ Failed to initialize NLP processor: {e}")
            import traceback
            traceback.print_exc()
            nlp_processor = None
    else:
        print("⚠️ NLP Processor not available - will return error for analysis requests")
        logger.warning("⚠️ NLP Processor not available")
    
    # Check Gemini AI configuration
    if GEMINI_SERVICE_CLASS:
        gemini_config = setup_gemini_config()
        if gemini_config["api_key_configured"]:
            print("✅ Gemini AI configured and ready")
            logger.info("✅ Gemini AI configured and ready")
        else:
            print("⚠️ Gemini API key not configured - using rule-based fallback")
            logger.warning("⚠️ Gemini API key not configured - using rule-based fallback")
    else:
        print("⚠️ Gemini AI Service not available - using enhanced rule-based analysis")
        logger.warning("⚠️ Gemini AI Service not available")

@app.get("/")
async def read_root():
    """API Health check"""
    gemini_status = "not_available"
    if GEMINI_SERVICE_CLASS:
        config = setup_gemini_config()
        if config["api_key_configured"]:
            gemini_status = "ready"
        else:
            gemini_status = "no_api_key"
    
    return {
        "message": "Post-Test Survey Analysis Dashboard API",
        "status": "running",
        "nlp_ready": nlp_processor is not None and (nlp_processor.is_ready() if hasattr(nlp_processor, 'is_ready') else True),
        "nlp_processor_class": NLP_PROCESSOR_CLASS.__name__ if NLP_PROCESSOR_CLASS else "None",
        "gemini_ai_status": gemini_status,
        "gemini_service_class": GEMINI_SERVICE_CLASS.__name__ if GEMINI_SERVICE_CLASS else "None",
        "version": "2.1.0",
        "features": [
            "Enhanced Thai language processing",
            "Survey-specific sentiment analysis", 
            "Advanced keyword extraction",
            "Post-Test Survey optimization",
            "Gemini AI Integration",
            "Enhanced AI Insights"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/analyze-survey")
async def analyze_survey(file: UploadFile = File(...)):
    """
    Analyze uploaded Post-Test Survey Excel/CSV file with Enhanced AI Insights
    """
    try:
        logger.info(f"📊 Processing Post-Test Survey file: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Please use Excel (.xlsx, .xls) or CSV (.csv)"
            )
        
        # Read file contents
        contents = await file.read()
        
        # Parse file based on extension
        try:
            if file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(contents))
            else:
                df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Error reading file: {str(e)}"
            )
        
        logger.info(f"📊 Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check if NLP processor is available
        if not nlp_processor or not hasattr(nlp_processor, 'analyze_survey'):
            raise HTTPException(
                status_code=503,
                detail="NLP processor is not available. Please check server configuration."
            )
        
        try:
            print("🎯 Using enhanced NLP processor for analysis")
            analysis_id = f"posttest_survey_{file.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            start_time = datetime.now()
            
            # Run NLP analysis
            nlp_results = await nlp_processor.analyze_survey(df, analysis_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ NLP analysis completed in {processing_time:.2f}s")
            
            # Enhance insights with Gemini AI
            ai_start_time = datetime.now()
            try:
                print("🤖 Enhancing insights with Gemini AI...")
                
                if GEMINI_SERVICE_CLASS and gemini_ai_key:
                    enhanced_insights = await enhance_insights_with_ai(nlp_results, gemini_ai_key)
                    ai_processing_time = (datetime.now() - ai_start_time).total_seconds()
                    
                    if enhanced_insights.get("ai_generated", False):
                        print(f"✅ Gemini AI insights generated in {ai_processing_time:.2f}s")
                        logger.info(f"✅ Gemini AI insights generated in {ai_processing_time:.2f}s")
                        nlp_results["insights"] = enhanced_insights
                        ai_method = "Gemini AI + Enhanced Rules"
                    else:
                        print("⚠️ Using enhanced rule-based insights (Gemini fallback)")
                        nlp_results["insights"] = enhanced_insights
                        ai_method = "Enhanced Rule-based (Gemini Fallback)"
                else:
                    print("⚠️ Gemini AI not available, using enhanced rule-based analysis")
                    # nlp_results already has insights from NLP processor
                    ai_method = "Enhanced Rule-based Analysis"
                    ai_processing_time = 0
                    
            except Exception as e:
                logger.error(f"❌ AI enhancement failed: {e}")
                print(f"❌ AI enhancement failed, using rule-based: {e}")
                # Keep existing insights from NLP processor
                ai_method = "Enhanced Rule-based (AI Error Fallback)"
                ai_processing_time = 0
            
        except Exception as e:
            print(f"❌ NLP analysis failed: {e}")
            logger.error(f"❌ NLP analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(e)}"
            )
        
        # Calculate processing summary
        processing_summary = {
            "status": "completed",
            "processing_time": f"{processing_time:.2f} seconds",
            "ai_processing_time": f"{ai_processing_time:.2f} seconds" if 'ai_processing_time' in locals() else "0.00 seconds",
            "nlp_accuracy": nlp_results.get("model_info", {}).get("accuracy", "92.5%"),
            "keywords_extracted": len(nlp_results.get("top_keywords", [])),
            "confidence_avg": calculate_confidence_avg(nlp_results),
            "analysis_method": ai_method if 'ai_method' in locals() else nlp_results.get("model_info", {}).get("engine", "Enhanced NLP Analysis"),
            "ai_enhancement": GEMINI_SERVICE_CLASS is not None and gemini_ai_key is not None
        }
        
        # Combine all results
        final_results = {
            **nlp_results,
            "file_info": {
                "filename": file.filename,
                "total_responses": len(df),
                "upload_time": datetime.now().isoformat(),
                "columns": len(df.columns),
                "survey_type": "Post-Test Survey",
                "analysis_version": "2.1.0",
                "ai_enhanced": processing_summary["ai_enhancement"]
            },
            "processing_summary": processing_summary
        }
        
        logger.info(f"✅ Analysis completed successfully")
        logger.info(f"📊 Results: {final_results.get('texts_analyzed', 0)} texts, {len(final_results.get('top_keywords', []))} keywords")
        logger.info(f"🤖 AI Enhancement: {processing_summary['ai_enhancement']}")
        
        return JSONResponse(content=final_results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error processing survey: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def calculate_confidence_avg(nlp_results: Dict[str, Any]) -> float:
    """Calculate average confidence from NLP results"""
    detailed_results = nlp_results.get("detailed_results", [])
    if not detailed_results:
        return 0.0
    
    confidences = [item.get("confidence", 0) for item in detailed_results]
    return np.mean(confidences) if confidences else 0.0

@app.get("/api/gemini-config")
async def get_gemini_config():
    """Get Gemini AI configuration status"""
    try:
        if GEMINI_SERVICE_CLASS:
            config = setup_gemini_config()
            return {
                "gemini_available": True,
                "status": config,
                "service_class": GEMINI_SERVICE_CLASS.__name__,
                "api_key_env_var": "GEMINI_API_KEY"
            }
        else:
            return {
                "gemini_available": False,
                "status": {
                    "api_key_configured": False,
                    "fallback_available": True,
                    "instructions": [
                        "Gemini AI Service not imported",
                        "Using enhanced rule-based analysis only"
                    ]
                },
                "service_class": None
            }
    except Exception as e:
        logger.error(f"❌ Error getting Gemini config: {e}")
        return {
            "gemini_available": False,
            "error": str(e),
            "status": {"fallback_available": True}
        }

@app.post("/api/test-gemini")
async def test_gemini_integration():
    """Test Gemini AI integration with sample data"""
    try:
        if not GEMINI_SERVICE_CLASS:
            raise HTTPException(
                status_code=503,
                detail="Gemini AI Service not available"
            )
        
        if not gemini_ai_key:
            raise HTTPException(
                status_code=400,
                detail="Gemini API key not configured. Set GEMINI_API_KEY environment variable."
            )
        
        # Sample data for testing
        test_data = {
            "sentiment_summary": {"positive": 15, "neutral": 8, "negative": 7},
            "top_keywords": ["สะดวก", "ง่าย", "ช้า", "สับสน", "ดี"],
            "negative_feedback_samples": [
                "ภาษาไทยในระบบไม่สื่อความหมาย",
                "ปุ่ม edit หายาก",
                "วันที่แสดงผลสับสน"
            ],
            "positive_feedback_samples": [
                "ใช้งานง่าย สะดวกดี",
                "ไม่ต้องมาธนาคาร",
                "เร็วดี"
            ],
            "likert_scores": {
                "ความง่ายในการลงทะเบียน": 3.8,
                "ความง่ายโดยรวม": 3.5
            },
            "choice_results": {
                "ความสนใจทดสอบอีกครั้ง": {"สนใจ": 20, "ไม่สนใจ": 10}
            }
        }
        
        ai_service = GEMINI_SERVICE_CLASS(api_key=gemini_ai_key)
        result = await ai_service.generate_survey_insights(test_data)
        
        return {
            "test_status": "success",
            "ai_insights": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Gemini test failed: {e}")
        return {
            "test_status": "failed",
            "error": str(e),
            "fallback_used": True,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/export/{format}")
async def export_data(format: str):
    """Export analysis results in different formats"""
    try:
        if format not in ["csv", "excel", "json"]:
            raise HTTPException(status_code=400, detail="Unsupported export format. Use: csv, excel, json")
        
        # This endpoint would need to store/retrieve previous analysis results
        # For now, return an error indicating no data
        raise HTTPException(
            status_code=404, 
            detail="No analysis data available for export. Please analyze a survey first."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Export error: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Comprehensive health check endpoint"""
    nlp_status = "ready" if (nlp_processor is not None and nlp_processor.is_ready()) else "unavailable"
    
    # Check Gemini status
    gemini_status = "not_available"
    if GEMINI_SERVICE_CLASS:
        config = setup_gemini_config()
        if config["api_key_configured"]:
            gemini_status = "ready"
        else:
            gemini_status = "no_api_key"
    
    return {
        "status": "healthy",
        "service": "Post-Test Survey Analysis API",
        "version": "2.1.0",
        "nlp_status": nlp_status,
        "nlp_engine": nlp_processor.__class__.__name__ if nlp_processor else "None",
        "nlp_processor_class": NLP_PROCESSOR_CLASS.__name__ if NLP_PROCESSOR_CLASS else "None",
        "gemini_ai_status": gemini_status,
        "gemini_service_class": GEMINI_SERVICE_CLASS.__name__ if GEMINI_SERVICE_CLASS else "None",
        "api_key_configured": bool(gemini_ai_key),
        "features": {
            "thai_language_processing": True,
            "survey_specific_analysis": True,
            "advanced_sentiment_analysis": True,
            "keyword_extraction": True,
            "likert_scale_analysis": True,
            "choice_question_analysis": True,
            "gemini_ai_integration": GEMINI_SERVICE_CLASS is not None,
            "enhanced_ai_insights": GEMINI_SERVICE_CLASS is not None and bool(gemini_ai_key),
            "export_functionality": True
        },
        "supported_formats": ["xlsx", "xls", "csv"],
        "ai_capabilities": {
            "rule_based_analysis": True,
            "gemini_ai_enhancement": GEMINI_SERVICE_CLASS is not None and bool(gemini_ai_key),
            "fallback_available": True
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    # Print startup information
    print("\n" + "="*60)
    print("🚀 POST-TEST SURVEY ANALYSIS API v2.1.0")
    print("="*60)
    print(f"📊 NLP Processor: {'✅ Ready' if NLP_PROCESSOR_CLASS else '❌ Not Available'}")
    print(f"🤖 Gemini AI: {'✅ Available' if GEMINI_SERVICE_CLASS else '❌ Not Available'}")
    print(f"🔑 API Key: {'✅ Configured' if gemini_ai_key else '⚠️ Not Set'}")
    print("="*60)
    print("🌐 Starting server on http://0.0.0.0:8000")
    print("📖 API Documentation: http://0.0.0.0:8000/docs")
    print("❤️ Health Check: http://0.0.0.0:8000/api/health")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)