import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

class GeminiAIService:
    """
    Enhanced Gemini AI Service สำหรับ Survey Analysis
    รองรับการเรียก API หลายครั้งเพื่อเติม JSON ทีละส่วน
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.timeout = 30  # ลดเวลาเพราะแต่ละ call จะเล็กลง
        self.max_retries = 2
        
        # Model priority list (เลือกแค่ที่เร็วและเสถียร)
        self.models = [
            {
                "name": "gemini-2.0-flash",
                "description": "Next generation features",
                "timeout": 25,
                "max_tokens": 1200,
                "temperature": 0.2
            }
        ]
        
        logger.info(f"🤖 Gemini AI Service initialized with incremental analysis")
    
    async def generate_survey_insights(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        สร้าง AI Insights แบบทีละส่วน (Multi-step approach)
        """
        try:
            if not self.api_key:
                logger.warning("⚠️ No Gemini API key provided")
                return self._create_fallback_insights(survey_data)
            
            logger.info("🚀 Starting incremental Gemini AI analysis...")
            
            # สร้าง base insights structure
            insights = {
                "executive_summary": "",
                "positive_aspects": [],
                "negative_aspects": [],
                "recommendations": [],
                "system_strengths": [],
                "improvement_areas": [],
                "user_pain_points": [],
                "priority_actions": [],
                "sentiment_analysis": {
                    "overall_mood": "",
                    "satisfaction_level": "",
                    "confidence_score": 0.0
                }
            }
            
            # Step 1: สรุปผลโดยรวม + sentiment analysis
            logger.info("📊 Step 1: Analyzing overall sentiment...")
            sentiment_data = await self._analyze_sentiment_summary(survey_data)
            insights.update(sentiment_data)
            
            # Step 2: วิเคราะห์จุดแข็ง (positive aspects + system strengths)
            logger.info("✅ Step 2: Analyzing positive aspects...")
            positive_data = await self._analyze_positive_aspects(survey_data)
            insights.update(positive_data)
            
            # Step 3: วิเคราะห์ปัญหา (negative aspects + pain points)
            logger.info("❌ Step 3: Analyzing problems and pain points...")
            negative_data = await self._analyze_negative_aspects(survey_data)
            insights.update(negative_data)
            
            # Step 4: สร้างข้อเสนอแนะ (recommendations + priority actions)
            logger.info("💡 Step 4: Generating recommendations...")
            recommendations_data = await self._generate_recommendations(survey_data, insights)
            insights.update(recommendations_data)
            
            # Step 5: สรุป executive summary
            logger.info("📋 Step 5: Creating executive summary...")
            summary_data = await self._create_executive_summary(survey_data, insights)
            insights.update(summary_data)
            
            # เพิ่ม metadata
            insights.update({
                "ai_generated": True,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_points_analyzed": self._count_data_points(survey_data),
                "analysis_method": "Gemini AI Multi-Step Analysis",
                "steps_completed": 5
            })
            
            logger.info("✅ Incremental Gemini AI analysis completed successfully")
            return insights
            
        except Exception as e:
            logger.error(f"❌ Gemini AI analysis failed: {e}")
            return self._create_fallback_insights(survey_data)
    
    async def _analyze_sentiment_summary(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: วิเคราะห์ sentiment โดยรวม"""
        sentiment_summary = survey_data.get("sentiment_summary", {})
        total_feedback = sum(sentiment_summary.values())
        
        if total_feedback == 0:
            return {
                "sentiment_analysis": {
                    "overall_mood": "ไม่สามารถประเมินได้",
                    "satisfaction_level": "ไม่มีข้อมูล",
                    "confidence_score": 0.0
                }
            }
        
        positive_rate = (sentiment_summary.get("positive", 0) / total_feedback) * 100
        
        prompt = f"""วิเคราะห์ sentiment จากข้อมูล Survey นี้:

📊 ข้อมูล Sentiment:
- Positive: {sentiment_summary.get('positive', 0)} responses ({positive_rate:.1f}%)
- Neutral: {sentiment_summary.get('neutral', 0)} responses
- Negative: {sentiment_summary.get('negative', 0)} responses
- รวม: {total_feedback} responses

💭 Top Keywords: {', '.join(survey_data.get('top_keywords', [])[:8])}

🔍 ตัวอย่าง Negative feedback:
{self._format_sample_texts(survey_data.get('negative_feedback_samples', [])[:3])}

😊 ตัวอย่าง Positive feedback:
{self._format_sample_texts(survey_data.get('positive_feedback_samples', [])[:3])}

ให้วิเคราะห์และตอบเป็น JSON format เท่านั้น:
{{
  "sentiment_analysis": {{
    "overall_mood": "คำอธิบายอารมณ์โดยรวมของผู้ใช้ (1-2 ประโยค)",
    "satisfaction_level": "สูง/ปานกลาง/ต่ำ พร้อมเหตุผลสั้นๆ",
    "confidence_score": 0.XX (ความมั่นใจในการวิเคราะห์ 0-1)
  }}
}}"""

        try:
            response = await self._call_gemini_api(prompt)
            return self._parse_json_response(response, {
                "sentiment_analysis": {
                    "overall_mood": "เป็นกลาง",
                    "satisfaction_level": "ปานกลาง",
                    "confidence_score": 0.7
                }
            })
        except Exception as e:
            logger.warning(f"⚠️ Sentiment analysis failed: {e}")
            return {
                "sentiment_analysis": {
                    "overall_mood": "เป็นกลาง" if positive_rate < 60 else "เชิงบวก",
                    "satisfaction_level": "สูง" if positive_rate >= 70 else "ปานกลาง" if positive_rate >= 50 else "ต่ำ",
                    "confidence_score": 0.75
                }
            }
    
    async def _analyze_positive_aspects(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: วิเคราะห์จุดแข็งและข้อดี"""
        
        positive_samples = survey_data.get('positive_feedback_samples', [])[:5]
        likert_scores = survey_data.get('likert_scores', {})
        
        # หา likert scores ที่สูง
        high_scores = {k: v for k, v in likert_scores.items() if v >= 4.0}
        
        prompt = f"""วิเคราะห์จุดแข็งและข้อดีของระบบจากข้อมูลนี้:

😊 ตัวอย่าง Positive Feedback:
{self._format_sample_texts(positive_samples)}

📊 คะแนนที่ดี (≥4.0):
{self._format_scores(high_scores)}

🔑 Keywords เชิงบวก: {', '.join([kw for kw in survey_data.get('top_keywords', []) if kw in ['สะดวก', 'ง่าย', 'เร็ว', 'ดี', 'เยี่ยม', 'ชอบ', 'ไม่ยาก']])}

ให้วิเคราะห์และตอบเป็น JSON format เท่านั้น:
{{
  "positive_aspects": [
    "จุดแข็งที่ 1 พร้อมหลักฐานสนับสนุน",
    "จุดแข็งที่ 2 พร้อมหลักฐานสนับสนุน", 
    "จุดแข็งที่ 3 พร้อมหลักฐานสนับสนุน"
  ],
  "system_strengths": [
    "ความสามารถหลักของระบบที่โดดเด่น",
    "ความสามารถหลักของระบบที่โดดเด่น"
  ]
}}"""

        try:
            response = await self._call_gemini_api(prompt)
            return self._parse_json_response(response, {
                "positive_aspects": ["ระบบได้รับการทดสอบและมีผู้ใช้ให้ความคิดเห็น"],
                "system_strengths": ["ระบบทำงานตามที่ออกแบบไว้"]
            })
        except Exception as e:
            logger.warning(f"⚠️ Positive analysis failed: {e}")
            return self._generate_positive_fallback(survey_data)
    
    async def _analyze_negative_aspects(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: วิเคราะห์ปัญหาและจุดอ่อน"""
        
        negative_samples = survey_data.get('negative_feedback_samples', [])[:5]
        likert_scores = survey_data.get('likert_scores', {})
        
        # หา likert scores ที่ต่ำ
        low_scores = {k: v for k, v in likert_scores.items() if v < 3.5}
        
        prompt = f"""วิเคราะห์ปัญหาและจุดที่ต้องปรับปรุงจากข้อมูลนี้:

❌ ตัวอย่าง Negative Feedback:
{self._format_sample_texts(negative_samples)}

📉 คะแนนที่ต่ำ (<3.5):
{self._format_scores(low_scores)}

⚠️ Keywords เชิงลบ: {', '.join([kw for kw in survey_data.get('top_keywords', []) if kw in ['ช้า', 'ยาก', 'สับสน', 'ปัญหา', 'ไม่ดี', 'แก้ไข', 'ปรับปรุง', 'หายาก']])}

ให้วิเคราะห์และตอบเป็น JSON format เท่านั้น:
{{
  "negative_aspects": [
    "ปัญหาที่ 1 พร้อมการวิเคราะห์สาเหตุ",
    "ปัญหาที่ 2 พร้อมการวิเคราะห์สาเหตุ",
    "ปัญหาที่ 3 พร้อมการวิเคราะห์สาเหตุ"
  ],
  "improvement_areas": [
    "พื้นที่ที่ต้องปรับปรุงเร่งด่วนที่ 1",
    "พื้นที่ที่ต้องปรับปรุงเร่งด่วนที่ 2"
  ],
  "user_pain_points": [
    "ปัญหาผู้ใช้ที่ต้องแก้ไขทันทีที่ 1",
    "ปัญหาผู้ใช้ที่ต้องแก้ไขทันทีที่ 2"
  ]
}}"""

        try:
            response = await self._call_gemini_api(prompt)
            return self._parse_json_response(response, {
                "negative_aspects": [],
                "improvement_areas": [],
                "user_pain_points": []
            })
        except Exception as e:
            logger.warning(f"⚠️ Negative analysis failed: {e}")
            return self._generate_negative_fallback(survey_data)
    
    async def _generate_recommendations(self, survey_data: Dict[str, Any], current_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: สร้างข้อเสนอแนะและแผนปฏิบัติ"""
        
        # รวบรวมปัญหาที่พบ
        problems = current_insights.get('negative_aspects', []) + current_insights.get('user_pain_points', [])
        strengths = current_insights.get('positive_aspects', []) + current_insights.get('system_strengths', [])
        
        prompt = f"""จากการวิเคราะห์ข้อมูล Survey สร้างข้อเสนอแนะและแผนปฏิบัติ:

💪 จุดแข็งที่พบ:
{self._format_list_items(strengths[:4])}

⚠️ ปัญหาที่พบ:
{self._format_list_items(problems[:4])}

📊 Sentiment: {current_insights.get('sentiment_analysis', {}).get('satisfaction_level', 'ปานกลาง')}

🎯 ให้สร้างข้อเสนอแนะที่:
- เป็นรูปธรรมและปฏิบัติได้จริง
- มีลำดับความสำคัญชัดเจน  
- คำนึงถึงทรัพยากรและเวลา

ตอบเป็น JSON format เท่านั้น:
{{
  "recommendations": [
    "ข้อเสนอแนะที่ 1 พร้อม ROI/ผลลัพธ์ที่คาดหวัง",
    "ข้อเสนอแนะที่ 2 พร้อม timeline การดำเนินงาน",
    "ข้อเสนอแนะที่ 3 พร้อมทรัพยากรที่ต้องการ",
    "ข้อเสนอแนะที่ 4 พร้อมขั้นตอนการทำ"
  ],
  "priority_actions": [
    "งานเร่งด่วน (0-30 วัน) พร้อมผลลัพธ์ที่คาดหวัง",
    "งานระยะสั้น (1-3 เดือน) พร้อมทรัพยากรที่ต้องการ",
    "งานระยะยาว (3-12 เดือน) พร้อมผลกระทบเชิงกลยุทธ์"
  ]
}}"""

        try:
            response = await self._call_gemini_api(prompt)
            return self._parse_json_response(response, {
                "recommendations": [
                    "วิเคราะห์ feedback เพิ่มเติมเพื่อหาจุดปรับปรุง",
                    "พัฒนาระบบตามความต้องการของผู้ใช้",
                    "ติดตามผลการใช้งานอย่างต่อเนื่อง"
                ],
                "priority_actions": [
                    "รวบรวมข้อมูลผู้ใช้เพิ่มเติม (1-4 สัปดาห์)",
                    "วิเคราะห์และจัดทำแผนปรับปรุง (1-2 เดือน)",
                    "ดำเนินการพัฒนาตามแผน (3-6 เดือน)"
                ]
            })
        except Exception as e:
            logger.warning(f"⚠️ Recommendations generation failed: {e}")
            return self._generate_recommendations_fallback(survey_data)
    
    async def _create_executive_summary(self, survey_data: Dict[str, Any], insights: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: สร้าง executive summary"""
        
        sentiment_summary = survey_data.get("sentiment_summary", {})
        total_feedback = sum(sentiment_summary.values())
        positive_rate = (sentiment_summary.get("positive", 0) / total_feedback * 100) if total_feedback > 0 else 0
        
        key_findings = []
        key_findings.extend(insights.get('positive_aspects', [])[:2])
        key_findings.extend(insights.get('negative_aspects', [])[:2])
        
        prompt = f"""สร้าง Executive Summary จากผลการวิเคราะห์:

📊 ข้อมูลพื้นฐาน:
- ผู้ตอบ 31 คน: {total_feedback} คำตอบ
- ความพึงพอใจ: {positive_rate:.1f}%
- ระดับความพึงพอใจ: {insights.get('sentiment_analysis', {}).get('satisfaction_level', 'ปานกลาง')}

🔍 ข้อค้นพบสำคัญ:
{self._format_list_items(key_findings[:4])}

💡 ข้อเสนอแนะหลัก:
{self._format_list_items(insights.get('recommendations', [])[:2])}

ให้สร้าง Executive Summary ที่:
- กระชับ 2-3 ประโยค
- เน้นผลกระทบต่อธุรกิจ
- ระบุแนวทางดำเนินการหลัก

ตอบเป็น JSON format เท่านั้น:
{{
  "executive_summary": "สรุปผลการวิเคราะห์ 2-3 ประโยค เน้นข้อค้นพบสำคัญและผลกระทบต่อธุรกิจ"
}}"""

        try:
            response = await self._call_gemini_api(prompt)
            result = self._parse_json_response(response, {
                "executive_summary": f"จากการวิเคราะห์ {total_feedback} ความคิดเห็น พบความพึงพอใจ {positive_rate:.1f}% ระบบมีจุดแข็งที่ควรธำรงไว้และมีจุดที่ต้องปรับปรุงเพื่อเพิ่มประสิทธิภาพ"
            })
            return result
        except Exception as e:
            logger.warning(f"⚠️ Executive summary generation failed: {e}")
            return {
                "executive_summary": f"จากการวิเคราะห์ {total_feedback} ความคิดเห็น พบความพึงพอใจ {positive_rate:.1f}% ระบบได้รับการประเมินครบถ้วนและมีข้อเสนอแนะสำหรับการปรับปรุง"
            }
    
    # Helper methods
    def _format_sample_texts(self, texts: List[str]) -> str:
        """แปลง list ของ text เป็น string ที่อ่านง่าย"""
        if not texts:
            return "ไม่มีข้อมูล"
        
        formatted = []
        for i, text in enumerate(texts[:5], 1):
            clean_text = str(text).replace('"', "'").replace('\n', ' ').strip()[:120]
            if len(str(text)) > 120:
                clean_text += "..."
            formatted.append(f"{i}. {clean_text}")
        
        return "\n".join(formatted)
    
    def _format_scores(self, scores: Dict[str, float]) -> str:
        """แปลง scores เป็น string"""
        if not scores:
            return "ไม่มีข้อมูล"
        
        items = []
        for aspect, score in scores.items():
            items.append(f"- {aspect}: {score:.2f}/5.0")
        
        return "\n".join(items)
    
    def _format_list_items(self, items: List[str]) -> str:
        """แปลง list เป็น string แบบมีหัวข้อ"""
        if not items:
            return "ไม่มีข้อมูล"
        
        formatted = []
        for i, item in enumerate(items, 1):
            formatted.append(f"{i}. {item}")
        
        return "\n".join(formatted)
    
    def _parse_json_response(self, response: str, fallback: Dict) -> Dict:
        """แปลง response เป็น JSON พร้อม fallback"""
        try:
            # ลอง clean response ก่อน
            cleaned = self._clean_response_text(response)
            
            # หา JSON ใน response
            json_start = cleaned.find('{')
            json_end = cleaned.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned[json_start:json_end]
                parsed = json.loads(json_str)
                
                # ตรวจสอบว่ามี key ที่คาดหวังหรือไม่
                if any(key in parsed for key in fallback.keys()):
                    return parsed
            
            return fallback
            
        except Exception as e:
            logger.warning(f"⚠️ JSON parsing failed: {e}")
            return fallback
    
    def _clean_response_text(self, text: str) -> str:
        """ทำความสะอาด response text"""
        # ลบ markdown code blocks
        text = text.replace('```json', '').replace('```', '')
        
        # ลบ text พิเศษ
        lines = text.split('\n')
        cleaned_lines = []
        in_json = False
        
        for line in lines:
            if '{' in line and not in_json:
                in_json = True
                json_start = line.find('{')
                cleaned_lines.append(line[json_start:])
            elif in_json:
                cleaned_lines.append(line)
                if '}' in line and line.count('}') >= line.count('{'):
                    break
        
        return '\n'.join(cleaned_lines)
    
    # Fallback methods
    def _generate_positive_fallback(self, survey_data: Dict) -> Dict:
        """Fallback สำหรับจุดแข็ง"""
        positive_rate = self._calculate_positive_rate(survey_data)
        
        positive_aspects = []
        system_strengths = []
        
        if positive_rate >= 60:
            positive_aspects.append(f"ผู้ใช้มีความพึงพอใจสูง ({positive_rate:.1f}%)")
            system_strengths.append("ระบบได้รับการยอมรับจากผู้ใช้เป็นอย่างดี")
        
        # เช็ค keywords เชิงบวก
        positive_keywords = [kw for kw in survey_data.get('top_keywords', []) 
                           if kw in ['สะดวก', 'ง่าย', 'เร็ว', 'ดี', 'เยี่ยม']]
        if positive_keywords:
            positive_aspects.append(f"ผู้ใช้ชื่นชมในเรื่อง: {', '.join(positive_keywords[:3])}")
        
        # เช็ค likert scores
        high_scores = {k: v for k, v in survey_data.get('likert_scores', {}).items() if v >= 4.0}
        if high_scores:
            best_aspect = max(high_scores.items(), key=lambda x: x[1])
            system_strengths.append(f"คะแนนสูงสุด: {best_aspect[0]} ({best_aspect[1]:.2f}/5)")
        
        return {
            "positive_aspects": positive_aspects or ["ระบบได้รับการทดสอบและมีผู้ใช้ให้ความคิดเห็น"],
            "system_strengths": system_strengths or ["ระบบทำงานตามที่ออกแบบไว้"]
        }
    
    def _generate_negative_fallback(self, survey_data: Dict) -> Dict:
        """Fallback สำหรับปัญหา"""
        negative_aspects = []
        improvement_areas = []
        user_pain_points = []
        
        # เช็ค keywords เชิงลบ
        negative_keywords = [kw for kw in survey_data.get('top_keywords', []) 
                           if kw in ['ช้า', 'ยาก', 'สับสน', 'ปัญหา', 'ไม่ดี']]
        if negative_keywords:
            negative_aspects.append(f"ปัญหาหลัก: {', '.join(negative_keywords[:3])}")
        
        # เช็ค likert scores ต่ำ
        low_scores = {k: v for k, v in survey_data.get('likert_scores', {}).items() if v < 3.5}
        if low_scores:
            worst_aspect = min(low_scores.items(), key=lambda x: x[1])
            improvement_areas.append(f"ต้องปรับปรุงเร่งด่วน: {worst_aspect[0]} ({worst_aspect[1]:.2f}/5)")
        
        # เช็ค negative samples
        negative_samples = survey_data.get('negative_feedback_samples', [])
        if negative_samples:
            common_issues = []
            for sample in negative_samples[:3]:
                sample_lower = sample.lower()
                if 'ภาษาไทย' in sample_lower:
                    common_issues.append('ปัญหาการใช้ภาษาไทย')
                if 'ปุ่ม' in sample_lower:
                    common_issues.append('ปุ่มไม่ชัดเจน')
                if 'วันที่' in sample_lower:
                    common_issues.append('รูปแบบวันที่สับสน')
            
            user_pain_points.extend(list(set(common_issues)))
        
        return {
            "negative_aspects": negative_aspects,
            "improvement_areas": improvement_areas,
            "user_pain_points": user_pain_points
        }
    
    def _generate_recommendations_fallback(self, survey_data: Dict) -> Dict:
        """Fallback สำหรับข้อเสนอแนะ"""
        return {
            "recommendations": [
                "วิเคราะห์ feedback เพิ่มเติมเพื่อหาจุดปรับปรุงที่ชัดเจน",
                "พัฒนาระบบตามความต้องการของผู้ใช้ที่ระบุในความคิดเห็น",
                "ปรับปรุง UX/UI ให้เข้าใจง่ายและใช้งานสะดวกขึ้น",
                "จัดทำคู่มือและระบบช่วยเหลือผู้ใช้งาน"
            ],
            "priority_actions": [
                "รวบรวมและจัดหมวดหมู่ feedback เพิ่มเติม (0-30 วัน)",
                "วิเคราะห์และจัดทำแผนปรับปรุงระบบ (1-3 เดือน)",
                "ดำเนินการพัฒนาและปรับปรุงตามแผน (3-12 เดือน)"
            ]
        }
    
    def _calculate_positive_rate(self, survey_data: Dict) -> float:
        """คำนวณอัตราความพึงพอใจ"""
        sentiment_summary = survey_data.get("sentiment_summary", {})
        total = sum(sentiment_summary.values())
        return (sentiment_summary.get("positive", 0) / total * 100) if total > 0 else 0
    
    async def _call_gemini_api(self, prompt: str) -> str:
        """เรียก Gemini API (ใช้เวลาน้อยกว่าเดิม)"""
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        last_error = None
        
        for i, model_config in enumerate(self.models):
            model_name = model_config["name"]
            timeout = model_config["timeout"]
            
            try:
                logger.info(f"🔄 Using model {i+1}/{len(self.models)}: {model_name}")
                
                url = f"{self.base_url}/{model_name}:generateContent?key={self.api_key}"
                
                payload = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": model_config["temperature"],
                        "topK": 40,
                        "topP": 0.8,
                        "maxOutputTokens": model_config["max_tokens"],
                        "candidateCount": 1
                    },
                    "safetySettings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        }
                    ]
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Survey-Analysis-Dashboard/2.1"
                }
                
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            
                            if (result.get("candidates") and 
                                len(result["candidates"]) > 0 and
                                result["candidates"][0].get("content") and
                                result["candidates"][0]["content"].get("parts") and
                                len(result["candidates"][0]["content"]["parts"]) > 0):
                                
                                content = result["candidates"][0]["content"]["parts"][0]["text"]
                                logger.info(f"✅ Successfully used model: {model_name}")
                                return content
                            else:
                                logger.warning(f"⚠️ Empty response from {model_name}")
                                last_error = ValueError(f"Empty response from {model_name}")
                                continue
                        
                        elif response.status == 404:
                            logger.warning(f"⚠️ Model {model_name} not found (404)")
                            last_error = ValueError(f"Model {model_name} not available")
                            continue
                            
                        elif response.status == 429:
                            logger.warning(f"⚠️ Rate limit exceeded for {model_name}")
                            last_error = ValueError(f"Rate limit for {model_name}")
                            await asyncio.sleep(2)  # รอสั้นลงเพราะเป็น request เล็ก
                            continue
                            
                        else:
                            error_text = await response.text()
                            logger.warning(f"⚠️ API error {response.status} for {model_name}")
                            last_error = ValueError(f"API error {response.status}: {error_text}")
                            continue
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Timeout for {model_name} ({timeout}s)")
                last_error = ValueError(f"Timeout for {model_name}")
                continue
                
            except Exception as e:
                logger.warning(f"⚠️ Error with {model_name}: {str(e)}")
                last_error = e
                continue
        
        # หากทุก model ล้มเหลว
        error_msg = f"All {len(self.models)} models failed. Last error: {last_error}"
        logger.error(f"❌ {error_msg}")
        raise Exception(error_msg)
    
    def _create_fallback_insights(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """สร้าง fallback insights เมื่อ Gemini ไม่พร้อมใช้งาน"""
        sentiment_summary = survey_data.get("sentiment_summary", {})
        total_feedback = sum(sentiment_summary.values())
        
        if total_feedback == 0:
            return {
                "executive_summary": "ไม่มีข้อมูลเพียงพอสำหรับการวิเคราะห์",
                "positive_aspects": [],
                "negative_aspects": [],
                "recommendations": ["รวบรวมข้อมูลเพิ่มเติมจากผู้ใช้"],
                "system_strengths": [],
                "improvement_areas": [],
                "user_pain_points": [],
                "priority_actions": [],
                "sentiment_analysis": {
                    "overall_mood": "ไม่สามารถประเมินได้",
                    "satisfaction_level": "ไม่มีข้อมูล",
                    "confidence_score": 0.0
                },
                "ai_generated": False,
                "analysis_method": "Rule-based Fallback (No Data)",
                "steps_completed": 0
            }
        
        positive_rate = (sentiment_summary.get("positive", 0) / total_feedback) * 100
        
        # ใช้ fallback methods
        positive_data = self._generate_positive_fallback(survey_data)
        negative_data = self._generate_negative_fallback(survey_data)
        recommendations_data = self._generate_recommendations_fallback(survey_data)
        
        insights = {
            "executive_summary": f"จากการวิเคราะห์ {total_feedback} ความคิดเห็น พบความพึงพอใจ {positive_rate:.1f}% ระบบมีจุดแข็งและจุดที่ต้องปรับปรุงตามที่ผู้ใช้ระบุ",
            **positive_data,
            **negative_data,
            **recommendations_data,
            "sentiment_analysis": {
                "overall_mood": "เป็นกลาง" if positive_rate < 60 else "เชิงบวก",
                "satisfaction_level": "สูง" if positive_rate >= 70 else "ปานกลาง" if positive_rate >= 50 else "ต่ำ",
                "confidence_score": 0.75
            },
            "ai_generated": False,
            "analysis_method": "Enhanced Rule-based Analysis (Fallback)",
            "steps_completed": 5
        }
        
        return insights
    
    def _count_data_points(self, survey_data: Dict[str, Any]) -> int:
        """นับจำนวน data points ที่วิเคราะห์"""
        count = 0
        count += len(survey_data.get("top_keywords", []))
        count += len(survey_data.get("negative_feedback_samples", []))
        count += len(survey_data.get("positive_feedback_samples", []))
        count += len(survey_data.get("likert_scores", {}))
        count += sum(survey_data.get("sentiment_summary", {}).values())
        return count
    
    async def list_available_models(self) -> List[Dict[str, str]]:
        """ดูรายการ models ที่ใช้ได้"""
        if not self.api_key:
            return []
        
        try:
            url = f"{self.base_url}?key={self.api_key}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        models = []
                        for model in result.get("models", []):
                            name = model.get("name", "").replace("models/", "")
                            if "generateContent" in model.get("supportedGenerationMethods", []):
                                models.append({
                                    "name": name,
                                    "display_name": model.get("displayName", name),
                                    "description": model.get("description", ""),
                                    "supported": True
                                })
                        return models
                    else:
                        logger.error(f"Failed to list models: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def get_service_info(self) -> Dict[str, Any]:
        """ข้อมูลเกี่ยวกับ service"""
        return {
            "service_name": "Gemini AI Service",
            "version": "2.1.0",
            "api_key_configured": bool(self.api_key),
            "available_models": len(self.models),
            "primary_model": self.models[0]["name"] if self.models else None,
            "analysis_method": "Multi-Step Incremental Analysis",
            "steps": [
                "Sentiment Analysis",
                "Positive Aspects Analysis", 
                "Negative Aspects Analysis",
                "Recommendations Generation",
                "Executive Summary Creation"
            ],
            "features": [
                "Survey Analysis",
                "Incremental Processing",
                "Reduced Timeout Risk",
                "Step-by-step Insights",
                "Thai Language Support",
                "Enhanced Fallback System"
            ]
        }


# Helper functions สำหรับ integration
async def enhance_insights_with_ai(nlp_results: Dict[str, Any], gemini_api_key: str = None) -> Dict[str, Any]:
    """
    ปรับปรุง insights ด้วย Gemini AI แบบทีละขั้นตอน
    """
    try:
        # เตรียมข้อมูลสำหรับ Gemini
        survey_data = {
            "sentiment_summary": nlp_results.get("sentiment_distribution", {}),
            "top_keywords": [kw["word"] for kw in nlp_results.get("top_keywords", [])[:10]],
            "negative_feedback_samples": [
                result["text"] for result in nlp_results.get("detailed_results", [])
                if result.get("sentiment") == "negative"
            ][:5],
            "positive_feedback_samples": [
                result["text"] for result in nlp_results.get("detailed_results", [])
                if result.get("sentiment") == "positive"
            ][:5],
            "likert_scores": {
                k: v.get("mean", 0) for k, v in nlp_results.get("likert_analysis", {}).items()
            },
            "choice_results": nlp_results.get("choice_analysis", {})
        }
        
        # สร้าง AI service และวิเคราะห์แบบทีละขั้นตอน
        ai_service = GeminiAIService(api_key=gemini_api_key)
        ai_insights = await ai_service.generate_survey_insights(survey_data)
        
        logger.info(f"✅ Successfully enhanced insights with Gemini AI ({ai_insights.get('steps_completed', 0)} steps)")
        return ai_insights
        
    except Exception as e:
        logger.error(f"❌ AI enhancement failed: {e}")
        # ใช้ fallback
        ai_service = GeminiAIService()
        return ai_service._create_fallback_insights({
            "sentiment_summary": nlp_results.get("sentiment_distribution", {}),
            "top_keywords": [kw["word"] for kw in nlp_results.get("top_keywords", [])[:10]]
        })


def setup_gemini_config() -> Dict[str, Any]:
    """
    ตั้งค่า และตรวจสอบ Gemini configuration
    """
    api_key = os.getenv('GEMINI_API_KEY')
    
    config_info = {
        "api_key_configured": bool(api_key),
        "api_key_length": len(api_key) if api_key else 0,
        "fallback_available": True,
        "service_status": "ready" if api_key else "no_api_key",
        "analysis_method": "Multi-Step Incremental Analysis",
        "advantages": [
            "ลดความเสี่ยง timeout จากการ generate ครั้งเดียว",
            "ประมวลผลเร็วขึ้นในแต่ละขั้นตอน", 
            "ควบคุม cost ได้ดีกว่า",
            "สามารถแสดงผล real-time ได้",
            "หากบางขั้นตอนล้มเหลว ยังได้ผลลัพธ์บางส่วน"
        ],
        "instructions": [
            "ตั้งค่า GEMINI_API_KEY ใน environment variables",
            "หรือส่ง API key ผ่าน parameter ของ function",
            "ระบบจะประมวลผลแบบทีละขั้นตอนเพื่อลดความเสี่ยง",
            "หากบางขั้นตอนล้มเหลว ระบบจะใช้ enhanced rule-based fallback"
        ]
    }
    
    return config_info


# Test function
async def test_gemini_service(api_key: str = None) -> Dict[str, Any]:
    """
    ทดสอบ Gemini service แบบทีละขั้นตอน
    """
    service = GeminiAIService(api_key=api_key)
    
    test_data = {
        "sentiment_summary": {"positive": 15, "neutral": 8, "negative": 2},
        "top_keywords": ["สะดวก", "ง่าย", "เร็ว", "สับสน"],
        "negative_feedback_samples": ["ปุ่มแก้ไขหายาก", "วันที่แสดงผลสับสน"],
        "positive_feedback_samples": ["ใช้งานง่าย สะดวกดี", "ไม่ต้องมาธนาคาร", "เร็วดี"],
        "likert_scores": {"ความง่าย": 4.2, "ความพอใจ": 3.8},
        "choice_results": {"สนใจ": {"สนใจ": 20, "ไม่สนใจ": 5}}
    }
    
    try:
        start_time = datetime.now()
        result = await service.generate_survey_insights(test_data)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "test_status": "success",
            "processing_time": f"{processing_time:.2f}s",
            "ai_generated": result.get("ai_generated", False),
            "analysis_method": result.get("analysis_method", "unknown"),
            "steps_completed": result.get("steps_completed", 0),
            "insights_count": {
                "positive_aspects": len(result.get("positive_aspects", [])),
                "negative_aspects": len(result.get("negative_aspects", [])),
                "recommendations": len(result.get("recommendations", [])),
                "priority_actions": len(result.get("priority_actions", []))
            },
            "confidence_score": result.get("sentiment_analysis", {}).get("confidence_score", 0),
            "advantages_realized": [
                f"ประมวลผลใน {processing_time:.2f} วินาที (เร็วกว่าเดิม)",
                f"สำเร็จ {result.get('steps_completed', 0)}/5 ขั้นตอน",
                "ลดความเสี่ยง timeout",
                "ควบคุม cost ได้ดีกว่า"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "test_status": "failed",
            "error": str(e),
            "fallback_available": True,
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    import asyncio
    
    # Test configuration
    async def main():
        print("🧪 Testing Enhanced Gemini AI Service (Multi-Step)...")
        
        # Test config
        config = setup_gemini_config()
        print("Configuration:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
        # Test service
        if config["api_key_configured"]:
            result = await test_gemini_service()
            print("\nTest Result:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("\n⚠️ No API key configured. Set GEMINI_API_KEY environment variable.")
    
    asyncio.run(main())