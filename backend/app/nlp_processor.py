import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Any, Tuple
import warnings
import re
import os
import json
import aiohttp
from collections import Counter
from datetime import datetime

# สำหรับ pythainlp (ถ้ามี)
try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.tag import pos_tag
    from pythainlp.classify import GzipModel
    PYTHAINLP_AVAILABLE = True
    print("✅ PyThaiNLP imported successfully")
except ImportError as e:
    PYTHAINLP_AVAILABLE = False
    print(f"⚠️ PyThaiNLP not available: {e}")
    print("Will use basic tokenization instead")

warnings.filterwarnings('ignore')

class NLPProcessor:
    def __init__(self, analysis_method: int = 0):
        """
        Initialize NLP Processor
        
        Args:
            analysis_method (int): 
                0 = Rule-based analysis (default)
                1 = GzipModel trained analysis
                2 = SSense API analysis (AI for Thai)
        """
        self.models_path = "./app/data/models"
        self.training_data_path = "./app/data/training/"
        self._ready = False
        
        # Analysis method selection
        self.analysis_method = analysis_method
        self.available_methods = {
            0: {"name": "Rule-based", "function": "_analyze_sentiment_rule_based"},
            1: {"name": "GzipModel", "function": "_analyze_sentiment_gzip_model"},
            2: {"name": "SSense API", "function": "_analyze_sentiment_ssense"},
        }
        
        # Trained model จาก GzipModel (สำหรับ method = 1)
        self.trained_model = None
        self.model_trained = False
        
        # SSense API configuration (สำหรับ method = 2)
        self.ssense_api_key = "5Rx7TLNeMdtYyhub2A74VJ3HVLThqWRk"
        self.ssense_url = "https://api.aiforthai.in.th/ssense"
        
        # คำบ่งชี้ความรู้สึกเฉพาะสำหรับ survey นี้ (สำหรับ fallback)
        self.positive_indicators = {
            'ดี', 'เยี่ยม', 'สุดยอด', 'ชอบ', 'เทพ', 'เจ๋ง', 'สะดวก', 'ง่าย', 
            'รวดเร็ว', 'ประทับใจ', 'พอใจ', 'ชัดเจน', 'เข้าใจง่าย', 'ใช้งานง่าย',
            'สบาย', 'ครบถ้วน', 'ถูกใจ', 'น่าใช้', 'สำเร็จ', 'ขอบคุณ', 'perfect',
            'good', 'nice', 'great', 'excellent', 'แนะนำ', 'ชื่นชม', 'ยอดเยี่ยม',
            'คุ้นเคยกับข้อมูล', 'ไม่ต้องมาธนาคาร', 'อิเล็กทรอนิก', 'ถนัด', 'ไม่ยาก',
            'ไม่ซับซ้อน', 'ขั้นตอนน้อย', 'เร็ว', 'ทันใจ', 'มีประโยชน์', 'โอเค', 'ok',
            'ได้', 'ปกติ', 'เหมาะสม', 'พอดี', 'ตรงตาม', 'ไม่สับสน', 'สนใจ'
        }
        
        self.negative_indicators = {
            'แย่', 'ไม่ดี', 'ช้า', 'ยาก', 'ซับซ้อน', 'สับสน', 'ไม่เข้าใจ', 'ปัญหา',
            'ผิดพลาด', 'ผิดหวัง', 'บั๊ก', 'error', 'ไม่สะดวก', 'ยุ่งยาก', 'ลำบาก',
            'หงุดหงิด', 'รำคาญ', 'ไม่ชอบ', 'ไม่พอใจ', 'ขัดข้อง', 'ไม่แนะนำ',
            'bad', 'terrible', 'horrible', 'ไม่สื่อ', 'ไม่ตรง', 'ไม่เหมาะ',
            'ปรับปรุง', 'แก้ไข', 'ต้องการเพิ่ม', 'ควรปรับ', 'ไม่มี', 'หาย',
            'ไม่สื่อความหมาย', 'ไม่ชัดเจน', 'ไม่เห็น', 'ไม่ครบ', 'ขาด', 'น้อย',
            'ไม่เพียงพอ', 'ไม่ตรงกัน', 'ผิด', 'ใช้ไม่ได้', 'ไม่สนใจ'
        }
        
        # คำที่เป็นกลาง
        self.neutral_indicators = {
            'องค์กร', 'มีอยู่แล้ว', 'ปกติ', 'ธรรมดา', 'เฉยๆ', 'ไม่แน่ใจ', 
            'ไม่รู้', 'อาจจะ', 'บางครั้ง', 'พอใช้ได้', 'ก็ได้', 'record',
            'ทดสอบ', 'ระบบ', 'ข้อมูล', 'แสดงผล', 'ตรวจสอบ', 'ไม่ใส่',
            'ไม่มีอะไร', 'เหมือนเดิม', 'ทั่วไป'
        }
        
        # คำสำคัญเฉพาะของ survey
        self.survey_keywords = {
            'เข้าสู่ระบบ': ['เข้าสู่ระบบ', 'ลงทะเบียน', 'login', 'otp', 'email', 'tha id'],
            'การใช้งาน': ['ใช้งาน', 'ใช้บริการ', 'ฟังก์ชัน', 'ฟีเจอร์', 'การทำงาน'],
            'ข้อมูล': ['ข้อมูล', 'ตราสารหนี้', 'ดอกเบี้ย', 'บัญชี', 'ประวัติ', 'รายละเอียด'],
            'วันที่': ['วันที่', 'เดือน', 'ปี', 'พศ', 'คศ', 'format', 'รูปแบบ'],
            'ปุ่ม': ['ปุ่ม', 'button', 'edit', 'แก้ไข', 'กด', 'คลิก'],
            'ภาษา': ['ภาษาไทย', 'ภาษา', 'คำ', 'ข้อความ', 'ศัพท์'],
            'ธนาคาร': ['ธนาคาร', 'ธปท', 'สาขา', 'มาธนาคาร'],
            'หน้าจอ': ['หน้าจอ', 'แสดงผล', 'เมนู', 'หน้า', 'screen'],
            'เวลา': ['เวลา', 'ระยะเวลา', 'นาน', 'เร็ว', 'ช้า', 'รอ']
        }
        
    async def initialize(self):
        """เตรียมระบบ NLP"""
        try:
            print("🚀 กำลังเตรียมระบบ Survey NLP Analysis...")
            print(f"🎯 Analysis Method: {self.analysis_method} ({self.available_methods[self.analysis_method]['name']})")
            
            os.makedirs(self.models_path, exist_ok=True)
            os.makedirs(self.training_data_path, exist_ok=True)
            
            # ทดสอบ PyThaiNLP ถ้ามี
            if PYTHAINLP_AVAILABLE:
                test_text = "วันนี้อากาศดีจัง"
                try:
                    tokens = word_tokenize(test_text)
                    print(f"✅ PyThaiNLP พร้อมใช้งาน")
                    
                    # เตรียมโมเดลตาม method ที่เลือก
                    await self._initialize_method()
                    
                except Exception as e:
                    print(f"⚠️ PyThaiNLP error: {e}")
                    if self.analysis_method == 1:
                        print("⚠️ GzipModel requires PyThaiNLP, switching to Rule-based")
                        self.analysis_method = 0
            else:
                print("⚠️ PyThaiNLP ไม่พร้อมใช้งาน")
                if self.analysis_method == 1:
                    print("⚠️ GzipModel requires PyThaiNLP, switching to Rule-based")
                    self.analysis_method = 0
            
            # ทดสอบ SSense API ถ้าเลือก method 2
            if self.analysis_method == 2:
                await self._test_ssense_api()
            
            self._ready = True
            method_name = self.available_methods[self.analysis_method]['name']
            print(f"✅ Survey NLP Processor พร้อมใช้งาน ({method_name})")
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")
            self._ready = True  # ใช้งานได้แม้มีปัญหา
    
    async def _initialize_method(self):
        """เตรียมโมเดลตาม analysis method ที่เลือก"""
        if self.analysis_method == 0:
            # Rule-based ไม่ต้องเตรียมอะไร
            print("📋 Rule-based analysis ready")
            
        elif self.analysis_method == 1:
            # GzipModel ต้องโหลดและเทรนโมเดล
            await self._load_and_train_gzip_model()
            
        elif self.analysis_method == 2:
            # SSense API ทดสอบการเชื่อมต่อ
            print("🌐 Testing SSense API connection...")
    
    async def _test_ssense_api(self):
        """ทดสอบการเชื่อมต่อ SSense API"""
        try:
            test_text = "ทดสอบระบบ"
            headers = {
                'Apikey': self.ssense_api_key
            }
            params = {'text': test_text}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.ssense_url, headers=headers, params=params, timeout=10) as response:
                    if response.status == 200:
                        result = await response.json()
                        print("✅ SSense API พร้อมใช้งาน")
                        return True
                    else:
                        print(f"⚠️ SSense API error: {response.status}")
                        print("🔄 จะใช้ Rule-based แทน")
                        self.analysis_method = 0
                        return False
        except Exception as e:
            print(f"⚠️ SSense API connection failed: {e}")
            print("🔄 จะใช้ Rule-based แทน")
            self.analysis_method = 0
            return False
    
    def set_analysis_method(self, method: int) -> bool:
        """เปลี่ยน analysis method"""
        if method in self.available_methods:
            self.analysis_method = method
            print(f"🔄 เปลี่ยนเป็น {self.available_methods[method]['name']}")
            return True
        else:
            print(f"❌ Invalid method {method}. Available: {list(self.available_methods.keys())}")
            return False
    
    def get_available_methods(self) -> Dict[int, Dict[str, Any]]:
        """ดูรายการ analysis methods ที่ใช้ได้"""
        return self.available_methods.copy()
    
    async def _load_and_train_gzip_model(self):
        """โหลดและเทรนโมเดล GzipModel (สำหรับ method = 1)"""
        try:
            print("🤖 กำลังโหลดข้อมูลสำหรับเทรน GzipModel...")
            print(f"📂 ค้นหาข้อมูลเทรนใน: {self.training_data_path}")
            
            # ตรวจสอบว่าโฟลเดอร์มีอยู่หรือไม่
            if not os.path.exists(self.training_data_path):
                print(f"❌ ไม่พบโฟลเดอร์: {self.training_data_path}")
                # ลองหาใน paths ทางเลือก
                alternative_paths = [
                    "/app/data/training",
                    "./data/training", 
                    "data/training",
                    "/data/training"
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        print(f"✅ พบโฟลเดอร์ใน: {alt_path}")
                        self.training_data_path = alt_path
                        break
                else:
                    print("❌ ไม่พบโฟลเดอร์ data/training ใน paths ใดๆ")
                    print("📁 ไฟล์และโฟลเดอร์ปัจจุบัน:")
                    try:
                        current_files = os.listdir(".")
                        for f in current_files:
                            print(f"  - {f}")
                            if os.path.isdir(f) and f == "data":
                                data_files = os.listdir(f)
                                for df in data_files:
                                    print(f"    - {f}/{df}")
                                    if os.path.isdir(os.path.join(f, df)) and df == "training":
                                        training_files = os.listdir(os.path.join(f, df))
                                        for tf in training_files:
                                            file_path = os.path.join(f, df, tf)
                                            if os.path.isfile(file_path):
                                                try:
                                                    with open(file_path, 'r', encoding='utf-8') as temp_f:
                                                        lines = temp_f.readlines()
                                                        print(f"      - {tf}: {len(lines)} บรรทัด")
                                                        # แสดง 3 บรรทัดแรก
                                                        for i, line in enumerate(lines[:3]):
                                                            print(f"        {i+1}: {line.strip()}")
                                                except Exception as e:
                                                    print(f"      - {tf}: ไม่สามารถอ่านได้ ({e})")
                    except Exception as e:
                        print(f"❌ ไม่สามารถแสดงรายการไฟล์: {e}")
            
            training_data = []
            files_info = {}
            
            # โหลดข้อมูล positive
            pos_file = os.path.join(self.training_data_path, "pos.txt")
            print(f"🔍 ตรวจสอบไฟล์: {pos_file}")
            if os.path.exists(pos_file):
                try:
                    with open(pos_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        files_info['pos_total_lines'] = len(lines)
                        valid_lines = 0
                        for line_num, line in enumerate(lines, 1):
                            line = line.strip()
                            if line:
                                training_data.append((line, "positive"))
                                valid_lines += 1
                            else:
                                print(f"    บรรทัด {line_num}: ว่าง")
                        print(f"📝 โหลดข้อมูล positive: {valid_lines}/{len(lines)} ประโยค (ข้าม {len(lines)-valid_lines} บรรทัดว่าง)")
                        
                        # แสดงตัวอย่างข้อมูล
                        print("    ตัวอย่างข้อมูล positive:")
                        pos_data = [x[0] for x in training_data if x[1] == "positive"]
                        for i, example in enumerate(pos_data[:3]):
                            print(f"      {i+1}: {example}")
                            
                except Exception as e:
                    print(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์ pos.txt: {e}")
            else:
                print(f"❌ ไม่พบไฟล์: {pos_file}")
            
            # โหลดข้อมูล negative
            neg_file = os.path.join(self.training_data_path, "neg.txt")
            print(f"🔍 ตรวจสอบไฟล์: {neg_file}")
            if os.path.exists(neg_file):
                try:
                    with open(neg_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        files_info['neg_total_lines'] = len(lines)
                        valid_lines = 0
                        for line_num, line in enumerate(lines, 1):
                            line = line.strip()
                            if line:
                                training_data.append((line, "negative"))
                                valid_lines += 1
                        print(f"📝 โหลดข้อมูล negative: {valid_lines}/{len(lines)} ประโยค")
                        
                        # แสดงตัวอย่างข้อมูล
                        print("    ตัวอย่างข้อมูล negative:")
                        neg_data = [x[0] for x in training_data if x[1] == "negative"]
                        for i, example in enumerate(neg_data[:3]):
                            print(f"      {i+1}: {example}")
                            
                except Exception as e:
                    print(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์ neg.txt: {e}")
            else:
                print(f"❌ ไม่พบไฟล์: {neg_file}")
            
            # โหลดข้อมูล neutral
            neu_file = os.path.join(self.training_data_path, "neu.txt")
            print(f"🔍 ตรวจสอบไฟล์: {neu_file}")
            if os.path.exists(neu_file):
                try:
                    with open(neu_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        files_info['neu_total_lines'] = len(lines)
                        valid_lines = 0
                        for line_num, line in enumerate(lines, 1):
                            line = line.strip()
                            if line:
                                training_data.append((line, "neutral"))
                                valid_lines += 1
                        print(f"📝 โหลดข้อมูล neutral: {valid_lines}/{len(lines)} ประโยค")
                        
                        # แสดงตัวอย่างข้อมูล
                        print("    ตัวอย่างข้อมูล neutral:")
                        neu_data = [x[0] for x in training_data if x[1] == "neutral"]
                        for i, example in enumerate(neu_data[:3]):
                            print(f"      {i+1}: {example}")
                            
                except Exception as e:
                    print(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์ neu.txt: {e}")
            else:
                print(f"❌ ไม่พบไฟล์: {neu_file}")
            
            print(f"📊 รวมข้อมูลเทรนทั้งหมด: {len(training_data)} ประโยค")
            print(f"📊 สรุปไฟล์: {files_info}")
            
            # ถ้าไม่มีข้อมูลเลย หรือมีน้อยมาก
            if len(training_data) == 0:
                print("❌ ไม่พบข้อมูลเทรนเลย กำลังสร้างข้อมูลตัวอย่าง...")
                training_data = await self._create_sample_training_data()
            elif len(training_data) < 50 and len(training_data) > 0:
                print(f"⚠️ ข้อมูลเทรนมีเพียง {len(training_data)} ประโยค จะเพิ่มข้อมูลตัวอย่าง...")
                # เก็บข้อมูลจริงไว้ แล้วเพิ่มข้อมูลตัวอย่าง
                real_data = training_data.copy()
                sample_data = await self._create_sample_training_data()
                training_data.extend(sample_data)
                print(f"📊 ข้อมูลหลังเพิ่มตัวอย่าง: {len(training_data)} ประโยค (จริง: {len(real_data)}, ตัวอย่าง: {len(sample_data)})")
            
            if len(training_data) > 0:
                print(f"🎯 เริ่มเทรน GzipModel ด้วยข้อมูล {len(training_data)} ประโยค...")
                
                # สร้างและเทรน GzipModel
                self.trained_model = GzipModel(training_data)
                self.model_trained = True
                
                # ทดสอบโมเดล
                test_result = self.trained_model.predict("ระบบใช้งานง่ายมาก", k=1)
                print(f"🧪 ทดสอบโมเดล: 'ระบบใช้งานง่ายมาก' -> {test_result}")
                
                print(f"✅ เทรน GzipModel สำเร็จ!")
                
            else:
                print("❌ ไม่พบข้อมูลเทรน จะสลับไปใช้ rule-based")
                self.analysis_method = 0
                
        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาดในการเทรน GzipModel: {e}")
            import traceback
            traceback.print_exc()
            print("Will fallback to rule-based analysis")
            self.analysis_method = 0
    
    def _tokenize_thai(self, text: str) -> List[str]:
        """แบ่งคำภาษาไทย"""
        if PYTHAINLP_AVAILABLE:
            try:
                return word_tokenize(text, keep_whitespace=False)
            except:
                pass
        
        # Basic tokenization fallback
        tokens = re.findall(r'[ก-ฮ]+|[a-zA-Z]+|\d+', text)
        return tokens
    
    def _preprocess_text(self, text: str) -> str:
        """ทำความสะอาดข้อความ"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            text = text.strip()
            text = re.sub(r'\r\n|\r|\n', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\u0e00-\u0e7fa-zA-Z0-9\s\-\.]', ' ', text)
            return text.strip()
        except Exception as e:
            print(f"Warning: preprocessing error: {e}")
            return text
    
    def _analyze_sentiment_rule_based(self, text: str) -> Dict[str, Any]:
        """วิเคราะห์ sentiment ด้วย rule-based approach (Method 0)"""
        try:
            text_lower = text.lower()
            tokens = self._tokenize_thai(text_lower)
            
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            positive_words = []
            negative_words = []
            
            # ตรวจสอบคำเฉพาะก่อน
            if "ไม่สับสน" in text_lower:
                positive_count += 2
                positive_words.append("ไม่สับสน")
            elif "สับสน" in text_lower and "ไม่" not in text_lower:
                negative_count += 1
                negative_words.append("สับสน")
            
            for token in tokens:
                if token in self.positive_indicators:
                    positive_count += 1
                    positive_words.append(token)
                elif token in self.negative_indicators:
                    negative_count += 1
                    negative_words.append(token)
                elif token in self.neutral_indicators:
                    neutral_count += 1
            
            total_sentiment = positive_count + negative_count + neutral_count
            
            if total_sentiment == 0:
                if any(word in text_lower for word in ['ต้องการ', 'ควร', 'ปรับปรุง', 'เพิ่ม', 'แก้ไข']):
                    sentiment = "negative"
                elif any(word in text_lower for word in ['ไม่มี', 'ไม่ใส่', 'ไม่ตอบ', '-']):
                    sentiment = "neutral"
                else:
                    sentiment = "neutral"
            else:
                if positive_count > negative_count:
                    sentiment = "positive"
                elif negative_count > positive_count:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "positive_indicators": positive_count,
                "negative_indicators": negative_count,
                "neutral_indicators": neutral_count,
                "positive_words": positive_words,
                "negative_words": negative_words,
                "method": "rule_based",
                "method_id": 0
            }
            
        except Exception as e:
            print(f"Error in rule-based sentiment analysis: {e}")
            return {
                "sentiment": "neutral",
                "positive_indicators": 0,
                "negative_indicators": 0,
                "neutral_indicators": 0,
                "positive_words": [],
                "negative_words": [],
                "method": "rule_based_error",
                "method_id": 0
            }
    
    def _analyze_sentiment_gzip_model(self, text: str) -> Dict[str, Any]:
        """วิเคราะห์ sentiment ด้วย GzipModel (Method 1)"""
        try:
            if not self.trained_model or not self.model_trained:
                # Fallback to rule-based if model not available
                print("⚠️ GzipModel not available, falling back to rule-based")
                return self._analyze_sentiment_rule_based(text)
            
            # ใช้โมเดลที่เทรนแล้ว
            predicted_sentiment = self.trained_model.predict(text, k=1)
            
            # คำนวณ rule-based เป็นตัวช่วยตรวจสอบ
            rule_result = self._analyze_sentiment_rule_based(text)
            
            return {
                "sentiment": predicted_sentiment,
                "positive_indicators": rule_result.get("positive_indicators", 0),
                "negative_indicators": rule_result.get("negative_indicators", 0),
                "neutral_indicators": rule_result.get("neutral_indicators", 0),
                "positive_words": rule_result.get("positive_words", []),
                "negative_words": rule_result.get("negative_words", []),
                "method": "gzip_model",
                "method_id": 1,
                "model_prediction": predicted_sentiment,
                "rule_prediction": rule_result["sentiment"],
                "agreement": predicted_sentiment == rule_result["sentiment"]
            }
            
        except Exception as e:
            print(f"Error in GzipModel analysis: {e}")
            # Fallback to rule-based
            return self._analyze_sentiment_rule_based(text)
    
    async def _analyze_sentiment_ssense(self, text: str) -> Dict[str, Any]:
        """วิเคราะห์ sentiment ด้วย SSense API (Method 2)"""
        try:
            headers = {
                'Apikey': self.ssense_api_key
            }
            params = {'text': text}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.ssense_url, headers=headers, params=params, timeout=10) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # แปลงผลลัพธ์จาก SSense
                        sentiment_data = result.get('sentiment', {})
                        polarity = sentiment_data.get('polarity', '')
                        
                        if polarity == 'positive':
                            sentiment = "positive"
                        elif polarity == 'negative':
                            sentiment = "negative"
                        else:
                            sentiment = "neutral"
                        
                        # ดึงข้อมูลเพิ่มเติมจาก preprocess
                        preprocess_data = result.get('preprocess', {})
                        positive_words = preprocess_data.get('pos', [])
                        negative_words = preprocess_data.get('neg', [])
                        keywords = preprocess_data.get('keyword', [])
                        
                        # ดึงข้อมูลจาก intention
                        intention_data = result.get('intention', {})
                        
                        return {
                            "sentiment": sentiment,
                            "positive_indicators": len(positive_words),
                            "negative_indicators": len(negative_words),
                            "neutral_indicators": 0,
                            "positive_words": positive_words,
                            "negative_words": negative_words,
                            "keywords": keywords,
                            "method": "ssense_api",
                            "method_id": 2,
                            "ssense_polarity": polarity,
                            "ssense_intention": intention_data,
                            "ssense_raw": result
                        }
                        
                    else:
                        print(f"⚠️ SSense API error: {response.status}")
                        # Fallback to rule-based
                        result = self._analyze_sentiment_rule_based(text)
                        result["method"] = "ssense_fallback_rule"
                        result["method_id"] = 2
                        return result
                        
        except Exception as e:
            print(f"Error in SSense API analysis: {e}")
            # Fallback to rule-based
            result = self._analyze_sentiment_rule_based(text)
            result["method"] = "ssense_fallback_error"
            result["method_id"] = 2
            return result
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """วิเคราะห์ sentiment ด้วยวิธีที่เลือกไว้"""
        # เลือก method ตามที่กำหนด
        method_info = self.available_methods[self.analysis_method]
        method_function = getattr(self, method_info["function"])
        
        if self.analysis_method == 2:
            # SSense API ต้องใช้ async - แต่ถ้า event loop ทำงานอยู่แล้วให้ fallback
            try:
                import asyncio
                
                # ตรวจสอบว่ามี event loop ทำงานอยู่หรือไม่
                try:
                    loop = asyncio.get_running_loop()
                    # ถ้ามี loop ทำงานอยู่ ให้ใช้ create_task หรือ fallback
                    print("⚠️ Event loop is running, using synchronous fallback for SSense")
                    # Fallback to rule-based
                    return self._analyze_sentiment_rule_based(text)
                except RuntimeError:
                    # ไม่มี loop ทำงานอยู่ สามารถสร้างใหม่ได้
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(method_function(text))
                    finally:
                        loop.close()
                        
            except Exception as e:
                print(f"Error in SSense API: {e}, falling back to rule-based")
                return self._analyze_sentiment_rule_based(text)
        else:
            return method_function(text)
    
    def _extract_keywords_advanced(self, text: str) -> List[Dict]:
        """สกัดคำสำคัญขั้นสูง"""
        try:
            tokens = self._tokenize_thai(text)
            if not tokens:
                return []
            
            # ใช้ POS tagging ถ้ามี PyThaiNLP
            if PYTHAINLP_AVAILABLE:
                try:
                    pos_tags = pos_tag(tokens)
                except:
                    pos_tags = [(token, "UNKNOWN") for token in tokens]
            else:
                pos_tags = [(token, "UNKNOWN") for token in tokens]
            
            keywords = []
            
            for word, pos in pos_tags:
                word_clean = word.strip().lower()
                
                if (len(word_clean) > 1 and 
                    not word_clean.isdigit() and
                    word_clean not in {'-', '/', '(', ')', '[', ']', 'และ', 'ที่', 'การ', 'ใน', 'ของ', 'เป็น', 'มี', 'ได้', 'จะ', 'ไว้', 'นี้', 'นั้น'}):
                    
                    base_score = 0.3
                    
                    if pos in ['ADJ', 'VERB']:
                        base_score += 0.3
                    elif pos in ['NOUN', 'ADV']:
                        base_score += 0.2
                    
                    sentiment_type = "neutral"
                    category = "general"
                    
                    if word_clean in self.positive_indicators:
                        base_score += 0.3
                        sentiment_type = "positive"
                    elif word_clean in self.negative_indicators:
                        base_score += 0.3
                        sentiment_type = "negative"
                    elif word_clean in self.neutral_indicators:
                        base_score += 0.1
                        sentiment_type = "neutral"
                    
                    for cat, keywords_list in self.survey_keywords.items():
                        if any(kw in word_clean for kw in keywords_list):
                            category = cat
                            base_score += 0.2
                            break
                    
                    keywords.append({
                        "word": word_clean,
                        "score": min(base_score, 1.0),
                        "frequency": 1,
                        "pos_tag": pos,
                        "sentiment_type": sentiment_type,
                        "category": category
                    })
            
            keywords.sort(key=lambda x: x["score"], reverse=True)
            return keywords[:8]
            
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            words = text.split()
            return [{"word": word.lower(), "score": 0.5, "frequency": 1, 
                    "pos_tag": "UNKNOWN", "sentiment_type": "neutral", "category": "general"} 
                   for word in words[:3] if len(word) > 1]
    
    async def analyze_single_text(self, text: str, column_name: str = "") -> Dict[str, Any]:
        """วิเคราะห์ข้อความเดี่ยว"""
        if not text or not text.strip():
            return {
                "sentiment": "neutral",
                "keywords": [],
                "text": text,
                "column": column_name
            }
        
        try:
            processed_text = self._preprocess_text(text)
            
            if not processed_text:
                return {
                    "sentiment": "neutral",
                    "keywords": [],
                    "text": text,
                    "column": column_name
                }
            
            # ใช้ method ที่เลือกไว้ - แยกการจัดการ async/sync
            if self.analysis_method == 2:
                # SSense API method - เรียกโดยตรง
                sentiment_result = await self._analyze_sentiment_ssense(processed_text)
            else:
                # Rule-based หรือ GzipModel
                sentiment_result = self.analyze_sentiment(processed_text)
            
            keywords = self._extract_keywords_advanced(processed_text)
            
            return {
                "sentiment": sentiment_result["sentiment"],
                "keywords": keywords,
                "text": text,
                "column": column_name,
                "debug": {
                    "processed_text": processed_text,
                    "positive_indicators": sentiment_result.get("positive_indicators", 0),
                    "negative_indicators": sentiment_result.get("negative_indicators", 0),
                    "positive_words": sentiment_result.get("positive_words", []),
                    "negative_words": sentiment_result.get("negative_words", []),
                    "method": sentiment_result.get("method", "unknown"),
                    "method_id": sentiment_result.get("method_id", self.analysis_method),
                    "analysis_method": self.analysis_method,
                    "ssense_raw": sentiment_result.get("ssense_raw") if self.analysis_method == 2 else None
                }
            }
            
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return {
                "sentiment": "neutral",
                "keywords": [],
                "text": text,
                "column": column_name
            }
    
    def _identify_survey_text_columns(self, df: pd.DataFrame) -> Dict[int, str]:
        """ระบุคอลัมน์ข้อความในแบบสอบถาม Post-Test Survey อย่างแม่นยำ"""
        
        column_mapping = {}
        
        for i, col_name in enumerate(df.columns):
            col_str = str(col_name).lower()
            
            # ข้ามคอลัมน์ที่เป็น demographic หรือ metadata
            if any(skip_word in col_str for skip_word in ['timestamp', 'เวลา', 'วันที่สร้าง', 'id', 'ลำดับ', 'ชื่อ', 'อายุ', 'เพศ', 'email']):
                continue
            
            # ข้ามคอลัมน์ที่เป็นคำถามแบบเลือก Yes/No หรือ สนใจ/ไม่สนใจ
            sample_data = df.iloc[:, i].dropna().astype(str).head(20).tolist()
            unique_values = set([str(val).strip() for val in sample_data])
            
            # ตรวจสอบว่าเป็นคำถาม สนใจ/ไม่สนใจ
            if (len(unique_values) <= 3 and 
                any(val in ['สนใจ', 'ไม่สนใจ', 'ยินยอม', 'ไม่ยินยอม'] for val in unique_values)):
                print(f"🚫 ข้าม column {i+1} ({col_name}): เป็นคำถามแบบเลือก")
                continue
                
            if not sample_data:
                continue
                
            # ตรวจสอบว่าเป็นคอลัมน์ที่มีข้อความความคิดเห็นหรือไม่
            text_count = 0
            
            for text in sample_data:
                text_clean = text.strip()
                
                # ข้ามข้อมูลที่ไม่ใช่ความคิดเห็น
                if (len(text_clean) <= 2 or 
                    text_clean in ['-', 'ไม่มี', 'ไม่ใส่', 'ไม่ตอบ', 'ยินยอม', 'ไม่ยินยอม', 'สนใจ', 'ไม่สนใจ'] or
                    text_clean.isdigit() or
                    text_clean in ['1', '2', '3', '4', '5'] or
                    text_clean in ['1.0', '2.0', '3.0', '4.0', '5.0']):
                    continue
                    
                # ตรวจสอบว่าเป็นคำตอบแบบเลือก (choice) หรือไม่
                if any(choice in text_clean for choice in [
                    'ตรวจสอบข้อมูลตราสารหนี้', 'ขอเอกสาร', 'ตรวจสอบการรับเงิน',
                    'Email', 'OTP', 'ThaiD', 'สร้างรหัสผ่าน'
                ]):
                    continue
                    
                # ถ้ามีอักขระภาษาไทยและความยาวเหมาะสม แสดงว่าเป็นความคิดเห็น
                if (any('\u0e00' <= char <= '\u0e7f' for char in text_clean) and 
                    len(text_clean) > 3):
                    text_count += 1
                    
            # ถ้ามีข้อความความคิดเห็นมากกว่า 30% ถือว่าเป็นคอลัมน์ข้อความ
            if text_count >= max(1, len(sample_data) * 0.3):
                # ใส่ชื่อคอลัมน์ตามหัวตาราง หรือสร้างชื่อใหม่
                if 'เหตุผล' in col_str and 'เข้าสู่ระบบ' in col_str:
                    column_mapping[i] = "เหตุผลการเข้าสู่ระบบ"
                elif 'ส่วนที่ใช้งานได้ดี' in col_str or 'ดีที่สุด' in col_str:
                    column_mapping[i] = "ส่วนที่ใช้งานได้ดีที่สุด"
                elif 'ข้อมูลเพิ่มเติม' in col_str or 'เพิ่มเติม' in col_str:
                    column_mapping[i] = "ข้อมูลที่ต้องการเพิ่มเติม"
                elif 'สับสน' in col_str and 'วันที่' in col_str and 'เหตุผล' in col_str:
                    column_mapping[i] = "เหตุผลที่สับสนเรื่องวันที่"
                elif 'ปรับปรุง' in col_str or 'แก้ไข' in col_str:
                    column_mapping[i] = "สิ่งที่ต้องการให้ปรับปรุง"
                elif 'แนะนำ' in col_str or 'เพิ่มเติม' in col_str:
                    column_mapping[i] = "คำแนะนำเพิ่มเติม"
                else:
                    # ตรวจสอบอีกครั้งว่าเป็นคอลัมน์ความคิดเห็นจริงๆ
                    print(f"🤔 ตรวจสอบ column {i+1} ({col_name}): {text_count}/{len(sample_data)} texts")
                    if text_count >= 3:  # ต้องมีความคิดเห็นอย่างน้อย 3 รายการ
                        column_mapping[i] = f"ความคิดเห็น_{str(col_name)}"
        
        print(f"🔍 พบคอลัมน์ข้อความ: {len(column_mapping)} คอลัมน์")
        for col_idx, col_name in column_mapping.items():
            print(f"  - คอลัมน์ {col_idx + 1}: {col_name}")
            
        return column_mapping
    
    def _analyze_likert_scales(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """วิเคราะห์คอลัมน์ Likert Scale อย่างอัตโนมัติ"""
        likert_analysis = {}
        
        for i, col_name in enumerate(df.columns):
            col_str = str(col_name).lower()
            
            # ข้ามคอลัมน์ที่ไม่ใช่ likert scale
            if any(skip_word in col_str for skip_word in ['timestamp', 'เวลา', 'วันที่', 'id', 'ลำดับ', 'email', 'ข้อมูล', 'ชื่อ']):
                continue
                
            try:
                col_data = df.iloc[:, i]
                numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                
                # ตรวจสอบว่าเป็น likert scale (ค่า 1-5) หรือไม่
                unique_values = set(numeric_data)
                is_likert = (len(unique_values) <= 5 and 
                           all(val in range(1, 6) for val in unique_values) and
                           len(numeric_data) > 0)
                
                if is_likert:
                    valid_scores = numeric_data[(numeric_data >= 1) & (numeric_data <= 5)]
                    
                    if len(valid_scores) > 0:
                        # สร้างชื่อที่เข้าใจได้
                        if 'ลงทะเบียน' in col_str or 'register' in col_str:
                            description = "ความง่ายในการลงทะเบียน"
                        elif 'โดยรวม' in col_str or 'overall' in col_str:
                            description = "ความง่ายโดยรวม"
                        elif 'เวลา' in col_str or 'time' in col_str:
                            description = "ความเหมาะสมของเวลา"
                        elif 'มั่นใจ' in col_str or 'confidence' in col_str:
                            description = "ความมั่นใจ"
                        elif 'พอใจ' in col_str or 'satisfaction' in col_str:
                            description = "ความพอใจ"
                        else:
                            description = f"คะแนน_{str(col_name)}"
                            
                        likert_analysis[description] = {
                            "mean": float(valid_scores.mean()),
                            "std": float(valid_scores.std()) if len(valid_scores) > 1 else 0.0,
                            "distribution": valid_scores.value_counts().sort_index().to_dict(),
                            "count": len(valid_scores),
                            "original_column": i,
                            "column_name": str(col_name)
                        }
                        
            except Exception as e:
                print(f"Could not analyze column {i} ({col_name}): {e}")
                continue
        
        return likert_analysis
    
    def _analyze_choice_questions(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """วิเคราะห์คำถามแบบเลือกตอบ"""
        choice_analysis = {}
        
        try:
            for i, col_name in enumerate(df.columns):
                col_str = str(col_name).lower()
                
                # ข้ามคอลัมน์ที่ไม่เกี่ยวข้อง
                if any(skip_word in col_str for skip_word in ['timestamp', 'เวลา', 'วันที่', 'id', 'ลำดับ', 'เหตุผล']):
                    continue
                
                # ดูข้อมูลในคอลัมน์
                col_data = df.iloc[:, i].dropna()
                if len(col_data) == 0:
                    continue
                
                # ตรวจสอบว่าเป็นคำถามแบบเลือกหรือไม่
                value_counts = col_data.value_counts()
                unique_values = set(col_data.astype(str))
                
                # ถ้าเป็นคำตอบแบบเลือก (มีค่าซ้ำๆ และไม่ใช่ likert scale)
                is_choice = (len(unique_values) <= 10 and 
                           len(value_counts) > 1 and
                           not all(str(val).isdigit() and 1 <= int(val) <= 5 for val in unique_values if str(val).isdigit()))
                
                if is_choice:
                    if 'ดีที่สุด' in col_str or 'best' in col_str:
                        choice_analysis["ส่วนที่ดีที่สุด"] = value_counts.to_dict()
                    elif 'วิธีเข้าสู่ระบบ' in col_str or 'login' in col_str:
                        choice_analysis["วิธีเข้าสู่ระบบ"] = value_counts.to_dict()
                    elif 'สับสน' in col_str and 'วันที่' in col_str and 'เหตุผล' not in col_str:
                        choice_analysis["ความสับสนเรื่องวันที่"] = value_counts.to_dict()
                    elif ('ทดสอบ' in col_str and 'อีกครั้ง' in col_str) or ('สนใจ' in col_str):
                        choice_analysis["ความสนใจทดสอบอีกครั้ง"] = value_counts.to_dict()
                    else:
                        # ตรวจสอบค่าที่เป็น choice questions จริงๆ
                        choice_keywords = ['ตรวจสอบข้อมูล', 'ขอเอกสาร', 'Email', 'OTP', 'ThaiD']
                        if any(any(keyword in str(val) for keyword in choice_keywords) for val in unique_values):
                            choice_analysis[f"เลือกตอบ_{str(col_name)}"] = value_counts.to_dict()
                        
        except Exception as e:
            print(f"Error analyzing choice questions: {e}")
        
        return choice_analysis
    
    async def generate_ai_insights(self, sentiment_dist: Dict, column_analysis: Dict, 
                                 keywords: List, detailed_results: List, 
                                 likert_analysis: Dict, choice_analysis: Dict) -> Dict:
        """สร้าง AI Insights ด้วย Gemini API"""
        
        # เตรียมข้อมูลสำหรับส่งให้ AI
        summary_data = {
            "sentiment_summary": sentiment_dist,
            "top_keywords": [kw["word"] for kw in keywords[:10]],
            "negative_feedback_samples": [
                result["text"] for result in detailed_results 
                if result["sentiment"] == "negative"
            ][:5],
            "positive_feedback_samples": [
                result["text"] for result in detailed_results 
                if result["sentiment"] == "positive"
            ][:5],
            "likert_scores": {k: v.get("mean", 0) for k, v in likert_analysis.items()},
            "choice_results": choice_analysis
        }
        
        # ใช้ Gemini API หรือ fallback เป็น rule-based
        try:
            # TODO: เพิ่ม Gemini API call ที่นี่
            ai_insights = await self._call_gemini_api(summary_data)
            return ai_insights
        except Exception as e:
            print(f"AI API failed, using fallback: {e}")
            return self._generate_survey_insights_fallback(
                sentiment_dist, column_analysis, keywords, detailed_results, 
                likert_analysis, choice_analysis
            )
    
    async def _call_gemini_api(self, data: Dict) -> Dict:
        """เรียก Gemini API สำหรับสร้าง insights"""
        # TODO: implement Gemini API call
        # สำหรับตอนนี้ใช้ fallback
        raise Exception("Gemini API not implemented yet")
    
    def _generate_survey_insights_fallback(self, sentiment_dist: Dict, column_analysis: Dict, 
                                 keywords: List, detailed_results: List, 
                                 likert_analysis: Dict, choice_analysis: Dict) -> Dict:
        """สร้าง insights เฉพาะสำหรับ Post-Test Survey (Fallback)"""
        insights = {
            "positive_aspects": [],
            "negative_aspects": [],
            "recommendations": [],
            "system_strengths": [],
            "improvement_areas": [],
            "user_pain_points": []
        }
        
        total_analyzed = sum(sentiment_dist.values())
        if total_analyzed == 0:
            return insights
        
        positive_percent = (sentiment_dist["positive"] / total_analyzed) * 100
        negative_percent = (sentiment_dist["negative"] / total_analyzed) * 100
        
        # วิเคราะห์จาก sentiment
        if positive_percent > 60:
            insights["positive_aspects"].append(f"ผู้ใช้มีความพึงพอใจสูง ({positive_percent:.1f}%)")
        elif positive_percent < 40:
            insights["negative_aspects"].append(f"ผู้ใช้มีความพึงพอใจต่ำ ({positive_percent:.1f}%)")
            
        # วิเคราะห์จาก keywords
        positive_keywords = [kw["word"] for kw in keywords[:10] 
                           if kw.get("sentiment_type") == "positive"]
        negative_keywords = [kw["word"] for kw in keywords[:10] 
                           if kw.get("sentiment_type") == "negative"]
        
        if positive_keywords:
            insights["system_strengths"].append(f"จุดแข็ง: {', '.join(positive_keywords[:5])}")
            
        if negative_keywords:
            insights["user_pain_points"].append(f"ปัญหาหลัก: {', '.join(negative_keywords[:5])}")
        
        # วิเคราะห์จาก likert analysis
        if likert_analysis:
            high_scores = {k: v for k, v in likert_analysis.items() if v.get("mean", 0) >= 4.0}
            low_scores = {k: v for k, v in likert_analysis.items() if v.get("mean", 0) < 3.5}
            
            if high_scores:
                best_aspect = max(high_scores.items(), key=lambda x: x[1].get("mean", 0))
                insights["positive_aspects"].append(f"คะแนนสูงสุด: {best_aspect[0]} ({best_aspect[1].get('mean', 0):.2f}/5)")
            
            if low_scores:
                worst_aspect = min(low_scores.items(), key=lambda x: x[1].get("mean", 0))
                insights["improvement_areas"].append(f"ต้องปรับปรุงเร่งด่วน: {worst_aspect[0]} ({worst_aspect[1].get('mean', 0):.2f}/5)")
        
        # วิเคราะห์จาก choice questions - เพิ่มข้อมูลสนใจ/ไม่สนใจ
        if choice_analysis:
            for question, answers in choice_analysis.items():
                if "สนใจ" in question:
                    total_respondents = sum(answers.values())
                    interested = answers.get("สนใจ", 0)
                    if total_respondents > 0:
                        interest_rate = (interested / total_respondents) * 100
                        if interest_rate >= 70:
                            insights["positive_aspects"].append(f"ความสนใจในการทดสอบอีกครั้งสูง ({interest_rate:.1f}%)")
                        elif interest_rate < 50:
                            insights["negative_aspects"].append(f"ความสนใจในการทดสอบอีกครั้งต่ำ ({interest_rate:.1f}%)")
                elif "สับสน" in question:
                    confused_count = sum(count for answer, count in answers.items() if "สับสน" in str(answer))
                    total_count = sum(answers.values())
                    if confused_count > total_count * 0.3:
                        insights["negative_aspects"].append(f"ผู้ใช้สับสนเรื่องวันที่ ({confused_count}/{total_count})")
        
        # วิเคราะห์จาก detailed results
        common_issues = []
        positive_themes = []
        
        for result in detailed_results:
            text_lower = result["text"].lower()
            
            # ปัญหาที่พบบ่อย
            if "ภาษาไทย" in text_lower and ("ไม่สื่อ" in text_lower or "ไม่เข้าใจ" in text_lower):
                common_issues.append("ปัญหาการใช้ภาษาไทย")
            if "ปุ่ม" in text_lower and ("edit" in text_lower or "แก้ไข" in text_lower or "หา" in text_lower):
                common_issues.append("ปุ่มแก้ไขหายาก")
            if "วันที่" in text_lower and "สับสน" in text_lower:
                common_issues.append("รูปแบบวันที่สับสน")
            
            # จุดเด่นที่พบบ่อย
            if result["sentiment"] == "positive":
                if "สะดวก" in text_lower or "ง่าย" in text_lower:
                    positive_themes.append("ความสะดวกและง่าย")
                if "ไม่ต้องมาธนาคาร" in text_lower:
                    positive_themes.append("ไม่ต้องมาธนาคาร")
                if "เร็ว" in text_lower or "รวดเร็ว" in text_lower:
                    positive_themes.append("ความรวดเร็ว")
        
        # รวบรวม insights
        unique_issues = list(set(common_issues))
        unique_themes = list(set(positive_themes))
        
        if unique_issues:
            insights["negative_aspects"].extend(unique_issues[:3])
        if unique_themes:
            insights["system_strengths"].extend(unique_themes[:3])
            
        # สร้างข้อเสนอแนะ
        recommendations = []
        
        if "ปัญหาการใช้ภาษาไทย" in unique_issues:
            recommendations.append("ปรับปรุงการใช้ภาษาไทยให้ชัดเจนและเข้าใจง่ายขึ้น")
        if "ปุ่มแก้ไขหายาก" in unique_issues:
            recommendations.append("ปรับปรุงการแสดงปุ่มแก้ไขให้เห็นได้ชัดเจน")
        if "รูปแบบวันที่สับสน" in unique_issues:
            recommendations.append("เปลี่ยนรูปแบบวันที่เป็น วัน-เดือน-ปี (พ.ศ.)")
        
        # เพิ่มข้อเสนอแนะทั่วไป
        if negative_percent > positive_percent:
            recommendations.append("ปรับปรุง UX/UI โดยรวมเพื่อเพิ่มความพึงพอใจ")
        if likert_analysis and any(v.get("mean", 0) < 3.5 for v in likert_analysis.values()):
            recommendations.append("ฝึกอบรมผู้ใช้งานเพื่อเพิ่มความมั่นใจ")
        
        recommendations.extend([
            "เพิ่มคำอธิบายและคู่มือการใช้งาน",
            "พัฒนาระบบ help และ FAQ"
        ])
        
        insights["recommendations"] = recommendations[:5]
        
        return insights
    
    async def analyze_survey(self, df: pd.DataFrame, analysis_id: str) -> Dict[str, Any]:
        """วิเคราะห์แบบสอบถาม Post-Test Survey แบบครบถ้วน"""
        print(f"🎯 เริ่มวิเคราะห์ Post-Test Survey ID: {analysis_id}")
        method_name = self.available_methods[self.analysis_method]['name']
        print(f"📊 ใช้ {method_name} (Method {self.analysis_method})")
        print(f"📏 ข้อมูล: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
        
        results = {
            "total_responses": len(df),
            "texts_analyzed": 0,
            "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "top_keywords": [],
            "detailed_results": [],
            "column_analysis": {},
            "model_info": {
                "engine": method_name,
                "preprocessing": "Thai tokenization + Advanced sentiment rules",
                "features": "Survey-specific keyword analysis + POS enhancement",
                "version": "Post-Test Survey Optimized v3.0 + Multiple Methods",
                "analysis_method": self.analysis_method,
                "method_name": method_name,
                "available_methods": self.get_available_methods()
            },
            "insights": {}
        }
        
        # ระบุคอลัมน์ข้อความ
        text_columns = self._identify_survey_text_columns(df)
        print(f"📝 พบคอลัมน์ข้อความ: {len(text_columns)} คอลัมน์")
        
        # วิเคราะห์ Likert scales
        likert_analysis = self._analyze_likert_scales(df)
        results["likert_analysis"] = likert_analysis
        print(f"📏 พบ Likert scales: {len(likert_analysis)} คอลัมน์")
        
        # วิเคราะห์ Choice questions
        choice_analysis = self._analyze_choice_questions(df)
        results["choice_analysis"] = choice_analysis
        print(f"☑️ พบ Choice questions: {len(choice_analysis)} คำถาม")
        
        # วิเคราะห์ข้อความ
        all_keywords = {}
        
        for col_idx, col_description in text_columns.items():
            print(f"🔍 กำลังวิเคราะห์: {col_description}")
            
            column_results = {
                "total_texts": 0,
                "sentiment_dist": {"positive": 0, "neutral": 0, "negative": 0},
                "sample_texts": {"positive": [], "negative": [], "neutral": []},
                "keywords_by_category": {}
            }
            
            if col_idx < len(df.columns):
                texts = df.iloc[:, col_idx].dropna().astype(str).tolist()
                
                for text in texts:
                    text_clean = text.strip()
                    
                    # ข้ามข้อความที่ไม่มีเนื้อหา
                    if (not text_clean or 
                        len(text_clean) <= 2 or 
                        text_clean in ['-', 'ไม่มี', 'ไม่ใส่', 'ไม่ตอบ', 'ยินยอม', 'ไม่ยินยอม', 'สนใจ', 'ไม่สนใจ'] or
                        text_clean.isdigit()):
                        continue
                        
                    analysis = await self.analyze_single_text(text_clean, col_description)
                    
                    results["detailed_results"].append({
                        "column": col_description,
                        "text": text_clean,
                        **analysis
                    })
                    
                    sentiment = analysis["sentiment"]
                    if sentiment in results["sentiment_distribution"]:
                        results["sentiment_distribution"][sentiment] += 1
                        column_results["sentiment_dist"][sentiment] += 1
                    
                    column_results["total_texts"] += 1
                    
                    # เก็บตัวอย่างข้อความ
                    if len(column_results["sample_texts"][sentiment]) < 3:
                        column_results["sample_texts"][sentiment].append(text_clean[:150])
                    
                    # รวบรวม keywords
                    for kw in analysis["keywords"]:
                        word = kw["word"]
                        category = kw.get("category", "general")
                        
                        if word not in all_keywords:
                            all_keywords[word] = {
                                "count": 0, 
                                "total_score": 0, 
                                "sentiment_type": kw.get("sentiment_type", "neutral"),
                                "category": category,
                                "columns": set()
                            }
                        
                        all_keywords[word]["count"] += 1
                        all_keywords[word]["total_score"] += kw["score"]
                        all_keywords[word]["columns"].add(col_description)
                        
                        if category not in column_results["keywords_by_category"]:
                            column_results["keywords_by_category"][category] = {}
                        if word not in column_results["keywords_by_category"][category]:
                            column_results["keywords_by_category"][category][word] = 0
                        column_results["keywords_by_category"][category][word] += 1
            
            if column_results["total_texts"] > 0:
                results["column_analysis"][col_description] = column_results
                print(f"  ✅ วิเคราะห์แล้ว: {column_results['total_texts']} ข้อความ")
        
        results["texts_analyzed"] = len(results["detailed_results"])
        
        # สร้างรายการ keywords
        sorted_keywords = sorted(
            all_keywords.items(),
            key=lambda x: (x[1]["count"], x[1]["total_score"]),
            reverse=True
        )[:25]
        
        results["top_keywords"] = [
            {
                "word": word,
                "count": data["count"],
                "avg_score": data["total_score"] / data["count"],
                "sentiment_type": data["sentiment_type"],
                "category": data["category"],
                "appears_in_columns": len(data["columns"])
            }
            for word, data in sorted_keywords
        ]
        
        # สร้าง AI insights
        results["insights"] = await self.generate_ai_insights(
            results["sentiment_distribution"], 
            results["column_analysis"],
            results["top_keywords"],
            results["detailed_results"],
            likert_analysis,
            choice_analysis
        )
        
        print(f"✅ วิเคราะห์เสร็จสิ้น:")
        print(f"  📊 ข้อความทั้งหมด: {results['texts_analyzed']}")
        print(f"  📈 Sentiment: +{results['sentiment_distribution']['positive']} ={results['sentiment_distribution']['neutral']} -{results['sentiment_distribution']['negative']}")
        print(f"  🔑 Keywords: {len(results['top_keywords'])} คำสำคัญ")
        print(f"  📏 Likert scales: {len(likert_analysis)} คำถาม")
        print(f"  ☑️ Choice questions: {len(choice_analysis)} คำถาม")
        print(f"  🤖 โมเดลที่ใช้: {method_name} (Method {self.analysis_method})")
        
        return results
    
    def is_ready(self) -> bool:
        """ตรวจสอบว่าระบบพร้อมใช้งาน"""
        return self._ready
    
    def get_model_info(self) -> Dict[str, Any]:
        """ข้อมูลเกี่ยวกับโมเดลที่ใช้งาน"""
        method_info = self.available_methods[self.analysis_method]
        
        return {
            "analysis_method": self.analysis_method,
            "method_name": method_info["name"],
            "pythainlp_available": PYTHAINLP_AVAILABLE,
            "training_data_path": self.training_data_path,
            "models_path": self.models_path,
            "available_methods": self.get_available_methods(),
            "features": {   
                "thai_tokenization": PYTHAINLP_AVAILABLE,
                "pos_tagging": PYTHAINLP_AVAILABLE,
                "gzip_model": self.analysis_method == 1 and self.model_trained,
                "ssense_api": self.analysis_method == 2,
                "rule_based": True,
                "survey_specific_keywords": True,
                "advanced_preprocessing": True,
                "method_switching": True
            },
            "method_specific": {
                "gzip_trained": self.model_trained if self.analysis_method == 1 else False,
                "training_data_available": os.path.exists(os.path.join(self.training_data_path, "pos.txt")),
                "ssense_api_key": bool(self.ssense_api_key) if self.analysis_method == 2 else False
            }
        }
    
    async def retrain_model(self, new_training_data: List[Tuple[str, str]] = None) -> bool:
        """เทรนโมเดลใหม่ด้วยข้อมูลใหม่ (สำหรับ GzipModel เท่านั้น)"""
        try:
            if self.analysis_method != 1:
                print(f"❌ การเทรนโมเดลใหม่ใช้ได้เฉพาะ GzipModel (Method 1) เท่านั้น")
                print(f"   ปัจจุบันใช้ {self.available_methods[self.analysis_method]['name']} (Method {self.analysis_method})")
                return False
                
            if not PYTHAINLP_AVAILABLE:
                print("❌ PyThaiNLP ไม่พร้อมใช้งาน ไม่สามารถเทรนโมเดลได้")
                return False
            
            print("🔄 กำลังเทรน GzipModel ใหม่...")
            
            # ใช้ข้อมูลที่ส่งมา หรือโหลดจากไฟล์
            if new_training_data:
                training_data = new_training_data
                print(f"📝 ใช้ข้อมูลใหม่: {len(training_data)} ประโยค")
            else:
                # โหลดข้อมูลจากไฟล์อีกครั้ง
                await self._load_and_train_gzip_model()
                return self.model_trained
            
            # สร้างโมเดลใหม่
            self.trained_model = GzipModel(training_data)
            self.model_trained = True
            
            print(f"✅ เทรน GzipModel ใหม่สำเร็จ!")
            return True
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการเทรนโมเดลใหม่: {e}")
            return False
    
    async def predict_sentiment_async(self, text: str) -> Dict[str, Any]:
        """ทำนาย sentiment สำหรับข้อความเดี่ยว (async version)"""
        try:
            processed_text = self._preprocess_text(text)
            
            if self.analysis_method == 2:
                # SSense API method
                result = await self._analyze_sentiment_ssense(processed_text)
            else:
                # Rule-based หรือ GzipModel
                result = self.analyze_sentiment(processed_text)
            
            return {
                "text": text,
                "processed_text": processed_text,
                "predicted_sentiment": result["sentiment"],
                "method": result.get("method", "unknown"),
                "method_id": result.get("method_id", self.analysis_method),
                "analysis_method": self.analysis_method,
                "method_name": self.available_methods[self.analysis_method]["name"],
                "ssense_data": result.get("ssense_raw") if self.analysis_method == 2 else None
            }
            
        except Exception as e:
            return {
                "text": text,
                "error": str(e),
                "predicted_sentiment": "neutral",
                "analysis_method": self.analysis_method
            }
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """ทำนาย sentiment สำหรับข้อความเดี่ยว (สำหรับทดสอบ)"""
        try:
            processed_text = self._preprocess_text(text)
            
            # สำหรับ SSense API ให้ fallback เป็น rule-based ใน sync context
            if self.analysis_method == 2:
                print("⚠️ SSense API not available in sync context, using rule-based fallback")
                original_method = self.analysis_method
                self.analysis_method = 0
                result = self.analyze_sentiment(processed_text)
                self.analysis_method = original_method
                result["method"] = "ssense_sync_fallback"
            else:
                result = self.analyze_sentiment(processed_text)
            
            return {
                "text": text,
                "processed_text": processed_text,
                "predicted_sentiment": result["sentiment"],
                "method": result.get("method", "unknown"),
                "method_id": result.get("method_id", self.analysis_method),
                "analysis_method": self.analysis_method,
                "method_name": self.available_methods[self.analysis_method]["name"]
            }
            
        except Exception as e:
            return {
                "text": text,
                "error": str(e),
                "predicted_sentiment": "neutral",
                "analysis_method": self.analysis_method
            }
    
    async def _create_sample_training_data(self) -> List[Tuple[str, str]]:
        """สร้างข้อมูลตัวอย่างสำหรับเทรนโมเดล"""
        sample_data = [
            # Positive samples
            ("ระบบใช้งานง่ายมาก", "positive"),
            ("สะดวกสบาย ไม่ต้องมาธนาคาร", "positive"),
            ("เร็วดี ประหยัดเวลา", "positive"),
            ("ชอบมาก แนะนำเลย", "positive"),
            ("ใช้งานได้ดีมาก", "positive"),
            
            # Negative samples  
            ("ช้ามาก สับสนมาก", "negative"),
            ("ยุ่งยาก ไม่เข้าใจ", "negative"),
            ("ปุ่มแก้ไขหายาก", "negative"),
            ("ภาษาไทยไม่สื่อความหมาย", "negative"),
            ("ควรปรับปรุงระบบ", "negative"),
            
            # Neutral samples
            ("ปกติ ไม่มีอะไร", "neutral"),
            ("ใช้ได้ พอใช้", "neutral"),
            ("ไม่มีความคิดเห็น", "neutral"),
            ("ทดสอบระบบ", "neutral"),
            ("ข้อมูลทั่วไป", "neutral")
        ]
        
        print(f"📝 สร้างข้อมูลตัวอย่าง: {len(sample_data)} ประโยค")
        return sample_data


# ตัวอย่างการใช้งาน
async def main():
    """ตัวอย่างการใช้งาน Enhanced NLP Processor with SSense API"""
    
    print("🚀 ทดสอบ NLP Processor with Multiple Methods\n")
    
    # ทดสอบทั้ง 3 methods
    for method in [0, 1, 2]:
        print(f"\n{'='*60}")
        print(f"🧪 ทดสอบ Method {method}: {['Rule-based', 'GzipModel', 'SSense API'][method]}")
        print(f"{'='*60}")
        
        # สร้าง processor
        processor = NLPProcessor(analysis_method=method)
        
        # เตรียมระบบ
        await processor.initialize()
        
        # ข้อมูลโมเดล
        model_info = processor.get_model_info()
        print(f"\n📋 ข้อมูลโมเดล:")
        print(f"   Method: {model_info['method_name']} (ID: {model_info['analysis_method']})")
        print(f"   PyThaiNLP: {'✅' if model_info['pythainlp_available'] else '❌'}")
        
        # ทดสอบ sentiment analysis
        test_texts = [
            "ระบบใช้งานง่ายมาก ชอบเลย",
            "ช้ามาก สับสนมาก", 
            "ปกติ ไม่มีอะไร",
            "ปุ่มแก้ไขหายาก ควรปรับปรุง",
            "สะดวกสบาย ไม่ต้องมาธนาคาร"
        ]
        
        print(f"\n🧪 ทดสอบการวิเคราะห์ sentiment:")
        for text in test_texts:
            result = processor.predict_sentiment(text)
            print(f"  📝 '{text}'")
            print(f"     → {result['predicted_sentiment']}")
            print(f"     → วิธี: {result['method']}")
        
        # ทดสอบการเปลี่ยน method
        available_methods = processor.get_available_methods()
        print(f"\n🔄 Methods ที่ใช้ได้: {list(available_methods.keys())}")
        for mid, minfo in available_methods.items():
            print(f"   {mid}: {minfo['name']}")
    
    print(f"\n{'='*60}")
    print("✅ การทดสอบเสร็จสิ้น!")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())