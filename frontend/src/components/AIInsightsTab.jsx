import React from 'react';

const AIInsightsTab = ({ uploadData, interestTotal, positivePercent }) => {
  const insights = uploadData.insights || {};

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900">AI Insights & Recommendations</h2>
        <div className="flex items-center space-x-2">
          <div className="text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded">
            AI Analysis 
          </div>
          {insights?.ai_generated && (
            <div className="text-xs text-blue-600 bg-blue-100 px-3 py-1 rounded">
              Gemini-2.0-flash
            </div>
          )}
          {uploadData?.model_info?.method_name && (
            <div className="text-xs text-purple-600 bg-purple-100 px-3 py-1 rounded">
               {uploadData.model_info.method_name}
            </div>
          )}
        </div>
      </div>

      {/* Executive Summary */}
      {insights?.executive_summary && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
          <h3 className="font-semibold text-blue-800 mb-2 flex items-center">
            <span className="mr-2"></span>สรุปภาพรวม
          </h3>
          <p className="text-sm text-blue-700">{insights.executive_summary}</p>
        </div>
      )}

      {/* Analysis Method Information */}
      {uploadData?.model_info && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-6">
          <h3 className="font-semibold text-gray-800 mb-3 flex items-center">
            <span className="mr-2"></span>วิธีการวิเคราะห์
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <div className="font-medium text-gray-700">โมเดลที่ใช้</div>
              <div className="text-gray-600">{uploadData.model_info.method_name || uploadData.model_info.engine}</div>
            </div>
          </div>
          
          {/* Available Methods */}
          {uploadData.model_info.available_methods && (
            <div className="mt-3 pt-3 border-t border-gray-300">
              <div className="text-xs text-gray-600 mb-2">วิธีการวิเคราะห์ที่สามารถใช้ได้:</div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(uploadData.model_info.available_methods).map(([id, method]) => (
                  <span 
                    key={id}
                    className={`px-2 py-1 rounded text-xs ${
                      parseInt(id) === uploadData.model_info.analysis_method
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-200 text-gray-700'
                    }`}
                  >
                    {method.name}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h3 className="font-semibold text-green-800 mb-3 flex items-center">
            <span className="mr-2">✅</span>จุดแข็งของระบบ
          </h3>
          <div className="text-sm text-green-700 space-y-1 max-h-40 overflow-y-auto">
            {insights.positive_aspects?.length > 0 ? (
              insights.positive_aspects.map((aspect, i) => (
                <div key={i} className="flex items-start">
                  <span className="mr-2 text-green-600">•</span><span>{aspect}</span>
                </div>
              ))
            ) : (
              <div className="text-gray-500 italic">ไม่มีข้อมูล</div>
            )}
          </div>
        </div>

        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="font-semibold text-red-800 mb-3 flex items-center">
            <span className="mr-2">⚠️</span>จุดที่ต้องปรับปรุง
          </h3>
          <div className="text-sm text-red-700 space-y-1 max-h-40 overflow-y-auto">
            {insights.negative_aspects?.length > 0 ? (
              insights.negative_aspects.map((aspect, i) => (
                <div key={i} className="flex items-start">
                  <span className="mr-2 text-red-600">•</span><span>{aspect}</span>
                </div>
              ))
            ) : (
              <div className="text-gray-500 italic">ไม่มีข้อมูล</div>
            )}
          </div>
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-800 mb-3 flex items-center">
            <span className="mr-2">💡</span>ข้อเสนอแนะ AI
          </h3>
          <div className="text-sm text-blue-700 space-y-1 max-h-40 overflow-y-auto">
            {insights.recommendations?.length > 0 ? (
              insights.recommendations.slice(0, 5).map((rec, i) => (
                <div key={i} className="flex items-start">
                  <span className="mr-2 font-medium text-blue-600">{i+1}.</span><span>{rec}</span>
                </div>
              ))
            ) : (
              <div className="text-gray-500 italic">ไม่มีข้อมูล</div>
            )}
          </div>
        </div>
      </div>

      {/* Priority Actions */}
      {insights?.priority_actions?.length > 0 && (
        <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 mb-6">
          <h3 className="font-semibold text-orange-800 mb-3 flex items-center">
            <span className="mr-2">🔥</span>การดำเนินการลำดับความสำคัญ
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {insights.priority_actions.slice(0, 3).map((action, i) => (
              <div key={i} className="bg-white border border-orange-300 rounded p-3">
                <div className="text-xs font-medium text-orange-600 mb-1">ลำดับที่ {i+1}</div>
                <div className="text-sm text-orange-800">{action}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Additional Insights */}
      {(insights.system_strengths?.length > 0 || insights.user_pain_points?.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {insights.system_strengths?.length > 0 && (
            <div className="border border-gray-200 rounded-lg p-4">
              <h3 className="font-semibold text-gray-800 mb-3 flex items-center">
                <span className="mr-2"></span>จุดเด่นของระบบ
              </h3>
              <div className="text-sm text-gray-700 space-y-1">
                {insights.system_strengths.map((strength, i) => (
                  <div key={i} className="flex items-start">
                    <span className="mr-2 text-yellow-500">-</span><span>{strength}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {insights.user_pain_points?.length > 0 && (
            <div className="border border-gray-200 rounded-lg p-4">
              <h3 className="font-semibold text-gray-800 mb-3 flex items-center">
                <span className="mr-2"></span>ปัญหาที่ผู้ใช้พบ
              </h3>
              <div className="text-sm text-gray-700 space-y-1">
                {insights.user_pain_points.map((pain, i) => (
                  <div key={i} className="flex items-start">
                    <span className="mr-2 text-red-500">-</span><span>{pain}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Interest Analysis */}
      {interestTotal.total > 0 && (
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 mb-6">
          <h3 className="font-semibold text-purple-800 mb-3 flex items-center">
            <span className="mr-2"></span>การวิเคราะห์ความสนใจ
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {Math.round((interestTotal.interested / interestTotal.total) * 100)}%
              </div>
              <div className="text-sm text-purple-700">สนใจทดสอบอีกครั้ง</div>
              <div className="text-xs text-purple-600 mt-1">{interestTotal.interested} จาก {interestTotal.total} คน</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-600">{interestTotal.total}</div>
              <div className="text-sm text-gray-700">ผู้ตอบทั้งหมด</div>
              <div className="text-xs text-gray-600 mt-1">ในคำถามนี้</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {interestTotal.interested >= interestTotal.notInterested ? '👍' : '👎'}
              </div>
              <div className="text-sm text-orange-700">
                {interestTotal.interested >= interestTotal.notInterested ? 'แนวโน้มดี' : 'ควรปรับปรุง'}
              </div>
              <div className="text-xs text-orange-600 mt-1">
                สนใจ {interestTotal.interested >= interestTotal.notInterested ? 'มากกว่า' : 'น้อยกว่า'} ไม่สนใจ
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Sentiment Analysis Summary */}
      {insights?.sentiment_analysis && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 mb-6">
          <h3 className="font-semibold text-indigo-800 mb-3 flex items-center">
            <span className="mr-2">📈</span>การวิเคราะห์ความรู้สึกโดยรวม
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="text-center">
              <div className="text-lg font-bold text-indigo-600">
                {insights.sentiment_analysis.overall_mood || 'เป็นกลาง'}
              </div>
              <div className="text-sm text-indigo-700">อารมณ์โดยรวม</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-indigo-600">
                {insights.sentiment_analysis.satisfaction_level || 'ปานกลาง'}
              </div>
              <div className="text-sm text-indigo-700">ระดับความพึงพอใจ</div>
            </div>
          </div>
        </div>
      )}

      {/* Summary Stats (เอาความมั่นใจออก) */}
      <div className="bg-gray-50 rounded-lg p-4 mb-6">
        <h3 className="text-base font-semibold mb-3 flex items-center">
          <span className="mr-2">📊</span>สรุปภาพรวม
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-gray-900">{positivePercent}%</div>
            <div className="text-sm text-gray-600">ความพึงพอใจโดยรวม</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-gray-900">{uploadData.top_keywords?.length || 0}</div>
            <div className="text-sm text-gray-600">คำสำคัญที่สกัด</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-gray-900">{uploadData.texts_analyzed || 0}</div>
            <div className="text-sm text-gray-600">ข้อความที่วิเคราะห์</div>
          </div>
        </div>
      </div>

      {/* Processing Information */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <div className="flex items-start">
          <div className="text-yellow-600 mr-3 text-lg flex-shrink-0">
            {insights?.ai_generated ? '-' : '-'}
          </div>
          <div>
            <h4 className="font-medium text-yellow-800 mb-1">
              {insights?.ai_generated ? 'AI-Enhanced Analysis' : 'การพัฒนา AI Insights เพิ่มเติม'}
            </h4>
            <p className="text-sm text-yellow-700 mb-2">
              {insights?.ai_generated ? (
                `การวิเคราะห์นี้ได้รับการปรับปรุงด้วย ${insights.analysis_method || 'Gemini AI'} 
                เพื่อให้ได้ข้อมูลเชิงลึกที่แม่นยำและครอบคลุมมากยิ่งขึ้น`
              ) : (
                `ระบบใช้ ${uploadData?.model_info?.method_name || 'Enhanced Rule-Based Analysis'} 
                สำหรับการวิเคราะห์ความรู้สึกและสกัดข้อมูลเชิงลึก`
              )}
            </p>
            
            {/* Processing Time Info */}
            <div className="text-xs text-yellow-600 space-y-1">
              {uploadData.processing_summary?.processing_time && (
                <div>เวลาประมวลผล NLP: {uploadData.processing_summary.processing_time}</div>
              )}
              {uploadData.processing_summary?.ai_processing_time && (
                <div>เวลาประมวลผล AI: {uploadData.processing_summary.ai_processing_time}</div>
              )}
              {uploadData.processing_summary?.method_used && (
                <div>วิธีการ: {uploadData.processing_summary.method_used}</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIInsightsTab;