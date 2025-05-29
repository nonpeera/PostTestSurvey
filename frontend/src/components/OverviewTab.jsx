import React from 'react';
import { SentimentChart, InterestChart, LikertChart } from './Charts';

const OverviewTab = ({ 
  uploadData, 
  selectedKeyword, 
  setSelectedKeyword, 
  interestTotal, 
  positivePercent, 
  neutralPercent, 
  negativePercent 
}) => {
  
  return (
    <>
      {/* Main Dashboard */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        
        {/* Sentiment Analysis Card */}
        <SentimentChart 
          uploadData={uploadData}
          positivePercent={positivePercent}
          neutralPercent={neutralPercent}
          negativePercent={negativePercent}
        />

        {/* Interest Chart Card */}
        <InterestChart interestTotal={interestTotal} />

        {/* Scale Scores Card */}
        <LikertChart uploadData={uploadData} />
      </div>

      {/* Keywords */}
      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">คำสำคัญที่พบบ่อย</h2>
        <div className="flex flex-wrap gap-2">
          {(uploadData.top_keywords || []).slice(0, 20).map((keyword, index) => (
            <button
              key={index}
              onClick={() => setSelectedKeyword(selectedKeyword === keyword.word ? null : keyword.word)}
              className={`relative px-3 py-1 rounded-full text-sm font-medium transition-colors cursor-pointer ${
                selectedKeyword === keyword.word
                  ? 'bg-blue-700 text-white'
                  : keyword.sentiment_type === 'positive'
                  ? 'bg-green-600 text-white hover:bg-green-700'
                  : keyword.sentiment_type === 'negative'
                  ? 'bg-red-600 text-white hover:bg-red-700'
                  : 'bg-gray-600 text-white hover:bg-gray-700'
              }`}
            >
              {keyword.word}
              <span className="absolute -top-1 -right-1 bg-yellow-500 text-black text-xs rounded-full w-5 h-5 flex items-center justify-center">
                {keyword.count}
              </span>
            </button>
          ))}
        </div>
        {selectedKeyword && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <p className="text-sm text-blue-800">
              กำลังกรองความคิดเห็นที่เกี่ยวข้องกับ "<strong>{selectedKeyword}</strong>" 
              - คลิกที่แท็บ "ความคิดเห็นทั้งหมด" เพื่อดูผลลัพธ์
            </p>
          </div>
        )}
      </div>

      {/* Choice Analysis */}
      {uploadData.choice_analysis && Object.keys(uploadData.choice_analysis).length > 0 && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">คำตอบแบบเลือก</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(uploadData.choice_analysis)
              .filter(([question]) => !question.includes("ความสนใจทดสอบอีกครั้ง"))
              .map(([question, answers]) => (
              <div key={question} className="border border-gray-200 rounded-lg p-4">
                <h3 className="font-medium text-gray-800 mb-3">{question}</h3>
                <div className="space-y-2">
                  {Object.entries(answers).slice(0, 5).map(([answer, count]) => (
                    <div key={answer} className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 flex-1 truncate">{answer}</span>
                      <span className="text-sm font-semibold text-gray-800 ml-2">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
};

export default OverviewTab;