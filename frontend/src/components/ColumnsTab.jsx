import React from 'react';

const ColumnsTab = ({ uploadData }) => {
  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive': return 'text-green-600 bg-green-50 border-green-200';
      case 'negative': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getSentimentLabel = (sentiment) => {
    switch (sentiment) {
      case 'positive': return 'เชิงบวก';
      case 'negative': return 'เชิงลบ';
      default: return 'เป็นกลาง';
    }
  };

  return (
    <div className="space-y-6">
      {uploadData.column_analysis && Object.keys(uploadData.column_analysis).length > 0 ? (
        Object.entries(uploadData.column_analysis).map(([columnName, columnData]) => (
          <div key={columnName} className="bg-white rounded-lg shadow-sm p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">{columnName}</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Sentiment Distribution */}
              <div>
                <h4 className="font-medium text-gray-800 mb-3">การกระจายความรู้สึก</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-green-600">เชิงบวก</span>
                    <span className="font-semibold">{columnData.sentiment_dist.positive}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">เป็นกลาง</span>
                    <span className="font-semibold">{columnData.sentiment_dist.neutral}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-red-600">เชิงลบ</span>
                    <span className="font-semibold">{columnData.sentiment_dist.negative}</span>
                  </div>
                  <div className="text-xs text-gray-500 mt-2">
                    รวม {columnData.total_texts} ความคิดเห็น
                  </div>
                </div>
              </div>

              {/* Sample Texts */}
              <div className="lg:col-span-2">
                <h4 className="font-medium text-gray-800 mb-3">ตัวอย่างความคิดเห็น</h4>
                <div className="space-y-3">
                  {['positive', 'negative', 'neutral'].map(sentiment => (
                    columnData.sample_texts[sentiment]?.length > 0 && (
                      <div key={sentiment} className="space-y-2">
                        <div className="text-xs font-medium text-gray-600 uppercase">
                          {getSentimentLabel(sentiment)}
                        </div>
                        {columnData.sample_texts[sentiment].slice(0, 2).map((text, i) => (
                          <div key={i} className={`text-sm p-2 rounded border-l-4 ${getSentimentColor(sentiment)}`}>
                            "{text}"
                          </div>
                        ))}
                      </div>
                    )
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))
      ) : (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="text-center py-8 text-gray-500">
            ไม่พบข้อมูลการวิเคราะห์ตามคำถาม
          </div>
        </div>
      )}
    </div>
  );
};

export default ColumnsTab;