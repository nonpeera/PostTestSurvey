import React, { useMemo } from 'react';

const FeedbackTab = ({
  uploadData,
  selectedColumn,
  setSelectedColumn,
  sentimentFilter,
  setSentimentFilter,
  sortBy,
  setSortBy,
  sortOrder,
  setSortOrder,
  currentPage,
  setCurrentPage,
  selectedKeyword,
  setSelectedKeyword,
  uniqueColumns,
  itemsPerPage
}) => {

  // Filter and sort feedback data
  const filteredAndSortedFeedback = useMemo(() => {
    if (!uploadData?.detailed_results) return [];

    let filtered = uploadData.detailed_results.filter(item => {
      const columnMatch = selectedColumn === 'all' || item.column === selectedColumn;
      const sentimentMatch = sentimentFilter === 'all' || item.sentiment === sentimentFilter;
      const keywordMatch = !selectedKeyword || 
        item.keywords?.some(kw => kw.word === selectedKeyword) ||
        item.text.toLowerCase().includes(selectedKeyword.toLowerCase());
      return columnMatch && sentimentMatch && keywordMatch;
    });

    filtered.sort((a, b) => {
      let aVal, bVal;
      
      switch (sortBy) {
        case 'sentiment':
          aVal = a.sentiment;
          bVal = b.sentiment;
          break;
        case 'column':
          aVal = a.column || '';
          bVal = b.column || '';
          break;
        case 'length':
          aVal = a.text.length;
          bVal = b.text.length;
          break;
        default:
          aVal = a.text.length;
          bVal = b.text.length;
      }

      if (sortOrder === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });

    return filtered;
  }, [uploadData, selectedColumn, sentimentFilter, selectedKeyword, sortBy, sortOrder]);

  // Pagination
  const paginatedFeedback = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return filteredAndSortedFeedback.slice(startIndex, startIndex + itemsPerPage);
  }, [filteredAndSortedFeedback, currentPage, itemsPerPage]);

  const totalPages = Math.ceil(filteredAndSortedFeedback.length / itemsPerPage);

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive': return 'text-green-600 bg-green-50 border-green-200';
      case 'negative': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getSentimentBadge = (sentiment) => {
    switch (sentiment) {
      case 'positive': return 'bg-green-600 text-white';
      case 'negative': return 'bg-red-600 text-white';
      default: return 'bg-gray-600 text-white';
    }
  };

  const getSentimentLabel = (sentiment) => {
    switch (sentiment) {
      case 'positive': return 'เชิงบวก';
      case 'negative': return 'เชิงลบ';
      default: return 'เป็นกลาง';
    }
  };

  const getMethodBadge = (method) => {
    switch (method) {
      case 'rule_based': 
      case 'rule_based_error':
        return { color: 'bg-blue-100 text-blue-800', label: 'Rule-based' };
      case 'gzip_model':
        return { color: 'bg-purple-100 text-purple-800', label: 'GzipModel' };
      case 'ssense_api':
      case 'ssense_fallback_rule':
      case 'ssense_fallback_error':
        return { color: 'bg-orange-100 text-orange-800', label: 'SSense API' };
      default:
        return { color: 'bg-gray-100 text-gray-800', label: 'Unknown' };
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900">ความคิดเห็นทั้งหมด</h2>
        {selectedKeyword && (
          <button
            onClick={() => setSelectedKeyword(null)}
            className="text-sm text-blue-600 hover:text-blue-700"
          >
            ยกเลิกการกรอง "{selectedKeyword}"
          </button>
        )}
      </div>

      {/* Analysis Method Info */}
      {uploadData?.model_info && (
        <div className="mb-4 p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-700">
              <span className="font-medium">วิธีการวิเคราะห์:</span> {uploadData.model_info.method_name || uploadData.model_info.engine}
            </div>
            <div className="text-xs text-gray-500">
              Version {uploadData.model_info.version || 'Unknown'}
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">คำถาม/คอลัมน์</label>
          <select
            value={selectedColumn}
            onChange={(e) => setSelectedColumn(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md text-sm"
          >
            <option value="all">ทั้งหมด</option>
            {uniqueColumns.map(column => (
              <option key={column} value={column}>{column}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">ความรู้สึก</label>
          <select
            value={sentimentFilter}
            onChange={(e) => setSentimentFilter(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md text-sm"
          >
            <option value="all">ทั้งหมด</option>
            <option value="positive">เชิงบวก</option>
            <option value="neutral">เป็นกลาง</option>
            <option value="negative">เชิงลบ</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">เรียงตาม</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md text-sm"
          >
            <option value="sentiment">ความรู้สึก</option>
            <option value="column">คำถาม</option>
            <option value="length">ความยาวข้อความ</option>
          </select>
        </div>
      </div>

      {/* Sort Order */}
      <div className="mb-4 flex items-center justify-between">
        <div className="text-sm text-gray-600">
          แสดง {paginatedFeedback.length} จาก {filteredAndSortedFeedback.length} รายการ
          {selectedKeyword && (
            <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
              กรองด้วย: {selectedKeyword}
            </span>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value)}
            className="px-2 py-1 text-xs border border-gray-300 rounded-md"
          >
            <option value="desc">มากไปน้อย</option>
            <option value="asc">น้อยไปมาก</option>
          </select>
          {totalPages > 1 && (
            <>
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 text-sm border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
              >
                ก่อนหน้า
              </button>
              <span className="text-sm text-gray-600">
                หน้า {currentPage} จาก {totalPages}
              </span>
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1 text-sm border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
              >
                ถัดไป
              </button>
            </>
          )}
        </div>
      </div>

      {/* Feedback List */}
      <div className="space-y-4">
        {paginatedFeedback.length > 0 ? (
          paginatedFeedback.map((feedback, index) => {
            const methodInfo = getMethodBadge(feedback.debug?.method || 'unknown');
            
            return (
              <div key={index} className={`border-l-4 p-4 rounded-r-lg border ${getSentimentColor(feedback.sentiment)}`}>
                <div className="flex items-start justify-between mb-2">
                  <div className="text-sm text-gray-700 leading-relaxed flex-1">
                    "{feedback.text}"
                  </div>
                </div>
                
                <div className="flex items-center justify-between text-xs flex-wrap gap-2">
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full font-medium ${getSentimentBadge(feedback.sentiment)}`}>
                      {getSentimentLabel(feedback.sentiment)}
                    </span>
                    <span className={`px-2 py-1 rounded text-xs ${methodInfo.color}`}>
                      {methodInfo.label}
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-2 text-gray-500">
                    <span className="max-w-xs truncate">
                      {feedback.column || 'ทั่วไป'}
                    </span>
                    {feedback.debug?.positive_words?.length > 0 && (
                      <span className="text-green-600" title={`คำบวก: ${feedback.debug.positive_words.join(', ')}`}>
                        +{feedback.debug.positive_words.length}
                      </span>
                    )}
                    {feedback.debug?.negative_words?.length > 0 && (
                      <span className="text-red-600" title={`คำลบ: ${feedback.debug.negative_words.join(', ')}`}>
                        -{feedback.debug.negative_words.length}
                      </span>
                    )}
                  </div>
                </div>

                {/* Keywords */}
                {feedback.keywords && feedback.keywords.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-gray-100">
                    <div className="text-xs text-gray-500 mb-1">คำสำคัญ:</div>
                    <div className="flex flex-wrap gap-1">
                      {feedback.keywords.slice(0, 5).map((keyword, kidx) => (
                        <span 
                          key={kidx}
                          className={`px-2 py-1 rounded text-xs ${
                            keyword.sentiment_type === 'positive' 
                              ? 'bg-green-100 text-green-700'
                              : keyword.sentiment_type === 'negative'
                              ? 'bg-red-100 text-red-700'
                              : 'bg-gray-100 text-gray-700'
                          }`}
                        >
                          {keyword.word}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Debug Info (สำหรับ SSense API) */}
                {feedback.debug?.ssense_raw && (
                  <div className="mt-2 pt-2 border-t border-gray-100">
                    <details className="text-xs">
                      <summary className="cursor-pointer text-gray-500 hover:text-gray-700">
                        ข้อมูลเพิ่มเติมจาก SSense API
                      </summary>
                      <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                        {feedback.debug.ssense_raw.intention && (
                          <div className="mb-2">
                            <span className="font-medium">ความตั้งใจ:</span>
                            {Object.entries(feedback.debug.ssense_raw.intention).map(([key, value]) => (
                              value > 0 && (
                                <span key={key} className="ml-2 px-1 py-0.5 bg-blue-100 text-blue-800 rounded">
                                  {key}: {value}%
                                </span>
                              )
                            ))}
                          </div>
                        )}
                        {feedback.debug.ssense_raw.preprocess?.segmented && (
                          <div>
                            <span className="font-medium">การตัดคำ:</span>
                            <span className="ml-2 text-gray-600">
                              {feedback.debug.ssense_raw.preprocess.segmented.join(' | ')}
                            </span>
                          </div>
                        )}
                      </div>
                    </details>
                  </div>
                )}
              </div>
            );
          })
        ) : (
          <div className="text-center py-8 text-gray-500">
            ไม่พบความคิดเห็นที่ตรงกับเงื่อนไขการกรอง
          </div>
        )}
      </div>
    </div>
  );
};

export default FeedbackTab;