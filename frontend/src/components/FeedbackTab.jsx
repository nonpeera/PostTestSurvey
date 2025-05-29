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
        case 'confidence':
          aVal = a.confidence || 0;
          bVal = b.confidence || 0;
          break;
        case 'sentiment':
          aVal = a.sentiment;
          bVal = b.sentiment;
          break;
        case 'column':
          aVal = a.column || '';
          bVal = b.column || '';
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

      {/* Filters */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
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
            <option value="confidence">ความมั่นใจ</option>
            <option value="sentiment">ความรู้สึก</option>
            <option value="column">คำถาม</option>
            <option value="length">ความยาวข้อความ</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">ลำดับ</label>
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md text-sm"
          >
            <option value="desc">มากไปน้อย</option>
            <option value="asc">น้อยไปมาก</option>
          </select>
        </div>
      </div>

      {/* Results Summary */}
      <div className="mb-4 flex items-center justify-between">
        <div className="text-sm text-gray-600">
          แสดง {paginatedFeedback.length} จาก {filteredAndSortedFeedback.length} รายการ
          {selectedKeyword && (
            <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
              กรองด้วย: {selectedKeyword}
            </span>
          )}
        </div>
        {totalPages > 1 && (
          <div className="flex items-center space-x-2">
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
          </div>
        )}
      </div>

      {/* Feedback List */}
      <div className="space-y-4">
        {paginatedFeedback.length > 0 ? (
          paginatedFeedback.map((feedback, index) => (
            <div key={index} className={`border-l-4 p-4 rounded-r-lg border ${getSentimentColor(feedback.sentiment)}`}>
              <div className="flex items-start justify-between mb-2">
                <div className="text-sm text-gray-700 leading-relaxed flex-1">
                  "{feedback.text}"
                </div>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className={`px-2 py-1 rounded-full font-medium ${getSentimentBadge(feedback.sentiment)}`}>
                  {getSentimentLabel(feedback.sentiment)}
                </span>
                <span className="text-gray-500">
                  ความมั่นใจ: {Math.round((feedback.confidence || 0) * 100)}%
                </span>
                <span className="text-gray-500 max-w-xs truncate">
                  {feedback.column || 'ทั่วไป'}
                </span>
              </div>
            </div>
          ))
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