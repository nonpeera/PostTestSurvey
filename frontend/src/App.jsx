import React, { useState, useRef, useCallback, useMemo } from 'react';
import UploadSection from './components/UploadSection';
import Header from './components/Header';
import OverviewTab from './components/OverviewTab';
import FeedbackTab from './components/FeedbackTab';
import AIInsightsTab from './components/AIInsightsTab';
import ColumnsTab from './components/ColumnsTab';

const SurveyDashboard = () => {
  const [uploadData, setUploadData] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedColumn, setSelectedColumn] = useState('all');
  const [sentimentFilter, setSentimentFilter] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const [sortBy, setSortBy] = useState('confidence');
  const [sortOrder, setSortOrder] = useState('desc');
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedKeyword, setSelectedKeyword] = useState(null);
  const fileInputRef = useRef(null);

  const API_BASE = 'http://localhost:8000';
  const itemsPerPage = 20;

  // Handle file upload
  const handleFileUpload = useCallback(async (file) => {
    if (!file) return;
    
    if (!file.name.match(/\.(xlsx|xls|csv)$/)) {
      alert('กรุณาเลือกไฟล์ Excel (.xlsx, .xls) หรือ CSV เท่านั้น');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      setUploadProgress(30);
      
      const response = await fetch(`${API_BASE}/api/analyze-survey`, {
        method: 'POST',
        body: formData,
      });

      setUploadProgress(80);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setUploadProgress(100);
      
      setTimeout(() => {
        setUploadData(data);
        setIsUploading(false);
        setUploadProgress(0);
        setCurrentPage(1);
      }, 500);

    } catch (error) {
      console.error('Upload error:', error);
      alert('เกิดข้อผิดพลาดในการอัปโหลด: ' + error.message);
      setIsUploading(false);
      setUploadProgress(0);
    }
  }, []);

  // Drag and drop handlers
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, [handleFileUpload]);

  const handleFileInputChange = useCallback((e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileUpload(file);
    }
  }, [handleFileUpload]);

  // Get unique columns for filter
  const uniqueColumns = useMemo(() => {
    if (!uploadData?.detailed_results) return [];
    const columns = [...new Set(uploadData.detailed_results.map(item => item.column))].filter(Boolean);
    return columns.filter(col => !col.includes('ชื่อ') && !col.includes('อายุ') && !col.includes('เพศ'));
  }, [uploadData]);

  // Calculate interest percentages
  const interestTotal = useMemo(() => {
    if (!uploadData?.choice_analysis?.["ความสนใจทดสอบอีกครั้ง"]) return { interested: 0, notInterested: 0, total: 0 };
    
    const data = uploadData.choice_analysis["ความสนใจทดสอบอีกครั้ง"];
    const interested = data["สนใจ"] || 0;
    const notInterested = data["ไม่สนใจ"] || 0;
    const total = interested + notInterested;
    
    return { interested, notInterested, total };
  }, [uploadData]);

  // Calculate percentages for sentiment
  const total = uploadData ? 
    (uploadData.sentiment_distribution?.positive || 0) + 
    (uploadData.sentiment_distribution?.neutral || 0) + 
    (uploadData.sentiment_distribution?.negative || 0) : 0;

  const positivePercent = total > 0 ? Math.round((uploadData.sentiment_distribution?.positive || 0) / total * 100) : 0;
  const neutralPercent = total > 0 ? Math.round((uploadData.sentiment_distribution?.neutral || 0) / total * 100) : 0;
  const negativePercent = total > 0 ? Math.round((uploadData.sentiment_distribution?.negative || 0) / total * 100) : 0;

  // Shared props object
  const sharedProps = {
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
    interestTotal,
    positivePercent,
    neutralPercent,
    negativePercent,
    itemsPerPage
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto">
        
        {/* Upload Section */}
        {!uploadData && (
          <UploadSection
            isDragOver={isDragOver}
            isUploading={isUploading}
            uploadProgress={uploadProgress}
            fileInputRef={fileInputRef}
            handleDragOver={handleDragOver}
            handleDragLeave={handleDragLeave}
            handleDrop={handleDrop}
            handleFileInputChange={handleFileInputChange}
          />
        )}

        {/* Header */}
        {uploadData && (
          <Header
            uploadData={uploadData}
            setUploadData={setUploadData}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
          />
        )}

        {/* Tab Content */}
        {uploadData && activeTab === 'overview' && (
          <OverviewTab {...sharedProps} />
        )}

        {uploadData && activeTab === 'feedback' && (
          <FeedbackTab {...sharedProps} />
        )}

        {uploadData && activeTab === 'insights' && (
          <AIInsightsTab {...sharedProps} />
        )}

        {uploadData && activeTab === 'columns' && (
          <ColumnsTab {...sharedProps} />
        )}
      </div>
    </div>
  );
};

export default SurveyDashboard;