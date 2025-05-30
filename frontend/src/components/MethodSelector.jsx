import React from 'react';
import MethodSelector from './MethodSelector';

const Header = ({ uploadData, setUploadData, activeTab, setActiveTab, onMethodChange }) => {
  return (
    <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            Survey Analysis Dashboard
          </h1>
          <p className="text-gray-600">
            ผลการวิเคราะห์ความรู้สึกและความคิดเห็นจากแบบสอบถาม Post-Test Survey
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <MethodSelector 
            onMethodChange={onMethodChange}
            currentMethod={uploadData?.model_info?.analysis_method}
          />
          <button 
            onClick={() => setUploadData(null)}
            className="bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700 transition-colors"
          >
            อัปโหลดไฟล์ใหม่
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="bg-gray-50 p-3 rounded-md">
          <div className="text-sm text-gray-600">ไฟล์</div>
          <div className="font-semibold text-gray-900 truncate">{uploadData.file_info?.filename || 'แบบสอบถาม'}</div>
        </div>
        <div className="bg-gray-50 p-3 rounded-md">
          <div className="text-sm text-gray-600">จำนวนผู้ตอบ</div>
          <div className="font-semibold text-gray-900">{uploadData.total_responses || 0}</div>
        </div>
        <div className="bg-gray-50 p-3 rounded-md">
          <div className="text-sm text-gray-600">ข้อความวิเคราะห์</div>
          <div className="font-semibold text-gray-900">{uploadData.texts_analyzed || 0}</div>
        </div>
        <div className="bg-gray-50 p-3 rounded-md">
          <div className="text-sm text-gray-600">เวลาประมวลผล</div>
          <div className="font-semibold text-gray-900">{uploadData.processing_summary?.processing_time || 'N/A'}</div>
        </div>
        <div className="bg-gray-50 p-3 rounded-md">
          <div className="text-sm text-gray-600">วิธีการวิเคราะห์</div>
          <div className="font-semibold text-gray-900 truncate">
            {uploadData.model_info?.method_name || uploadData.processing_summary?.method_used || 'N/A'}
          </div>
        </div>
      </div>

      {/* Analysis Method Info Banner */}
      {uploadData?.model_info && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="text-blue-600">
                {uploadData.model_info.analysis_method === 0 && '📋'}
                {uploadData.model_info.analysis_method === 1 && '🤖'}
                {uploadData.model_info.analysis_method === 2 && '🌐'}
              </div>
              <div>
                <div className="text-sm font-medium text-blue-900">
                  การวิเคราะห์ด้วย {uploadData.model_info.method_name}
                </div>
                <div className="text-xs text-blue-700">
                  {uploadData.model_info.features || uploadData.model_info.preprocessing}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-blue-600">Version {uploadData.model_info.version || '3.0'}</div>
              {uploadData.processing_summary?.ai_enhancement && (
                <div className="text-xs text-purple-600">✨ AI Enhanced</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="mt-6 border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', label: 'ภาพรวม' },
            { id: 'feedback', label: 'ความคิดเห็นทั้งหมด' },
            { id: 'insights', label: 'AI Insights' },
            { id: 'columns', label: 'วิเคราะห์ตามคำถาม' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>
    </div>
  );
};

export default Header;