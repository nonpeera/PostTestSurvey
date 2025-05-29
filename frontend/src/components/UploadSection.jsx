import React from 'react';

const UploadSection = ({
  isDragOver,
  isUploading,
  uploadProgress,
  fileInputRef,
  handleDragOver,
  handleDragLeave,
  handleDrop,
  handleFileInputChange
}) => {
  return (
    <div className="bg-white rounded-lg shadow-sm p-8 mb-8">
      <div 
        className={`border-2 border-dashed rounded-lg p-12 text-center transition-all duration-300 cursor-pointer ${
          isDragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="text-4xl mb-4 text-gray-400">📊</div>
        <h3 className="text-xl font-semibold text-gray-700 mb-2">อัปโหลดแบบสอบถาม Post-Test Survey</h3>
        <p className="text-gray-600 mb-6">ลากไฟล์ Excel (.xlsx) มาวางที่นี่ หรือคลิกเพื่อเลือกไฟล์</p>
        <button className="bg-blue-600 text-white px-6 py-2 rounded-md font-medium hover:bg-blue-700 transition-colors">
          เลือกไฟล์
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".xlsx,.xls,.csv"
          onChange={handleFileInputChange}
          className="hidden"
        />
      </div>

      {isUploading && (
        <div className="mt-6">
          <div className="bg-gray-200 rounded-full h-3 overflow-hidden">
            <div 
              className="bg-blue-600 h-full transition-all duration-500 ease-out"
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
          <p className="text-center mt-2 text-gray-600">
            {uploadProgress < 50 ? 'กำลังอัปโหลด...' : 
             uploadProgress < 90 ? 'กำลังวิเคราะห์...' : 'เกือบเสร็จแล้ว...'}
          </p>
        </div>
      )}
    </div>
  );
};

export default UploadSection;