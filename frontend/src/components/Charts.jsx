import React from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title } from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title);

export const SentimentChart = ({ uploadData, positivePercent, neutralPercent, negativePercent }) => {
  const sentimentChartData = {
    labels: ['เชิงบวก', 'เป็นกลาง', 'เชิงลบ'],
    datasets: [{
      data: [
        uploadData.sentiment_distribution?.positive || 0,
        uploadData.sentiment_distribution?.neutral || 0,
        uploadData.sentiment_distribution?.negative || 0
      ],
      backgroundColor: ['#10b981', '#6b7280', '#ef4444'],
      borderWidth: 2,
      borderColor: '#ffffff'
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          padding: 15,
          usePointStyle: true,
          font: { size: 12 }
        }
      }
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">การวิเคราะห์ความรู้สึก</h2>

      <div className="grid grid-cols-3 gap-2 mb-6">
        <div className="text-center p-3 bg-green-50 rounded-lg border border-green-200">
          <div className="text-xl font-bold text-green-600">{positivePercent}%</div>
          <div className="text-xs text-green-700 mt-1">เชิงบวก</div>
          <div className="text-xs text-green-600 mt-1">{uploadData.sentiment_distribution?.positive || 0} ความคิดเห็น</div>
        </div>
        <div className="text-center p-3 bg-gray-50 rounded-lg border border-gray-200">
          <div className="text-xl font-bold text-gray-600">{neutralPercent}%</div>
          <div className="text-xs text-gray-700 mt-1">เป็นกลาง</div>
          <div className="text-xs text-gray-600 mt-1">{uploadData.sentiment_distribution?.neutral || 0} ความคิดเห็น</div>
        </div>
        <div className="text-center p-3 bg-red-50 rounded-lg border border-red-200">
          <div className="text-xl font-bold text-red-600">{negativePercent}%</div>
          <div className="text-xs text-red-700 mt-1">เชิงลบ</div>
          <div className="text-xs text-red-600 mt-1">{uploadData.sentiment_distribution?.negative || 0} ความคิดเห็น</div>
        </div>
      </div>

      <div className="h-48">
        <Doughnut data={sentimentChartData} options={chartOptions} />
      </div>
    </div>
  );
};

export const InterestChart = ({ interestTotal }) => {
  if (interestTotal.total === 0) return null;

  const interestChartData = {
    labels: ['สนใจ', 'ไม่สนใจ'],
    datasets: [{
      data: [interestTotal.interested, interestTotal.notInterested],
      backgroundColor: ['#10b981', '#ef4444'],
      borderWidth: 2,
      borderColor: '#ffffff'
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          padding: 15,
          usePointStyle: true,
          font: { size: 12 }
        }
      }
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">ความสนใจทดสอบอีกครั้ง</h2>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
          <div className="text-2xl font-bold text-green-600">
            {Math.round((interestTotal.interested / interestTotal.total) * 100)}%
          </div>
          <div className="text-sm text-green-700 mt-1">สนใจ</div>
          <div className="text-xs text-green-600 mt-1">{interestTotal.interested} คน</div>
        </div>
        <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
          <div className="text-2xl font-bold text-red-600">
            {Math.round((interestTotal.notInterested / interestTotal.total) * 100)}%
          </div>
          <div className="text-sm text-red-700 mt-1">ไม่สนใจ</div>
          <div className="text-xs text-red-600 mt-1">{interestTotal.notInterested} คน</div>
        </div>
      </div>

      <div className="h-48">
        <Doughnut data={interestChartData} options={chartOptions} />
      </div>
    </div>
  );
};

export const LikertChart = ({ uploadData }) => {
  const scaleChartData = {
    labels: Object.keys(uploadData.likert_analysis || {}),
    datasets: [{
      label: 'ค่าเฉลี่ย',
      data: Object.values(uploadData.likert_analysis || {}).map(item => item.mean || 0),
      backgroundColor: '#3b82f6',
      borderColor: '#1d4ed8',
      borderWidth: 1,
    }]
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        max: 5,
        ticks: { stepSize: 1 }
      },
      x: {
        ticks: {
          maxRotation: 45,
          font: { size: 10 }
        }
      }
    },
    plugins: {
      legend: { display: false }
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">คะแนนความพึงพอใจ (1-5)</h2>

      {Object.keys(uploadData.likert_analysis || {}).length > 0 ? (
        <>
          <div className="h-48 mb-6">
            <Bar data={scaleChartData} options={barChartOptions} />
          </div>

          <div className="space-y-3 max-h-32 overflow-y-auto">
            {Object.entries(uploadData.likert_analysis || {}).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <span className="text-sm text-gray-600 flex-1 truncate">{key}</span>
                <div className="flex-1 mx-4 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-blue-600 rounded-full transition-all duration-1000"
                    style={{ width: `${(value.mean / 5) * 100}%` }}
                  ></div>
                </div>
                <span className="text-sm font-semibold text-gray-800 w-12 text-right">
                  {value.mean?.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="h-48 flex items-center justify-center text-gray-500">
          ไม่พบข้อมูลคะแนน Likert Scale
        </div>
      )}
    </div>
  );
};