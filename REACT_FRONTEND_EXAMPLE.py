"""
React Frontend for rPPG API (TypeScript/TSX)
==============================================

Sample component for uploading video and displaying results.
Can be integrated into a Create React App or Next.js project.
"""

# This is a Python file, so I'll provide the code as a string that should be saved as JSX/TSX

REACT_COMPONENT_CODE = """
// components/rPPGAnalysis.tsx

import React, { useState } from 'react';
import axios from 'axios';
import { AlertCircle, CheckCircle, TrendingUp, Upload } from 'lucide-react';

interface VitalsResult {
  heart_rate_bpm: number;
  heart_rate_confidence: string;
  sdnn_ms: number | null;
  pnn50_percent: number | null;
  stress_level: number | null;
  bp_systolic: number | null;
  bp_diastolic: number | null;
  bp_note: string;
  spo2: number | null;
  spo2_note: string;
  rr_interval_count: number;
}

interface RiskResult {
  risk_score: number;
  risk_level: 'LOW' | 'MODERATE' | 'HIGH';
  alerts: string[];
  recommendation: string;
}

interface AnalysisResponse {
  request_id: string;
  timestamp: string;
  vitals: VitalsResult;
  risk: RiskResult;
  analysis_time_sec: number;
  disclaimer: string;
  message: string;
}

const rPPGAnalysis: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      // Validate file type
      const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska'];
      if (!validTypes.includes(selectedFile.type)) {
        setError('Invalid file format. Please upload MP4, MOV, AVI, or MKV.');
        return;
      }
      // Validate file size (500 MB max)
      if (selectedFile.size > 500 * 1024 * 1024) {
        setError('File too large. Maximum 500 MB.');
        return;
      }
      setFile(selectedFile);
      setError(null);
      setResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post<AnalysisResponse>(
        `${API_URL}/analyze`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = progressEvent.total
              ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
              : 0;
            setProgress(percentCompleted);
          },
        }
      );
      setResult(response.data);
    } catch (err: any) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Analysis failed. Please try again.'
      );
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'LOW':
        return 'text-green-600';
      case 'MODERATE':
        return 'text-yellow-600';
      case 'HIGH':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getRiskBg = (level: string) => {
    switch (level) {
      case 'LOW':
        return 'bg-green-50';
      case 'MODERATE':
        return 'bg-yellow-50';
      case 'HIGH':
        return 'bg-red-50';
      default:
        return 'bg-gray-50';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            ❤️ Experimental Vitals Analysis
          </h1>
          <p className="text-gray-600">
            Camera-based vital signs estimation (rPPG) - Research Prototype
          </p>
        </div>

        {/* Critical Disclaimer */}
        <div className="mb-8 p-4 bg-red-50 border-l-4 border-red-500 rounded">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-red-700">
              <strong>⚠️ EXPERIMENTAL RESEARCH PROTOTYPE</strong>
              <p className="mt-1">
                This app is NOT clinically validated. Do NOT use for medical decisions.
                Blood pressure and SpO₂ estimates are uncalibrated and may have ±15% error.
                Consult healthcare professionals for medical concerns.
              </p>
            </div>
          </div>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-lg shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6">
            📹 Upload Your Video
          </h2>

          <div className="flex flex-col gap-6">
            {/* File Input */}
            <div className="border-2 border-dashed border-indigo-300 rounded-lg p-8 text-center hover:border-indigo-500 transition cursor-pointer">
              <input
                type="file"
                accept="video/mp4,.mov,video/x-msvideo,.mkv"
                onChange={handleFileSelect}
                disabled={loading}
                className="hidden"
                id="file-input"
              />
              <label htmlFor="file-input" className="cursor-pointer block">
                <Upload className="w-12 h-12 text-indigo-500 mx-auto mb-3" />
                <p className="text-lg font-medium text-gray-900">
                  {file ? file.name : 'Click to upload or drag and drop'}
                </p>
                <p className="text-sm text-gray-600 mt-1">
                  MP4, MOV, AVI, MKV (max 500 MB, 10-30 seconds recommended)
                </p>
              </label>
            </div>

            {/* Recommendations */}
            <div className="bg-blue-50 border border-blue-200 rounded p-4">
              <h3 className="font-semibold text-blue-900 mb-2">
                📋 For Best Results:
              </h3>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>✅ Well-lit environment (natural or bright indoor light)</li>
                <li>✅ Face mostly visible, centered in frame</li>
                <li>✅ 10-30 seconds of video</li>
                <li>✅ Minimal head movement and facial expressions</li>
                <li>✅ High-quality camera (smartphone is fine)</li>
              </ul>
            </div>

            {/* Analyze Button */}
            <button
              onClick={handleAnalyze}
              disabled={!file || loading}
              className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white font-bold py-3 px-6 rounded-lg transition"
            >
              {loading ? (
                <span>🔍 Analyzing ({progress}%)...</span>
              ) : (
                <span>🔍 Analyze Video</span>
              )}
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-8 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
              <div className="text-red-700">{error}</div>
            </div>
          </div>
        )}

        {/* Results Display */}
        {result && (
          <div className="space-y-8">
            {/* Success Message */}
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg flex items-center gap-3">
              <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
              <div className="text-green-800">
                ✅ Analysis complete in {result.analysis_time_sec.toFixed(1)}s
              </div>
            </div>

            {/* Main Vitals */}
            <div className="bg-white rounded-lg shadow-lg p-8">
              <h3 className="text-2xl font-semibold text-gray-900 mb-6">
                📊 Estimated Vital Signs
              </h3>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {/* Heart Rate */}
                <div className="bg-gradient-to-br from-pink-50 to-red-50 p-4 rounded-lg">
                  <p className="text-gray-600 text-sm mb-1">❤️ Heart Rate</p>
                  <p className="text-3xl font-bold text-red-600">
                    {result.vitals.heart_rate_bpm.toFixed(1)}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Confidence: {result.vitals.heart_rate_confidence}
                  </p>
                </div>

                {/* Stress Level */}
                <div className="bg-gradient-to-br from-yellow-50 to-orange-50 p-4 rounded-lg">
                  <p className="text-gray-600 text-sm mb-1">😰 Stress (0-10)</p>
                  <p className="text-3xl font-bold text-orange-600">
                    {result.vitals.stress_level?.toFixed(1) || 'N/A'}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">Experimental</p>
                </div>

                {/* HRV (SDNN) */}
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-4 rounded-lg">
                  <p className="text-gray-600 text-sm mb-1">📈 HRV (SDNN)</p>
                  <p className="text-3xl font-bold text-blue-600">
                    {result.vitals.sdnn_ms?.toFixed(0) || 'N/A'}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">ms</p>
                </div>

                {/* pNN50 */}
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-4 rounded-lg">
                  <p className="text-gray-600 text-sm mb-1">📊 pNN50</p>
                  <p className="text-3xl font-bold text-green-600">
                    {result.vitals.pnn50_percent?.toFixed(1) || 'N/A'}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">%</p>
                </div>
              </div>
            </div>

            {/* Experimental Vitals */}
            <div className="bg-white rounded-lg shadow-lg p-8">
              <h3 className="text-2xl font-semibold text-gray-900 mb-6">
                🩺 Experimental Vitals (NOT Validated)
              </h3>

              <div className="grid md:grid-cols-2 gap-6">
                {/* Blood Pressure */}
                <div className="border-l-4 border-orange-400 pl-4">
                  <h4 className="font-semibold text-gray-900 mb-2">
                    Blood Pressure (Experimental)
                  </h4>
                  {result.vitals.bp_systolic ? (
                    <>
                      <p className="text-2xl font-bold text-gray-900">
                        {result.vitals.bp_systolic.toFixed(0)}/
                        {result.vitals.bp_diastolic?.toFixed(0)} mmHg
                      </p>
                      <p className="text-xs text-gray-600 mt-2">
                        ⚠️ {result.vitals.bp_note}
                      </p>
                    </>
                  ) : (
                    <p className="text-gray-500">Unable to estimate</p>
                  )}
                </div>

                {/* SpO2 */}
                <div className="border-l-4 border-blue-400 pl-4">
                  <h4 className="font-semibold text-gray-900 mb-2">
                    SpO₂ (Oxygen Saturation)
                  </h4>
                  <p className="text-gray-500 mb-2">Not estimated</p>
                  <p className="text-xs text-gray-600">
                    ℹ️ {result.vitals.spo2_note}
                  </p>
                </div>
              </div>
            </div>

            {/* Risk Assessment */}
            <div className={`${getRiskBg(result.risk.risk_level)} rounded-lg shadow-lg p-8 border-l-4 border-r-4 border-gray-300`}>
              <h3 className="text-2xl font-semibold text-gray-900 mb-4">
                ⚠️ Risk Assessment (Experimental)
              </h3>

              <div className="flex items-center gap-4 mb-6">
                <div>
                  <p className="text-gray-600 text-sm">Risk Level</p>
                  <p className={`text-4xl font-bold ${getRiskColor(result.risk.risk_level)}`}>
                    {result.risk.risk_level}
                  </p>
                </div>
                <div className="text-3xl">
                  {result.risk.risk_level === 'LOW'
                    ? '✅'
                    : result.risk.risk_level === 'MODERATE'
                    ? '⚠️'
                    : '🔴'}
                </div>
              </div>

              <div className="bg-white bg-opacity-70 rounded p-4 mb-4">
                <p className="text-gray-900">{result.risk.recommendation}</p>
              </div>

              {result.risk.alerts.length > 0 && (
                <div className="bg-white bg-opacity-70 rounded p-4">
                  <p className="font-semibold text-gray-900 mb-2">Alerts:</p>
                  <ul className="text-sm text-gray-700 space-y-1">
                    {result.risk.alerts.map((alert, i) => (
                      <li key={i}>• {alert}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Metadata */}
            <div className="text-center text-sm text-gray-600">
              <p>Request ID: {result.request_id}</p>
              <p>Analyzed: {new Date(result.timestamp).toLocaleString()}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default rPPGAnalysis;
"""

# Sample Next.js page wrapping this component
NEXT_PAGE_EXAMPLE = """
// pages/vitals.tsx

import Head from 'next/head';
import rPPGAnalysis from '../components/rPPGAnalysis';

export default function VitalsPage() {
  return (
    <>
      <Head>
        <title>Experimental rPPG Vitals - Research Prototype</title>
        <meta name="description" content="Experimental camera-based vitals estimation" />
      </Head>
      <rPPGAnalysis />
    </>
  );
}
"""

# Next.js environment config
NEXT_ENV_EXAMPLE = """
# .env.local

REACT_APP_API_URL=http://localhost:8000
# For production:
# REACT_APP_API_URL=https://api.example.com
"""

# Tailwind CSS config (if using Tailwind, which above component assumes)
TAILWIND_CONFIG = """
// tailwind.config.js

module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
"""

print("React Frontend Component Example")
print("=" * 50)
print("\nTo integrate this:")
print("1. Create a Next.js or Create React App project")
print("2. Copy the rPPGAnalysis component above")
print("3. Install dependencies: npm install axios lucide-react")
print("4. Set REACT_APP_API_URL environment variable")
print("5. Import and use the component in a page")
print("\nThe component handles:")
print("  - File upload with validation")
print("  - API calls to FastAPI backend")
print("  - Progress tracking")
print("  - Error handling")
print("  - Results display with disclaimers")
print("  - Responsive design (mobile-friendly)")
