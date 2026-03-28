import { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity, Search, Stethoscope, ChevronDown,
  ChevronUp, AlertCircle, ShieldCheck, Thermometer, Info, Sparkles
} from 'lucide-react';
import './App.css';

// --- Animated Counter ---
const CountUp = ({ end, duration = 1.8 }) => {
  const [count, setCount] = useState(0);
  useEffect(() => {
    let start = 0;
    const increment = end / (duration * 60);
    const timer = setInterval(() => {
      start += increment;
      if (start >= end) {
        setCount(end);
        clearInterval(timer);
      } else {
        setCount(start);
      }
    }, 1000 / 60);
    return () => clearInterval(timer);
  }, [end, duration]);
  return <>{Math.floor(count)}</>;
};

// --- Loading Skeleton ---
const LoadingSkeleton = () => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    style={{ marginTop: '30px' }}
  >
    {[1, 2, 3].map((i) => (
      <div key={i} className="skeleton-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
          <div className="skeleton" style={{ width: '45%', height: '22px' }}></div>
          <div className="skeleton" style={{ width: '12%', height: '22px' }}></div>
        </div>
        <div className="skeleton" style={{ width: '100%', height: '6px', marginBottom: '14px' }}></div>
        <div className="skeleton" style={{ width: '75%', height: '14px' }}></div>
      </div>
    ))}
  </motion.div>
);

// --- Severity Badge ---
const SeverityBadge = ({ level }) => (
  <span className={`severity-badge severity-${level}`}>
    {level}
  </span>
);

// --- Disease Result Card ---
const DiseaseCard = ({ data, isTopResult, index }) => {
  const [expanded, setExpanded] = useState(isTopResult);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
      className={`result-card ${isTopResult ? 'top-result' : ''}`}
    >
      {/* Header */}
      <div className="result-header" onClick={() => setExpanded(!expanded)}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '4px' }}>
            <h3 className="disease-name">{data.disease}</h3>
            <SeverityBadge level={data.severity} />
          </div>

          <div className="confidence-row">
            <div className="confidence-track">
              <motion.div
                className={`confidence-fill ${isTopResult ? 'top' : 'other'}`}
                initial={{ width: 0 }}
                animate={{ width: `${data.confidence}%` }}
                transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
              />
            </div>
            <span className={`confidence-value ${isTopResult ? 'top' : 'other'}`}>
              <CountUp end={data.confidence} />%
            </span>
          </div>
        </div>

        {expanded
          ? <ChevronUp size={20} color="#8B95A5" />
          : <ChevronDown size={20} color="#8B95A5" />
        }
      </div>

      {/* Expanded Content */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            style={{ overflow: 'hidden' }}
          >
            <div className="result-body">

              {/* AI Explanation — Top Result Only */}
              {isTopResult && data.ai_explanation && (
                <div className="ai-explanation">
                  <div className="ai-explanation-header">
                    <Sparkles size={14} />
                    AI Insight
                  </div>
                  <p>{data.ai_explanation}</p>
                </div>
              )}

              {/* Precautions */}
              {data.precautions && data.precautions.length > 0 && (
                <div className="precautions-section">
                  <div className="precautions-label">Recommended Actions</div>
                  <div className="precaution-pills">
                    {data.precautions.map((action, k) => (
                      <span key={k} className="precaution-pill">
                        <ShieldCheck size={12} /> {action}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Static Description */}
              <div className="static-description">
                <Info size={14} />
                <span>{data.description}</span>
              </div>

            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};


// ============================================
// MAIN APP
// ============================================
function App() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post('http://localhost:8000/predict', { text: inputText });
      if (response.data.status === 'success') {
        setTimeout(() => {
          setResult(response.data);
          setLoading(false);
        }, 600);
      } else {
        setError(response.data.message);
        setLoading(false);
      }
    } catch (err) {
      setError("Server connection failed. Is the backend running?");
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAnalyze();
    }
  };

  return (
    <div className="app-container">

      {/* ---- Header ---- */}
      <motion.div
        className="header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="header-icon">
          <Stethoscope size={28} color="#FFFFFF" />
        </div>
        <h1>MedInsight <span>AI</span></h1>
        <p>Advanced symptom analysis powered by machine learning &amp; AI-driven medical context.</p>
      </motion.div>

      {/* ---- Input Card ---- */}
      <motion.div
        className="card"
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
      >
        <div className="card-inner">
          <label className="input-label">
            <Activity size={14} color="#C53030" />
            Describe your symptoms
          </label>
          <textarea
            id="symptom-input"
            className="input-field"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="e.g. I've had a severe rash on my legs, been throwing up, and have a high fever..."
            rows={4}
          />
          <button
            id="analyse-btn"
            onClick={handleAnalyze}
            disabled={loading}
            className="btn-analyse"
          >
            {loading ? 'Analysing…' : (
              <>
                Analyse <Search size={18} />
              </>
            )}
          </button>
        </div>
      </motion.div>

      {/* ---- Error ---- */}
      <AnimatePresence>
        {error && (
          <motion.div
            className="error-box"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <AlertCircle size={18} />
            <span>{error}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ---- Loading ---- */}
      <AnimatePresence>
        {loading && <LoadingSkeleton />}
      </AnimatePresence>

      {/* ---- Results ---- */}
      <AnimatePresence>
        {result && !loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            {/* Detected Symptoms Tag Cloud */}
            <div className="symptoms-cloud">
              <div className="symptoms-cloud-label">Symptoms Detected</div>
              <div className="symptoms-tags">
                {result.extracted_symptoms.map((sym, i) => (
                  <motion.span
                    key={i}
                    className="symptom-tag"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: i * 0.04 }}
                  >
                    <Thermometer size={12} /> {sym.replace(/_/g, ' ')}
                  </motion.span>
                ))}
              </div>
            </div>

            {/* Diagnosis Cards */}
            <div className="results-section">
              {result.predictions.map((pred, idx) => (
                <DiseaseCard
                  key={idx}
                  data={pred}
                  isTopResult={idx === 0}
                  index={idx}
                />
              ))}
            </div>

            {/* Disclaimer */}
            <motion.div
              className="disclaimer"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
            >
              <p>
                <ShieldCheck size={14} />
                <strong>Medical Disclaimer:</strong> This system uses AI to provide preliminary
                educational information. It is not a substitute for professional medical advice,
                diagnosis, or treatment. Always consult a qualified healthcare provider.
              </p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;