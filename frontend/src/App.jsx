import { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, Search, BookOpen, Stethoscope, ChevronDown, 
  ChevronUp, AlertCircle, ShieldCheck, Thermometer, Info 
} from 'lucide-react';

// --- ANIMATED NUMBER COMPONENT ---
const CountUp = ({ end, duration = 2 }) => {
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

// --- LOADING SKELETON COMPONENT ---
const LoadingSkeleton = () => (
  <motion.div 
    initial={{ opacity: 0 }} 
    animate={{ opacity: 1 }} 
    exit={{ opacity: 0 }}
    style={{ marginTop: '30px' }}
  >
    {/* Simulate 3 cards loading */}
    {[1, 2, 3].map((i) => (
      <div key={i} className="glass-card" style={{ padding: '20px', marginBottom: '15px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
          <div className="skeleton" style={{ width: '40%', height: '24px' }}></div>
          <div className="skeleton" style={{ width: '15%', height: '24px' }}></div>
        </div>
        <div className="skeleton" style={{ width: '100%', height: '8px', marginBottom: '15px' }}></div>
        <div className="skeleton" style={{ width: '80%', height: '16px' }}></div>
      </div>
    ))}
  </motion.div>
);

// --- RESULT CARD COMPONENT ---
const DiseaseCard = ({ data, isTopResult }) => {
  const [expanded, setExpanded] = useState(isTopResult);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="glass-card"
      style={{ 
        marginBottom: '16px', 
        overflow: 'hidden',
        borderLeft: isTopResult ? '4px solid #3b82f6' : '4px solid #64748b',
        boxShadow: isTopResult ? '0 10px 40px -10px rgba(59, 130, 246, 0.2)' : 'none'
      }}
    >
      {/* Header */}
      <div 
        onClick={() => setExpanded(!expanded)}
        style={{ padding: '20px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
      >
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h3 style={{ margin: 0, fontSize: '18px', color: '#f8fafc', fontWeight: '600' }}>
              {data.disease}
            </h3>
            {isTopResult && (
              <span style={{ background: 'rgba(59, 130, 246, 0.2)', color: '#60a5fa', fontSize: '11px', padding: '2px 8px', borderRadius: '10px', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                Most Likely
              </span>
            )}
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '8px' }}>
            {/* Custom Progress Bar */}
            <div style={{ width: '100%', maxWidth: '200px', height: '6px', background: '#334155', borderRadius: '3px', overflow: 'hidden' }}>
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${data.confidence}%` }}
                transition={{ duration: 1.5, ease: "easeOut" }}
                style={{ height: '100%', background: isTopResult ? '#3b82f6' : '#94a3b8', borderRadius: '3px' }}
              />
            </div>
            <span style={{ fontSize: '14px', color: isTopResult ? '#60a5fa' : '#94a3b8', fontWeight: 'bold' }}>
              <CountUp end={data.confidence} />%
            </span>
          </div>
        </div>
        {expanded ? <ChevronUp size={20} color="#64748b" /> : <ChevronDown size={20} color="#64748b" />}
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
            <div style={{ padding: '0 20px 24px 20px', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
              
              {/* Precautions (Pills) */}
              {data.precautions && data.precautions.length > 0 && (
                <div style={{ marginTop: '16px' }}>
                  <span style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    Recommended Actions
                  </span>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '8px' }}>
                    {data.precautions.map((action, k) => (
                      <span key={k} style={{ 
                        background: 'rgba(16, 185, 129, 0.1)', color: '#34d399', 
                        padding: '6px 12px', borderRadius: '8px', fontSize: '13px', 
                        border: '1px solid rgba(16, 185, 129, 0.2)', display: 'flex', alignItems: 'center', gap: '5px'
                      }}>
                        <ShieldCheck size={12} /> {action}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Web Context (Insight Card) */}
              <div style={{ marginTop: '20px', background: 'linear-gradient(145deg, rgba(30,41,59,0.5), rgba(15,23,42,0.5))', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <h4 style={{ margin: '0 0 8px 0', fontSize: '13px', color: '#60a5fa', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <BookOpen size={14} /> 
                  {data.web_context.includes("Wikipedia") ? "Wikipedia Summary" : "AI Web Insight"}
                </h4>
                <p style={{ margin: 0, fontSize: '14px', lineHeight: '1.6', color: '#cbd5e1' }}>
                  {data.web_context || "No live web context available."}
                </p>
              </div>

              {/* Static Description */}
              <div style={{ marginTop: '16px', fontSize: '13px', color: '#64748b', display: 'flex', gap: '8px', fontStyle: 'italic' }}>
                <Info size={14} style={{ minWidth: '14px', marginTop: '2px' }} />
                <span>{data.description}</span>
              </div>

            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

// --- MAIN APP ---
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
        // Minimal delay to show off the skeleton animation (optional)
        setTimeout(() => {
            setResult(response.data);
            setLoading(false);
        }, 800);
      } else {
        setError(response.data.message);
        setLoading(false);
      }
    } catch (err) {
      setError("Server connection failed. Is the backend running?");
      setLoading(false);
    }
  };

  return (
    // ADDED: margin: '0 auto' is the key fix here
    <div style={{ 
      width: '100%', 
      maxWidth: '800px', 
      margin: '0 auto',  // <--- THIS CENTERS IT
      padding: '40px 20px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center' // Ensures children are centered too
    }}>
      
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }} 
        animate={{ opacity: 1, y: 0 }}
        style={{ textAlign: 'center', marginBottom: '40px', width: '100%' }}
      >
        <div style={{ 
          background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.1))', 
          width: '64px', height: '64px', borderRadius: '20px', 
          display: 'flex', alignItems: 'center', justifyContent: 'center', 
          margin: '0 auto 20px auto', border: '1px solid rgba(59, 130, 246, 0.3)'
        }}>
          <Stethoscope size={32} color="#60a5fa" />
        </div>
        <h1 style={{ margin: '0 0 10px 0', fontSize: '36px', fontWeight: '800', letterSpacing: '-1px' }}>
          MedInsight <span style={{ color: '#3b82f6' }}>AI</span>
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '16px', maxWidth: '400px', margin: '0 auto' }}>
          Advanced Symptom Analysis powered by Machine Learning & Real-time Web Data.
        </p>
      </motion.div>

      {/* Input Section */}
      <motion.div 
        className="glass-card"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
        style={{ padding: '6px', width: '100%' }} // Ensure full width within container
      >
        <div style={{ background: '#0f172a', borderRadius: '14px', padding: '20px' }}>
          <label style={{ display: 'block', marginBottom: '12px', fontSize: '14px', color: '#cbd5e1', fontWeight: '500', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={16} color="#3b82f6" /> Describe your symptoms
          </label>
          <textarea
            className="input-field"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="e.g. I have a severe rash on my legs, vomiting, and a high fever..."
            rows={4}
          />
          <div style={{ marginTop: '16px' }}>
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="btn-primary"
            >
              {loading ? 'Processing...' : (
                <>
                   Analyze Symptoms <Search size={18} />
                </>
              )}
            </button>
          </div>
        </div>
      </motion.div>

      {/* Error Message */}
      <AnimatePresence>
        {error && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }} 
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            style={{ 
              marginTop: '20px', padding: '16px', background: 'rgba(239, 68, 68, 0.1)', 
              color: '#fca5a5', borderRadius: '12px', display: 'flex', gap: '12px', alignItems: 'center',
              border: '1px solid rgba(239, 68, 68, 0.2)', width: '100%'
            }}
          >
            <AlertCircle size={20} /> 
            <span style={{ fontSize: '14px' }}>{error}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading State */}
      <AnimatePresence>
        {loading && (
          <div style={{ width: '100%' }}>
            <LoadingSkeleton />
          </div>
        )}
      </AnimatePresence>

      {/* Results Section */}
      <AnimatePresence>
        {result && !loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ marginTop: '40px', width: '100%' }}>
            
            {/* Detected Entities */}
            <div style={{ marginBottom: '25px', textAlign: 'center' }}>
              <span style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 'bold' }}>
                Symptoms Detected
              </span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', justifyContent: 'center', marginTop: '10px' }}>
                {result.extracted_symptoms.map((sym, i) => (
                  <motion.span 
                    key={i}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: i * 0.05 }}
                    style={{ 
                      background: '#1e293b', color: '#e2e8f0', padding: '8px 16px', 
                      borderRadius: '20px', fontSize: '13px', border: '1px solid #334155',
                      display: 'flex', alignItems: 'center', gap: '6px'
                    }}
                  >
                    <Thermometer size={12} color="#3b82f6" /> {sym.replace('_', ' ')}
                  </motion.span>
                ))}
              </div>
            </div>

            {/* Diagnosis Cards */}
            {result.predictions.map((pred, idx) => (
              <DiseaseCard key={idx} data={pred} isTopResult={idx === 0} />
            ))}

            {/* Disclaimer Footer */}
            <motion.div 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }} 
              transition={{ delay: 1 }}
              style={{ 
                marginTop: '40px', padding: '20px', textAlign: 'center', 
                borderTop: '1px solid #1e293b' 
              }}
            >
              <p style={{ fontSize: '12px', color: '#475569', maxWidth: '400px', margin: '0 auto', lineHeight: '1.5' }}>
                <ShieldCheck size={16} style={{ marginBottom: '-3px', marginRight: '5px' }} />
                <strong>Medical Disclaimer:</strong> This system uses AI to provide preliminary educational information. It is not a substitute for professional medical advice, diagnosis, or treatment.
              </p>
            </motion.div>

          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;