import React, { useState } from 'react';
import { 
  Zap, 
  Settings2, 
  ShieldCheck, 
  Loader2, 
  Globe, 
  BarChart3, 
  Calendar, 
  Activity, 
  TrendingUp,
  AlertTriangle
} from 'lucide-react';

const App = () => {
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  
  // These keys MUST match the keys in your Python load_and_preprocess_data function
  const [formData, setFormData] = useState({
    "Start Year": 2024,
    "Start Month": 6,
    "Start Day": 15,
    "Region": "Asia",
    "Country": "India",
    "Subregion": "Southern Asia",
    "Latitude": 20.5937,
    "Longitude": 78.9629,
    "Magnitude": 6.5,
    "Total Affected": 10000,
    "Total Deaths": 0,
    "No. Injured": 0
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    // Handle both string and number inputs correctly for the model
    const val = e.target.type === 'number' ? parseFloat(value) : value;
    
    setFormData(prev => ({
      ...prev,
      [name]: val
    }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    setError(null);

    try {
      // Connects to your local Flask/FastAPI backend
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setPrediction(data);
      } else {
        setError(data.message || "Model Inference Failed.");
      }
    } catch (err) {
      setError("Network Error: Ensure your Python backend (app.py) is running on http://127.0.0.1:5000 and CORS is enabled.");
    } finally {
      setLoading(false);
    }
  };

  return (
    /* Outer container designed to work with your centered body CSS */
    <div className="min-h-screen w-full flex flex-col items-center justify-start py-10 px-4">
      <div className="w-full max-w-6xl">
        {/* Branding Header */}
        <header className="mb-12 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          <div className="text-left">
            <div className="flex items-center gap-3 mb-2">
              <div className="bg-blue-600 p-2 rounded-xl shadow-lg shadow-blue-500/30">
                <Activity className="w-7 h-7 text-white" />
              </div>
              <h1 className="text-4xl font-black text-slate-900 dark:text-white tracking-tight m-0 border-none">
                N-XGB <span className="text-blue-600">Sentinel</span>
              </h1>
            </div>
            <p className="text-slate-500 font-semibold text-lg ml-1">
              Hybrid Neural-XGBoost Framework • <span className="text-blue-600">97.22% Precision</span>
            </p>
          </div>
          
          <div className="flex items-center gap-3 px-5 py-2.5 rounded-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-sm text-sm font-bold text-slate-600 dark:text-slate-300">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500"></span>
            </span>
            Neural Extractor: ACTIVE
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
          {/* LEFT: INPUT FORM */}
          <div className="lg:col-span-7 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-[2rem] p-8 shadow-xl shadow-slate-200/50 dark:shadow-none">
            <div className="flex items-center gap-3 mb-10 border-b border-slate-100 dark:border-slate-700 pb-5">
              <Settings2 className="w-6 h-6 text-blue-500" />
              <h2 className="text-lg font-bold text-slate-800 dark:text-slate-200 uppercase tracking-widest">Incident Parameters</h2>
            </div>

            <form onSubmit={handlePredict} className="space-y-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                {/* Section 1: Geography */}
                <div className="space-y-5 text-left">
                  <div className="flex items-center gap-2 text-xs font-black text-slate-400 uppercase tracking-widest">
                    <Globe className="w-4 h-4" /> Geographical Context
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-[11px] font-black text-slate-500 ml-1 uppercase">Region</label>
                    <select name="Region" value={formData.Region} onChange={handleChange} className="w-full bg-slate-50 dark:bg-slate-900 border-2 border-slate-100 dark:border-slate-700 rounded-2xl p-4 focus:border-blue-500 outline-none transition-all text-base font-bold text-slate-900 dark:text-white">
                      <option>Asia</option>
                      <option>Americas</option>
                      <option>Europe</option>
                      <option>Africa</option>
                      <option>Oceania</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-[11px] font-black text-slate-500 ml-1 uppercase">Country</label>
                    <input type="text" name="Country" value={formData.Country} onChange={handleChange} className="w-full bg-slate-50 dark:bg-slate-900 border-2 border-slate-100 dark:border-slate-700 rounded-2xl p-4 focus:border-blue-500 outline-none text-base font-bold text-slate-900 dark:text-white" />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-[11px] font-black text-slate-500 ml-1 uppercase">Lat</label>
                      <input type="number" step="0.001" name="Latitude" value={formData.Latitude} onChange={handleChange} className="w-full bg-slate-50 dark:bg-slate-900 border-2 border-slate-100 dark:border-slate-700 rounded-2xl p-4 text-base font-bold outline-none text-slate-900 dark:text-white" />
                    </div>
                    <div className="space-y-2">
                      <label className="text-[11px] font-black text-slate-500 ml-1 uppercase">Lon</label>
                      <input type="number" step="0.001" name="Longitude" value={formData.Longitude} onChange={handleChange} className="w-full bg-slate-50 dark:bg-slate-900 border-2 border-slate-100 dark:border-slate-700 rounded-2xl p-4 text-base font-bold outline-none text-slate-900 dark:text-white" />
                    </div>
                  </div>
                </div>

                {/* Section 2: Impact */}
                <div className="space-y-5 text-left">
                  <div className="flex items-center gap-2 text-xs font-black text-slate-400 uppercase tracking-widest">
                    <BarChart3 className="w-4 h-4" /> Intensity Data
                  </div>

                  <div className="space-y-2">
                    <label className="text-[11px] font-black text-slate-500 ml-1 uppercase">Magnitude</label>
                    <input type="number" step="0.1" name="Magnitude" value={formData.Magnitude} onChange={handleChange} className="w-full bg-slate-50 dark:bg-slate-900 border-2 border-slate-100 dark:border-slate-700 rounded-2xl p-4 text-base font-bold outline-none text-slate-900 dark:text-white" />
                  </div>

                  <div className="space-y-2">
                    <label className="text-[11px] font-black text-slate-500 ml-1 uppercase">Affected Population</label>
                    <input type="number" name="Total Affected" value={formData["Total Affected"]} onChange={handleChange} className="w-full bg-slate-50 dark:bg-slate-900 border-2 border-slate-100 dark:border-slate-700 rounded-2xl p-4 text-base font-bold outline-none text-slate-900 dark:text-white" />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-[11px] font-black text-slate-500 ml-1 uppercase">Deaths</label>
                      <input type="number" name="Total Deaths" value={formData["Total Deaths"]} onChange={handleChange} className="w-full bg-slate-50 dark:bg-slate-900 border-2 border-slate-100 dark:border-slate-700 rounded-2xl p-4 text-base font-bold outline-none text-slate-900 dark:text-white" />
                    </div>
                    <div className="space-y-2">
                      <label className="text-[11px] font-black text-slate-500 ml-1 uppercase">Injured</label>
                      <input type="number" name="No. Injured" value={formData["No. Injured"]} onChange={handleChange} className="w-full bg-slate-50 dark:bg-slate-900 border-2 border-slate-100 dark:border-slate-700 rounded-2xl p-4 text-base font-bold outline-none text-slate-900 dark:text-white" />
                    </div>
                  </div>
                </div>
              </div>

              {/* Section 3: Time */}
              <div className="p-6 bg-blue-50/50 dark:bg-slate-900/50 rounded-3xl border-2 border-blue-100/50 dark:border-slate-700">
                <div className="flex items-center gap-2 text-xs font-black text-blue-400 uppercase tracking-widest mb-6">
                  <Calendar className="w-4 h-4" /> Temporal Dimension
                </div>
                <div className="grid grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <label className="text-[10px] text-slate-400 font-black px-1 uppercase">Year</label>
                    <input type="number" name="Start Year" value={formData["Start Year"]} onChange={handleChange} className="w-full bg-white dark:bg-slate-800 border-2 border-slate-100 dark:border-slate-700 rounded-2xl p-3 text-center text-lg font-black outline-none text-slate-900 dark:text-white focus:border-blue-500" />
                  </div>
                  <div className="space-y-2">
                    <label className="text-[10px] text-slate-400 font-black px-1 uppercase">Month</label>
                    <input type="number" name="Start Month" min="1" max="12" value={formData["Start Month"]} onChange={handleChange} className="w-full bg-white dark:bg-slate-800 border-2 border-slate-100 dark:border-slate-700 rounded-2xl p-3 text-center text-lg font-black outline-none text-slate-900 dark:text-white focus:border-blue-500" />
                  </div>
                  <div className="space-y-2">
                    <label className="text-[10px] text-slate-400 font-black px-1 uppercase">Day</label>
                    <input type="number" name="Start Day" min="1" max="31" value={formData["Start Day"]} onChange={handleChange} className="w-full bg-white dark:bg-slate-800 border-2 border-slate-100 dark:border-slate-700 rounded-2xl p-3 text-center text-lg font-black outline-none text-slate-900 dark:text-white focus:border-blue-500" />
                  </div>
                </div>
              </div>

              <button 
                type="submit" 
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white py-5 rounded-3xl font-black text-xl shadow-xl shadow-blue-200 dark:shadow-none transition-all flex items-center justify-center gap-4 disabled:opacity-50 border-none transform active:scale-[0.98]"
              >
                {loading ? <Loader2 className="w-6 h-6 animate-spin" /> : <Zap className="w-6 h-6 fill-current" />}
                {loading ? 'RUNNING N-XGB SIMULATION...' : 'EXECUTE PREDICTION'}
              </button>
            </form>
          </div>

          {/* RIGHT: RESULTS PANEL */}
          <div className="lg:col-span-5 h-full">
            <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-[2rem] p-10 shadow-xl shadow-slate-200/50 dark:shadow-none sticky top-10 flex flex-col justify-center min-h-[600px] overflow-hidden">
              
              {/* Background Decoration */}
              <div className="absolute top-0 right-0 p-10 opacity-[0.03] dark:opacity-[0.05] pointer-events-none">
                <Activity className="w-64 h-64 text-blue-600" />
              </div>

              {/* Error State */}
              {error && (
                <div className="relative z-10 flex flex-col items-center justify-center text-center space-y-6 animate-in fade-in zoom-in duration-300">
                   <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-full">
                    <AlertTriangle className="w-16 h-16 text-red-500" />
                   </div>
                   <div className="space-y-2">
                    <h3 className="text-xl font-black text-slate-900 dark:text-white uppercase tracking-tight">Backend Unreachable</h3>
                    <p className="text-red-600 font-bold text-sm px-6 leading-relaxed">{error}</p>
                   </div>
                   <button onClick={() => setError(null)} className="px-8 py-3 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 rounded-2xl text-xs font-black transition-colors border-none text-slate-600 dark:text-slate-300 uppercase tracking-widest">Try Again</button>
                </div>
              )}

              {/* Idle State */}
              {!prediction && !error && (
                <div className="relative z-10 flex flex-col items-center justify-center text-center p-10 space-y-8">
                  <div className="w-32 h-32 bg-slate-50 dark:bg-slate-900 rounded-[2.5rem] flex items-center justify-center rotate-3 border-2 border-slate-100 dark:border-slate-800">
                    <TrendingUp className="w-16 h-16 text-slate-200 dark:text-slate-700" />
                  </div>
                  <div className="space-y-3">
                    <p className="text-blue-600 font-black uppercase text-xs tracking-[0.3em]">System Standby</p>
                    <p className="text-slate-400 text-base font-medium max-w-[280px] leading-relaxed">
                      Deep Feature Extractor initialized. Awaiting multidimensional vector inputs for hybrid classification.
                    </p>
                  </div>
                </div>
              )}

              {/* Prediction Result */}
              {prediction && !error && (
                <div className="relative z-10 animate-in fade-in slide-in-from-bottom-12 duration-1000 w-full text-left">
                  <div className="flex flex-col gap-6 mb-12">
                    <div className="space-y-1">
                      <span className="text-xs font-black tracking-[0.3em] text-blue-600 uppercase">Analysis Outcome</span>
                      <h2 className="text-6xl font-black text-slate-900 dark:text-white tracking-tighter leading-tight m-0 border-none">
                        {prediction.prediction}
                      </h2>
                    </div>
                    <div className={`self-start px-6 py-2 rounded-2xl text-xs font-black uppercase tracking-[0.2em] border-2 ${prediction.risk_analysis === 'High' ? 'bg-orange-50 text-orange-600 border-orange-100 dark:bg-orange-900/20 dark:border-orange-900/30' : 'bg-green-50 text-green-600 border-green-100 dark:bg-green-900/20 dark:border-green-900/30'}`}>
                      {prediction.risk_analysis} Risk Level
                    </div>
                  </div>

                  <div className="space-y-12">
                    <div className="space-y-4">
                      <div className="flex justify-between items-end">
                        <span className="text-xs font-black text-slate-400 uppercase tracking-widest">Model Confidence</span>
                        <span className="text-3xl font-black text-blue-600 font-mono">{prediction.confidence}</span>
                      </div>
                      <div className="h-4 w-full bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden p-1 border border-slate-50 dark:border-slate-800">
                        <div 
                          className="h-full bg-blue-600 rounded-full transition-all duration-[1.5s] ease-out shadow-lg shadow-blue-500/50"
                          style={{ width: prediction.confidence }}
                        ></div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-5">
                      <div className="bg-slate-50 dark:bg-slate-900 p-6 rounded-3xl border-2 border-slate-100 dark:border-slate-800">
                        <p className="text-[10px] text-slate-400 font-black uppercase tracking-widest mb-1">Embedding</p>
                        <p className="text-sm font-black text-slate-800 dark:text-slate-200">32D Residual</p>
                      </div>
                      <div className="bg-slate-50 dark:bg-slate-900 p-6 rounded-3xl border-2 border-slate-100 dark:border-slate-800">
                        <p className="text-[10px] text-slate-400 font-black uppercase tracking-widest mb-1">Architecture</p>
                        <p className="text-sm font-black text-blue-600">N-XGB Hybrid</p>
                      </div>
                    </div>

                    <div className="mt-10 pt-10 border-t-2 border-slate-100 dark:border-slate-700">
                      <div className="flex items-center gap-3 text-xs font-black text-slate-400 uppercase tracking-widest mb-4">
                        <ShieldCheck className="w-5 h-5 text-green-500" /> Model Insights
                      </div>
                      <p className="text-sm text-slate-500 dark:text-slate-400 font-medium leading-relaxed">
                        Input features were processed via <span className="text-slate-900 dark:text-white font-bold">Residual Block Neural Encoding</span> to extract non-linear dependencies before final decision mapping by the <span className="text-blue-600 font-bold">XGBoost Optimized Classifier</span>.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;