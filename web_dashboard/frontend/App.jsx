import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

// Connect to WebSocket
const socket = io('http://localhost:5000');

function App() {
  const [trainingStatus, setTrainingStatus] = useState({
    is_training: false,
    current_episode: 0,
    total_episodes: 0,
    current_reward: 0,
    best_reward: 0
  });
  
  const [metricsData, setMetricsData] = useState([]);
  const [models, setModels] = useState([]);
  const [darkMode, setDarkMode] = useState(true);

  useEffect(() => {
    // Subscribe to live metrics
    socket.on('connected', (data) => {
      console.log('Connected to server:', data);
    });

    socket.on('metrics_update', (data) => {
      setMetricsData(prev => [...prev, data].slice(-100));
      setTrainingStatus(prev => ({
        ...prev,
        current_episode: data.episode,
        current_reward: data.reward
      }));
    });

    // Fetch initial data
    fetchModels();
    fetchTrainingStatus();

    return () => {
      socket.off('connected');
      socket.off('metrics_update');
    };
  }, []);

  const fetchTrainingStatus = async () => {
    const response = await fetch('http://localhost:5000/api/training/status');
    const data = await response.json();
    setTrainingStatus(data);
  };

  const fetchModels = async () => {
    const response = await fetch('http://localhost:5000/api/models');
    const data = await response.json();
    setModels(data.models || []);
  };

  const startTraining = async () => {
    const config = {
      episodes: 1000,
      learning_rate: 0.0005,
      batch_size: 128
    };

    const response = await fetch('http://localhost:5000/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });

    if (response.ok) {
      socket.emit('subscribe_metrics');
      fetchTrainingStatus();
    }
  };

  const stopTraining = async () => {
    await fetch('http://localhost:5000/api/training/stop', { method: 'POST' });
    fetchTrainingStatus();
  };

  return (
    <div className={`app ${darkMode ? 'dark' : 'light'}`}>
      {/* Header */}
      <header className="header">
        <h1>🤖 Robot Navigation - AI Training Dashboard</h1>
        <button onClick={() => setDarkMode(!darkMode)}>
          {darkMode ? '☀️' : '🌙'}
        </button>
      </header>

      {/* Main Content */}
      <div className="main-content">
        {/* Control Panel */}
        <div className="control-panel">
          <h2>🎮 Training Control</h2>
          
          <div className="status-card">
            <h3>Status</h3>
            <p>
              {trainingStatus.is_training ? '🟢 Training' : '🔴 Idle'}
            </p>
          </div>

          <div className="metrics-card">
            <h3>📊 Current Metrics</h3>
            <p>Episode: {trainingStatus.current_episode} / {trainingStatus.total_episodes}</p>
            <p>Reward: {trainingStatus.current_reward.toFixed(2)}</p>
            <p>Best Reward: {trainingStatus.best_reward.toFixed(2)}</p>
          </div>

          <div className="button-group">
            {!trainingStatus.is_training ? (
              <button className="btn-primary" onClick={startTraining}>
                ▶️ Start Training
              </button>
            ) : (
              <button className="btn-danger" onClick={stopTraining}>
                ⏹️ Stop Training
              </button>
            )}
          </div>
        </div>

        {/* Visualization */}
        <div className="visualization">
          <h2>📈 Live Metrics</h2>
          
          {/* Rewards Chart */}
          <div className="chart-container">
            <h3>Rewards Over Time</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={metricsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="reward" 
                  stroke="#0d7377" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Steps Chart */}
          <div className="chart-container">
            <h3>Steps Per Episode</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={metricsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip />
                <Area 
                  type="monotone" 
                  dataKey="steps" 
                  stroke="#ff9500" 
                  fill="#ff9500" 
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Models Panel */}
        <div className="models-panel">
          <h2>💾 Saved Models</h2>
          
          <div className="models-list">
            {models.map(model => (
              <div key={model.id} className="model-card">
                <h3>{model.name}</h3>
                <p>Size: {(model.size / 1024 / 1024).toFixed(2)} MB</p>
                <p>Modified: {new Date(model.modified).toLocaleDateString()}</p>
                <button className="btn-small">📂 Load</button>
                <button className="btn-small btn-danger">🗑️ Delete</button>
              </div>
            ))}
          </div>
        </div>
      </div>

      <style jsx>{`
        .app {
          min-height: 100vh;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        }

        .app.dark {
          background-color: #1e1e1e;
          color: #ffffff;
        }

        .app.light {
          background-color: #ffffff;
          color: #000000;
        }

        .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px;
          background-color: #2d2d2d;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .main-content {
          display: grid;
          grid-template-columns: 300px 1fr 300px;
          gap: 20px;
          padding: 20px;
        }

        .control-panel, .visualization, .models-panel {
          background-color: #2d2d2d;
          padding: 20px;
          border-radius: 10px;
        }

        .status-card, .metrics-card {
          background-color: #3d3d3d;
          padding: 15px;
          border-radius: 8px;
          margin-bottom: 15px;
        }

        .chart-container {
          margin-bottom: 30px;
        }

        .btn-primary {
          background-color: #0d7377;
          color: white;
          border: none;
          padding: 12px 24px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 16px;
          width: 100%;
        }

        .btn-primary:hover {
          background-color: #14a085;
        }

        .btn-danger {
          background-color: #dc3545;
        }

        .btn-danger:hover {
          background-color: #c82333;
        }

        .models-list {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .model-card {
          background-color: #3d3d3d;
          padding: 15px;
          border-radius: 8px;
        }

        .btn-small {
          background-color: #0d7377;
          color: white;
          border: none;
          padding: 6px 12px;
          border-radius: 4px;
          cursor: pointer;
          margin-right: 5px;
          font-size: 12px;
        }
      `}</style>
    </div>
  );
}

export default App;
