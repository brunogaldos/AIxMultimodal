import React, { useState, useRef, useEffect } from 'react';
import AuthForm from './components/AuthForm';
import HealthStatus from './components/HealthStatus';
import { getApiKey } from './api';
import './App.css';

function App() {
  const [authed, setAuthed] = useState(!!getApiKey());
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [analysisType, setAnalysisType] = useState('policy_analysis');
  const [timeSeriesData, setTimeSeriesData] = useState([]); // <-- new state
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Updated file handler to parse JSON for time series
  const handleFileSelect = async (event) => {
    const files = Array.from(event.target.files);
    for (const file of files) {
      if (file.type === 'application/json') {
        const text = await file.text();
        try {
          const json = JSON.parse(text);
          if (Array.isArray(json) && json[0]?.timestamp && json[0]?.value && json[0]?.metric) {
            setTimeSeriesData(json);
          }
        } catch (e) {
          // ignore parse error
        }
      }
    }
    setSelectedFiles(prev => [...prev, ...files]);
  };

  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const sendMessage = async () => {
    if (!inputText.trim() && selectedFiles.length === 0 && timeSeriesData.length === 0) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: inputText,
      files: selectedFiles,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setSelectedFiles([]);
    setIsLoading(true);

    try {
      // Prepare multimodal data
      const multimodalData = {
        data: {
          text: inputText ? [inputText] : [],
          images: [],
          time_series: timeSeriesData,
          geospatial: []
        },
        parameters: {
          time_horizon: '1y',
          explainability_level: 'detailed'
        }
      };

      // Process files (images only)
      for (const file of selectedFiles) {
        if (file.type.startsWith('image/')) {
          const reader = new FileReader();
          reader.onload = (e) => {
            const base64 = e.target.result.split(',')[1];
            multimodalData.data.images.push({
              url: `data:${file.type};base64,${base64}`,
              mime_type: file.type,
              description: file.name
            });
          };
          reader.readAsDataURL(file);
        }
      }

      // Determine endpoint and request structure based on analysis type
      let endpoint, requestData;
      if (analysisType === 'trade_forecast') {
        endpoint = '/forecast/trade';
        requestData = {
          ...multimodalData,
          forecast_type: 'trading_strategy'
        };
      } else {
        endpoint = '/analyze/policy';
        requestData = {
          ...multimodalData,
          analysis_type: 'impact_assessment'
        };
      }

      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${getApiKey()}`
        },
        body: JSON.stringify(requestData)
      });

      const result = await response.json();

      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        text: result.results?.insights?.[0] || result.results?.strategies?.[0]?.action || 'Analysis completed successfully.',
        data: result,
        timestamp: new Date().toLocaleTimeString()
      };

      setMessages(prev => [...prev, aiMessage]);
      setTimeSeriesData([]); // clear after send
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        text: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  if (!authed) {
    return (
      <div className="app">
        <div className="auth-container">
          <h1>🤖 Multimodal AI Assistant</h1>
          <p>Connect to your AI backend to start chatting</p>
          <AuthForm onAuth={() => setAuthed(true)} />
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 Multimodal AI Assistant</h1>
        <HealthStatus />
      </header>

      <div className="chat-container">
        <div className="sidebar">
          <div className="analysis-type-selector">
            <h3>Analysis Type</h3>
            <select 
              value={analysisType} 
              onChange={(e) => setAnalysisType(e.target.value)}
            >
              <option value="policy_analysis">Policy Analysis</option>
              <option value="trade_forecast">Trade Forecast</option>
            </select>
          </div>

          <div className="file-upload">
            <h3>Upload Files</h3>
            <input
              type="file"
              multiple
              accept="image/*,audio/*,video/*,.csv,.json"
              onChange={handleFileSelect}
              className="file-input"
            />
            <div className="file-list">
              {selectedFiles.map((file, index) => (
                <div key={index} className="file-item">
                  <span>{file.name}</span>
                  <button onClick={() => removeFile(index)}>×</button>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="chat-main">
          <div className="messages-container">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <h2>Welcome to Multimodal AI Assistant!</h2>
                <p>You can:</p>
                <ul>
                  <li>Ask questions about policy analysis</li>
                  <li>Upload images, audio, or data files</li>
                  <li>Get trade forecasts and insights</li>
                  <li>Analyze multimodal data</li>
                </ul>
                <p>Start by typing a message or uploading a file!</p>
              </div>
            ) : (
              messages.map((message) => (
                <div key={message.id} className={`message ${message.type}`}>
                  <div className="message-header">
                    <span className="message-author">
                      {message.type === 'user' ? 'You' : 'AI Assistant'}
                    </span>
                    <span className="message-time">{message.timestamp}</span>
                  </div>
                  <div className="message-content">
                    <p>{message.text}</p>
                    {message.files && message.files.length > 0 && (
                      <div className="message-files">
                        {message.files.map((file, index) => (
                          <div key={index} className="file-preview">
                            {file.type.startsWith('image/') ? (
                              <img src={URL.createObjectURL(file)} alt={file.name} />
                            ) : (
                              <span>{file.name}</span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                    {message.data && (
                      <details className="message-details">
                        <summary>View Analysis Details</summary>
                        <pre>{JSON.stringify(message.data, null, 2)}</pre>
                      </details>
                    )}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="message ai">
                <div className="message-content">
                  <div className="loading">AI is thinking...</div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-container">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here... (Press Enter to send, Shift+Enter for new line)"
              rows={3}
            />
            <button 
              onClick={sendMessage} 
              disabled={isLoading || (!inputText.trim() && selectedFiles.length === 0 && timeSeriesData.length === 0)}
              className="send-button"
            >
              {isLoading ? 'Sending...' : 'Send'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
