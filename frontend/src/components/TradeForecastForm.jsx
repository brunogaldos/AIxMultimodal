import React, { useState } from 'react';
import { forecastTrade } from '../api';

const defaultData = {
  data: {
    time_series: [
      { timestamp: '2025-01-01T00:00:00Z', value: 100.0, metric: 'commodity_price' }
    ]
  },
  forecast_type: 'trading_strategy',
  parameters: { asset_class: 'commodities', time_horizon: '1m', risk_tolerance: 'medium', explainability_level: 'detailed' }
};

export default function TradeForecastForm() {
  const [input, setInput] = useState(JSON.stringify(defaultData, null, 2));
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const data = JSON.parse(input);
      const res = await forecastTrade(data);
      setResult(res);
    } catch (err) {
      setResult({ error: 'Invalid input or server error.' });
    }
    setLoading(false);
  };

  return (
    <div style={{ marginBottom: 32 }}>
      <h3>Trade Forecast</h3>
      <form onSubmit={handleSubmit}>
        <textarea
          rows={8}
          cols={60}
          value={input}
          onChange={e => setInput(e.target.value)}
        />
        <br />
        <button type="submit" disabled={loading}>{loading ? 'Forecasting...' : 'Forecast'}</button>
      </form>
      {result && (
        <pre style={{ background: '#f4f4f4', padding: 10, marginTop: 10 }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
} 