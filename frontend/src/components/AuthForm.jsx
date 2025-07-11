import React, { useState } from 'react';
import { setApiKey, getApiKey } from '../api';

export default function AuthForm({ onAuth }) {
  const [key, setKey] = useState(getApiKey());

  const handleSubmit = (e) => {
    e.preventDefault();
    setApiKey(key);
    onAuth(key);
  };

  return (
    <form onSubmit={handleSubmit} className="auth-form">
      <div className="form-group">
        <label htmlFor="api-key">API Key:</label>
        <input
          id="api-key"
          type="text"
          value={key}
          onChange={e => setKey(e.target.value)}
          placeholder="demo-api-key"
          className="form-input"
        />
      </div>
      <button type="submit" className="auth-button">Connect to AI Backend</button>
    </form>
  );
} 