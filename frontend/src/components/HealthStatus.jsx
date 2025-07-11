import React, { useEffect, useState } from 'react';
import { healthCheck } from '../api';

export default function HealthStatus() {
  const [status, setStatus] = useState('checking');
  const [timestamp, setTimestamp] = useState('');

  useEffect(() => {
    healthCheck().then(res => {
      setStatus(res.status);
      setTimestamp(res.timestamp || '');
    }).catch(() => setStatus('error'));
  }, []);

  return (
    <div className={`health-status ${status}`}>
      <span>API Status: {status}</span>
      {timestamp && <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>{timestamp}</span>}
    </div>
  );
} 