// Minimal API helper for MultimodalAPI backend
const BASE_URL = 'http://localhost:8000';

export function setApiKey(key) {
  localStorage.setItem('apiKey', key);
}

export function getApiKey() {
  return localStorage.getItem('apiKey') || '';
}

function authHeaders() {
  const key = getApiKey();
  return key ? { 'Authorization': `Bearer ${key}` } : {};
}

export async function healthCheck() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}

export async function analyzePolicy(data) {
  const res = await fetch(`${BASE_URL}/analyze/policy`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(data),
  });
  return res.json();
}

export async function forecastTrade(data) {
  const res = await fetch(`${BASE_URL}/forecast/trade`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(data),
  });
  return res.json();
} 