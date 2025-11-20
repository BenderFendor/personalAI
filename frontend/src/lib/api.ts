const API_BASE = 'http://localhost:8000/api';

export interface Session {
  id: string;
  title: string;
  started_at: string;
  file: string;
  tokens_in: number;
  tokens_out: number;
}

export async function getSessions(): Promise<Session[]> {
  const res = await fetch(`${API_BASE}/sessions`);
  if (!res.ok) throw new Error('Failed to fetch sessions');
  const data = await res.json();
  // The backend returns a list of dicts, which matches Session[]
  return data; 
}

export async function loadSession(sessionId: string) {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/load`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to load session');
  return res.json();
}

export async function clearHistory() {
    const res = await fetch(`${API_BASE}/history/clear`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to clear history');
    return res.json();
}

export async function getConfig() {
  const res = await fetch(`${API_BASE}/config`);
  if (!res.ok) throw new Error('Failed to fetch config');
  return res.json();
}

export async function newChat() {
  const res = await fetch(`${API_BASE}/chat/new`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to create new chat');
  return res.json();
}
