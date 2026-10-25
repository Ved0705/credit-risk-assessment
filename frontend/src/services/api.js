// Thin wrapper around the FastAPI backend. All errors become readable messages.
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export class ApiError extends Error {
  constructor(message, details = []) {
    super(message)
    this.details = details
  }
}

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(`${API_URL}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
  } catch {
    throw new ApiError(`Cannot reach the backend at ${API_URL}. Is the FastAPI server running?`)
  }
  let body = null
  try {
    body = await res.json()
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) {
    if (res.status === 422 && Array.isArray(body?.detail)) {
      throw new ApiError('Some inputs are invalid.', body.detail)
    }
    if (res.status === 503) {
      throw new ApiError('The model is not available. Train it first: python -m app.ml.train')
    }
    throw new ApiError(typeof body?.detail === 'string' ? body.detail : `Request failed (${res.status}).`)
  }
  return body
}

const post = (path, data) => request(path, { method: 'POST', body: JSON.stringify(data) })

export const api = {
  health: () => request('/health'),
  features: () => request('/model/features'),
  metrics: () => request('/model/metrics'),
  importance: () => request('/model/importance'),
  fairness: () => request('/fairness'),
  stats: () => request('/assessments/stats'),
  recent: (limit = 10) => request(`/assessments?limit=${limit}`),
  predict: (applicant) => post('/predict', applicant),
  explain: (applicant) => post('/explain', applicant),
  stability: (applicant, n_samples = 20, noise_level = 0.05) =>
    post('/stability', { applicant, n_samples, noise_level }),
}
