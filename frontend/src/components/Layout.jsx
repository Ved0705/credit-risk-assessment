import { Suspense, useCallback, useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { API_URL, api } from '../services/api'
import { Loading } from './ui'

const NAV = [
  ['/', 'Dashboard', 'M3 3h7v7H3zM14 3h7v4h-7zM14 10h7v11h-7zM3 13h7v8H3z'],
  ['/assess', 'Credit Assessment', 'M4 20h16M6 16l9-9 3 3-9 9H6z'],
  ['/explain', 'Explainability', 'M4 6h10M4 12h16M4 18h7'],
  ['/fairness', 'Fairness', 'M12 3v18M5 7h14M5 7l-3 7h6zM19 7l-3 7h6z'],
]

function Icon({ d }) {
  return (
    <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" aria-hidden="true">
      <path d={d} />
    </svg>
  )
}

/** Polls /health so the UI notices when the backend starts or stops. */
function useHealth(intervalMs = 10000) {
  const [health, setHealth] = useState({ status: 'checking' })
  const check = useCallback(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: 'offline' }))
  }, [])
  useEffect(() => {
    check()
    const t = setInterval(check, intervalMs)
    return () => clearInterval(t)
  }, [check, intervalMs])
  return [health, check]
}

export default function Layout() {
  const [health, recheck] = useHealth()
  const offline = health.status === 'offline'
  const degraded = health.status === 'degraded'
  const label = { ok: 'API online', degraded: 'Model not loaded', offline: 'API offline', checking: 'Checking API…' }[health.status]

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">CL</span>
          <div>
            <b>CreditLens</b>
            <small>Explainable credit risk</small>
          </div>
        </div>
        <nav>
          {NAV.map(([to, text, icon]) => (
            <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => (isActive ? 'active' : '')}>
              <Icon d={icon} /> <span>{text}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className={`health ${health.status}`}>
            <i /> {label}
          </div>
          {health.model_name && <small>Model: {health.model_name}</small>}
          <small className="disclaimer">Academic demo — not for real lending decisions.</small>
        </div>
      </aside>
      <main className="content">
        {(offline || degraded) && (
          <div className="banner" role="alert">
            <div>
              <b>{offline ? `Backend not reachable at ${API_URL}` : 'The backend is running but the model is not loaded'}</b>
              <span>
                {offline ? (
                  <>Start it from the <code>backend</code> folder: <code>uvicorn app.main:app --reload --port 8000</code></>
                ) : (
                  <>Train it from the <code>backend</code> folder: <code>python -m app.ml.train</code>, then restart the server.</>
                )}
              </span>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={recheck}>Check again</button>
          </div>
        )}
        <Suspense fallback={<Loading />}>
          <Outlet />
        </Suspense>
      </main>
    </div>
  )
}
