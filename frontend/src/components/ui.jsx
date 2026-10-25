// Small shared UI building blocks (components only; helpers live in src/lib)
export function Card({ title, subtitle, children, actions, className = '' }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-head">
          <div>
            {title && <h2>{title}</h2>}
            {subtitle && <p className="muted">{subtitle}</p>}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

export function StatCard({ label, value, hint, tone }) {
  return (
    <div className={`stat ${tone ? `tone-${tone}` : ''}`}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {hint && <span className="stat-hint">{hint}</span>}
    </div>
  )
}

export function Badge({ children, tone = 'neutral' }) {
  return <span className={`badge tone-${tone}`}>{children}</span>
}

export function ErrorBox({ error, onRetry }) {
  if (!error) return null
  return (
    <div className="error" role="alert">
      <div>
        <strong>{error.message}</strong>
        {error.details?.length > 0 && (
          <ul>
            {error.details.map((d, i) => (
              <li key={i}>
                <code>{d.field || 'input'}</code>: {d.message}
              </li>
            ))}
          </ul>
        )}
      </div>
      {onRetry && (
        <button className="btn btn-ghost btn-sm" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  )
}

export function Loading({ text = 'Loading…' }) {
  return (
    <div className="loading" role="status">
      <span className="spinner" /> {text}
    </div>
  )
}

export function Callout({ children, tone = 'info' }) {
  return <div className={`callout tone-${tone}`}>{children}</div>
}

export function EmptyState({ title, children, action }) {
  return (
    <div className="empty">
      <b>{title}</b>
      {children && <p className="muted">{children}</p>}
      {action}
    </div>
  )
}
