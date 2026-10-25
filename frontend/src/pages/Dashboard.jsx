import { Link } from 'react-router-dom'
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import { Badge, Card, EmptyState, ErrorBox, Loading, StatCard } from '../components/ui'
import { pct, prettyDecision, statusTone } from '../lib/format'
import { useApi } from '../lib/useApi'
import { useTitle } from '../lib/useTitle'
import { api } from '../services/api'

const ATTR_NAMES = { sex: 'Sex', age_group: 'Age group', foreign_worker_group: 'Foreign worker' }

export default function Dashboard() {
  useTitle('Dashboard')
  const stats = useApi(() => api.stats(), [])
  const metrics = useApi(() => api.metrics(), [])
  const fair = useApi(() => api.fairness(), [])
  const recent = useApi(() => api.recent(8), [])
  const features = useApi(() => api.features(), [])

  const s = stats.data
  const m = metrics.data
  const labels = features.data?.labels || {}
  const hasData = s && s.total_assessments > 0
  const pie = hasData ? [
    { name: 'Low', value: s.by_category.LOW, color: '#1f9d6b' },
    { name: 'Medium', value: s.by_category.MEDIUM, color: '#e3a008' },
    { name: 'High', value: s.by_category.HIGH, color: '#d6453d' },
  ].filter((x) => x.value > 0) : []

  return (
    <>
      <header className="page-head row">
        <div>
          <h1>Dashboard</h1>
          <p className="muted">Assessments made in this session, model quality and the fairness audit.</p>
        </div>
        <Link to="/assess" className="btn btn-primary">+ New assessment</Link>
      </header>
      <ErrorBox error={stats.error || metrics.error} onRetry={() => { stats.reload(); metrics.reload(); fair.reload(); recent.reload() }} />

      <h3 className="section-title">This session</h3>
      <div className="stat-row">
        <StatCard label="Assessments" value={s ? s.total_assessments : '–'} hint="since the backend started" />
        <StatCard label="Low risk" value={hasData ? `${s.low_risk} · ${pct(s.low_risk_pct)}` : '–'} tone={hasData ? 'good' : undefined} />
        <StatCard label="High risk" value={hasData ? `${s.high_risk} · ${pct(s.high_risk_pct)}` : '–'} tone={hasData ? 'bad' : undefined} />
        <StatCard label="Needs review" value={hasData ? s.by_category.MEDIUM : '–'} hint="borderline (medium band)" tone={hasData ? 'warn' : undefined} />
      </div>

      <h3 className="section-title">Model (held-out test set, 200 applicants)</h3>
      <div className="stat-row">
        <StatCard label="Model" value={m ? m.selected_model : '–'} hint={m ? `threshold ${pct(m.decision_threshold)}` : ''} />
        <StatCard label="ROC-AUC" value={m ? m.selected_metrics.roc_auc.toFixed(3) : '–'} hint="ranking quality (0.5 = random)" />
        <StatCard label="Recall" value={m ? pct(m.selected_metrics.recall, 1) : '–'} hint="defaulters caught" />
        <StatCard label="Accuracy" value={m ? pct(m.selected_metrics.accuracy, 1) : '–'} />
      </div>

      <div className="three-col">
        <Card title="Risk mix of assessments">
          {stats.loading ? <Loading /> : !hasData ? (
            <EmptyState title="No assessments yet" action={<Link className="btn btn-ghost btn-sm" to="/assess">Assess an applicant</Link>}>
              History is kept in memory and resets when the backend restarts.
            </EmptyState>
          ) : (
            <ResponsiveContainer width="100%" height={210}>
              <PieChart>
                <Pie isAnimationActive={false} data={pie} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80} label={(e) => `${e.name}: ${e.value}`}>
                  {pie.map((p) => <Cell key={p.name} fill={p.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Fairness summary" actions={<Link className="btn btn-ghost btn-sm" to="/fairness">Details →</Link>}>
          {fair.loading && <Loading />}
          <ErrorBox error={fair.error} />
          {fair.data && (
            <ul className="fair-list">
              {fair.data.attributes.map((a) => (
                <li key={a.attribute}>
                  <span>{ATTR_NAMES[a.attribute] || a.attribute}</span>
                  <span className="muted small">approval gap {pct(a.demographic_parity_difference)}</span>
                  <Badge tone={statusTone(a.status)}>{a.status}</Badge>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="Model in use">
          {m && (
            <div className="small">
              <p><b>{m.selected_model}</b> trained on {m.dataset.train_rows} applicants of the German Credit dataset.</p>
              <p>{m.selection_reason}</p>
              <p>An applicant is classed <b>high risk</b> when the predicted probability of default is at least <b>{pct(m.decision_threshold)}</b>.</p>
            </div>
          )}
        </Card>
      </div>

      <Card title="Recent assessments" subtitle="Only the decision and an anonymous input hash are stored — never the raw applicant data.">
        {recent.data?.length ? (
          <table className="table">
            <thead><tr><th>Time</th><th>Decision</th><th>Band</th><th className="num">P(default)</th><th>Main factor</th><th>Input hash</th></tr></thead>
            <tbody>
              {recent.data.map((r) => {
                const top = r.top_factors?.[0]
                return (
                  <tr key={r.assessment_id}>
                    <td>{new Date(r.timestamp).toLocaleString()}</td>
                    <td><Badge tone={statusTone(r.prediction)}>{prettyDecision(r.prediction)}</Badge></td>
                    <td>{r.risk_category}</td>
                    <td className="num">{pct(r.probability_of_default, 1)}</td>
                    <td>
                      {top ? <>{labels[top.feature] || top.feature} <span className={top.shap_value > 0 ? 'txt-bad' : 'txt-good'}>{top.shap_value > 0 ? '▲' : '▼'}</span></> : '–'}
                    </td>
                    <td><code>{r.input_fingerprint}</code></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        ) : <p className="muted">No assessments stored yet.</p>}
      </Card>
    </>
  )
}
