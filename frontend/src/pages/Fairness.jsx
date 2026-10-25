import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {Badge, Callout, Card, ErrorBox, Loading } from '../components/ui'
import { ciText, pct, statusTone } from '../lib/format'
import { api } from '../services/api'
import { useApi } from '../lib/useApi'
import { useTitle } from '../lib/useTitle'

const TITLES = { sex: 'Sex', age_group: 'Age group', foreign_worker_group: 'Foreign-worker status' }

function AttributeCard({ a }) {
  const chart = a.groups.map((g) => ({
    group: `${g.group} (n=${g.count})`,
    'Approval rate': +(g.approval_rate * 100).toFixed(1),
    'Wrongly rejected (FPR)': +(g.false_positive_rate * 100).toFixed(1),
    'Missed defaulters (FNR)': +(g.false_negative_rate * 100).toFixed(1),
  }))
  return (
    <Card
      title={TITLES[a.attribute] || a.attribute}
      subtitle={a.description}
      actions={<Badge tone={statusTone(a.status)}>{a.status}</Badge>}
    >
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={chart} margin={{ top: 8, right: 8, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="group" tick={{ fontSize: 12 }} />
          <YAxis unit="%" tick={{ fontSize: 12 }} domain={[0, 100]} />
          <Tooltip formatter={(v) => `${v}%`} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar isAnimationActive={false} dataKey="Approval rate" fill="#3d5af1" />
          <Bar isAnimationActive={false} dataKey="Wrongly rejected (FPR)" fill="#e3a008" />
          <Bar isAnimationActive={false} dataKey="Missed defaulters (FNR)" fill="#d6453d" />
        </BarChart>
      </ResponsiveContainer>
      <table className="table">
        <thead>
          <tr><th>Group</th><th className="num">n</th><th className="num">Approval</th><th className="num">Actually good</th><th className="num">FPR</th><th className="num">FNR</th><th className="num">Accuracy</th></tr>
        </thead>
        <tbody>
          {a.groups.map((g) => (
            <tr key={g.group}>
              <td>{g.group}</td><td className="num">{g.count}</td><td className="num">{pct(g.approval_rate, 1)}</td>
              <td className="num">{pct(g.actual_good_rate, 1)}</td><td className="num">{pct(g.false_positive_rate, 1)}</td>
              <td className="num">{pct(g.false_negative_rate, 1)}</td><td className="num">{pct(g.accuracy, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="stat-row">
        <div className="stat"><span className="stat-label">Demographic parity difference</span><span className="stat-value sm">{pct(a.demographic_parity_difference, 1)}</span><span className="stat-hint">gap in approval rates{a.demographic_parity_difference_ci && <><br />{ciText(a.demographic_parity_difference_ci)}</>}</span></div>
        <div className="stat"><span className="stat-label">Demographic parity ratio</span><span className="stat-value sm">{a.demographic_parity_ratio.toFixed(2)}</span><span className="stat-hint">1.0 = equal; &lt;0.8 often flagged</span></div>
        <div className="stat"><span className="stat-label">Equalized odds difference</span><span className="stat-value sm">{pct(a.equalized_odds_difference, 1)}</span><span className="stat-hint">largest gap in error rates{a.equalized_odds_difference_ci && <><br />{ciText(a.equalized_odds_difference_ci)}</>}</span></div>
      </div>
      <p>{a.interpretation}</p>
      {a.warnings.map((w) => <Callout key={w} tone="warn">{w}</Callout>)}
    </Card>
  )
}

export default function Fairness() {
  useTitle('Fairness')
  const f = useApi(() => api.fairness(), [])
  return (
    <>
      <header className="page-head">
        <h1>Fairness Analysis</h1>
        <p className="muted">Group-level audit with Fairlearn on the held-out test set.</p>
      </header>
      {f.loading && <Loading />}
      <ErrorBox error={f.error} onRetry={f.reload} />
      {f.data && (
        <>
          <Callout tone={statusTone(f.data.overall_status) === 'good' ? 'info' : 'warn'}>
            <b>Overall fairness status: {f.data.overall_status}.</b> Evaluated on the {f.data.evaluated_on}. A group is
            ACCEPTABLE when both the approval-rate gap and the error-rate gap are below {pct(f.data.thresholds_used.acceptable_below)},
            NEEDS ATTENTION below {pct(f.data.thresholds_used.needs_attention_below)}, otherwise CONCERNING.
            These thresholds are a common rule of thumb, not a legal standard.
            {f.data.thresholds_used.min_group_size && <> An attribute with any group smaller than {f.data.thresholds_used.min_group_size} test
            applicants is marked INSUFFICIENT DATA and left out of the overall status. Ranges are 95% bootstrap intervals.</>}
          </Callout>
          <div className="glossary">
            <div><b>Approval rate</b> share of the group predicted LOW RISK</div>
            <div><b>FPR</b> good payers wrongly rejected</div>
            <div><b>FNR</b> defaulters wrongly approved</div>
            <div><b>Actually good</b> real good-credit share in the data</div>
          </div>
          <div className="stack">
            {f.data.attributes.map((a) => <AttributeCard key={a.attribute} a={a} />)}
          </div>
          <Card title="Limitations of this fairness analysis">
            <ul className="limitations">
              {f.data.limitations.map((l) => <li key={l}>{l}</li>)}
            </ul>
          </Card>
        </>
      )}
    </>
  )
}
