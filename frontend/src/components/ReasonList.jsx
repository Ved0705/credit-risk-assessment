import { formatValue } from '../lib/format'

/** Plain-language key reasons, split into factors raising and lowering risk. */
export default function ReasonList({ contributions, n = 3 }) {
  const up = contributions.filter((c) => c.shap_value > 0).slice(0, n)
  const down = contributions.filter((c) => c.shap_value < 0).slice(0, n)
  const item = (c) => (
    <li key={c.feature}>
      <b>{c.label}</b>: {formatValue(c.feature, c.value)}
    </li>
  )
  return (
    <div className="reasons">
      <div className="reason-col bad">
        <h4>Factors increasing risk</h4>
        {up.length ? <ul>{up.map(item)}</ul> : <p className="muted small">None of note.</p>}
      </div>
      <div className="reason-col good">
        <h4>Factors reducing risk</h4>
        {down.length ? <ul>{down.map(item)}</ul> : <p className="muted small">None of note.</p>}
      </div>
    </div>
  )
}
