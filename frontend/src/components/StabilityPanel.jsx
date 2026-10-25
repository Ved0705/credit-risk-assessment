import { formatValue, pct, statusTone } from '../lib/format'
import { Badge } from './ui'

export default function StabilityPanel({ data }) {
  return (
    <div>
      <div className="stat-row">
        <div className="stat"><span className="stat-label">Original prediction</span><span className="stat-value sm">{data.original_prediction === 'HIGH_RISK' ? 'HIGH RISK' : 'LOW RISK'}</span></div>
        <div className="stat"><span className="stat-label">Perturbed samples</span><span className="stat-value sm">{data.n_samples}</span></div>
        <div className="stat"><span className="stat-label">Same prediction</span><span className="stat-value sm">{data.same_prediction}/{data.n_samples}</span></div>
        <div className={`stat tone-${statusTone(data.status)}`}><span className="stat-label">Stability</span><span className="stat-value sm">{pct(data.stability_score)}</span><Badge tone={statusTone(data.status)}>{data.status}</Badge></div>
      </div>
      <p>{data.interpretation}</p>
      <p className="muted small">
        Probability across perturbed samples: mean {pct(data.probability_mean, 1)}, range {pct(data.probability_min, 1)} – {pct(data.probability_max, 1)}.
        Randomly perturbed: {data.perturbed_features.join(', ')} (±{pct(data.noise_level)}). Age is not perturbed because it is a sensitive attribute.
      </p>
      <h4>Feature sensitivity (one feature changed at a time)</h4>
      <table className="table">
        <thead>
          <tr><th>Feature</th><th>Original</th><th>Tested values</th><th className="num">Max Δ probability</th><th>Flips decision?</th></tr>
        </thead>
        <tbody>
          {data.sensitive_features.map((s) => (
            <tr key={s.feature} className={s.flips_decision ? 'row-flag' : ''}>
              <td>{s.label}</td>
              <td>{formatValue(s.feature, s.original_value)}</td>
              <td>{s.tested_values.map((v) => formatValue(s.feature, v)).join(', ')}</td>
              <td className="num">{pct(s.max_probability_change, 1)}</td>
              <td>{s.flips_decision ? <Badge tone="bad">YES</Badge> : <Badge tone="good">no</Badge>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
