import { pct, statusTone } from '../lib/format'
import { Badge } from './ui'

/** Verdict + probability meter with the decision threshold marked. */
export default function RiskResult({ result, assessedAt }) {
  const high = result.prediction === 'HIGH_RISK'
  const p = result.probability_of_default
  return (
    <div className={`verdict ${high ? 'verdict-bad' : 'verdict-good'}`}>
      <div className="verdict-top">
        <div>
          <span className="eyebrow">Credit risk decision</span>
          <h3 data-testid="verdict">{high ? 'HIGH RISK' : 'LOW RISK'}</h3>
        </div>
        <Badge tone={statusTone(result.risk_category)}>{result.risk_category} risk band</Badge>
      </div>

      <div className="meter" role="img" aria-label={`Probability of default ${pct(p, 1)}, threshold ${pct(result.decision_threshold)}`}>
        <div className="meter-threshold" style={{ left: `${result.decision_threshold * 100}%` }}>
          <span>threshold {pct(result.decision_threshold)}</span>
        </div>
        <div className="meter-marker" style={{ left: `${p * 100}%` }}>
          <span>{pct(p, 1)}</span>
        </div>
      </div>
      <div className="meter-scale"><span>0% · safe</span><span>probability of default</span><span>100%</span></div>

      <div className="verdict-stats">
        <div><span>Probability of default</span><b>{pct(p, 1)}</b></div>
        <div><span>Risk score</span><b>{result.risk_score}/100</b></div>
        <div><span>Confidence in decision</span><b>{pct(result.confidence, 1)}</b></div>
        <div><span>Model</span><b>{result.model_name}</b></div>
      </div>
      <p className="verdict-summary">{result.summary}</p>
      {result.risk_category === 'MEDIUM' && (
        <p className="small note">
          <b>Borderline case:</b> the probability is above the cautious decision threshold but below 50%.
          A human reviewer should look at this application.
        </p>
      )}
      <p className="verdict-meta">
        Assessment <code>{result.assessment_id?.slice(0, 8)}</code>
        {assessedAt && <> · {new Date(assessedAt).toLocaleString()}</>}
      </p>
    </div>
  )
}
