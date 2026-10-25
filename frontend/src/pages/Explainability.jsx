import { Link } from 'react-router-dom'
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useAssessment } from '../lib/assessment'
import ShapChart, { ShapTable } from '../components/ShapChart'
import {Callout, Card, ErrorBox, Loading } from '../components/ui'
import { pct } from '../lib/format'
import { api } from '../services/api'
import { useApi } from '../lib/useApi'
import { useTitle } from '../lib/useTitle'

export default function Explainability() {
  useTitle('Explainability')
  const imp = useApi(() => api.importance(), [])
  const { assessment } = useAssessment()
  const ex = assessment.explanation

  return (
    <>
      <header className="page-head">
        <h1>Explainability</h1>
        <p className="muted">SHAP (SHapley Additive exPlanations) splits a prediction into the contribution of each feature.</p>
      </header>

      <Callout>
        <b>How to read SHAP values.</b> The model starts from the average risk of all applicants. Each feature then
        pushes the risk <span className="txt-bad">up (positive value → towards HIGH RISK)</span> or{' '}
        <span className="txt-good">down (negative value → towards LOW RISK)</span>. Adding all contributions to the
        average gives this applicant's prediction. SHAP explains what the model learned from historical data; it does
        not prove that a feature <i>causes</i> default.
      </Callout>

      <div className="two-col">
        <Card title="Local explanation — current applicant">
          {!ex ? (
            <p className="muted">
              No applicant assessed yet. <Link to="/assess">Run an assessment</Link> first.
            </p>
          ) : (
            <>
              <p>
                Prediction: <b className={ex.prediction === 'HIGH_RISK' ? 'txt-bad' : 'txt-good'}>{ex.prediction.replace('_', ' ')}</b>
                {' '}· probability of default <b>{pct(ex.probability_of_default, 1)}</b>
                {' '}· average applicant (base value) <b>{ex.output_units === 'probability' ? pct(ex.base_value, 1) : ex.base_value}</b>
              </p>
              <ShapChart contributions={ex.contributions} limit={18} />
              <p className="muted small">
                SHAP values are in units of <b>{ex.output_units}</b>. Base value + sum of all values ≈ predicted probability.
              </p>
            </>
          )}
        </Card>

        <Card title="Global feature importance" subtitle="Average absolute SHAP value over the 200 test applicants">
          {imp.loading && <Loading />}
          <ErrorBox error={imp.error} onRetry={imp.reload} />
          {imp.data && (
            <ResponsiveContainer width="100%" height={560}>
              <BarChart data={imp.data.importance} layout="vertical" margin={{ left: 10, right: 20 }}>
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis type="category" dataKey="label" width={210} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v) => [v.toFixed(4), 'mean |SHAP|']} />
                <Bar isAnimationActive={false} dataKey="mean_abs_shap" fill="#3d5af1" radius={[3, 3, 3, 3]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>

      {ex && (
        <Card title="All factors for this applicant">
          <ShapTable contributions={ex.contributions} />
        </Card>
      )}
    </>
  )
}
