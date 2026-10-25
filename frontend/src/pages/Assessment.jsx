import { useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import ApplicantForm from '../components/ApplicantForm'
import ReasonList from '../components/ReasonList'
import RiskResult from '../components/RiskResult'
import ShapChart from '../components/ShapChart'
import StabilityPanel from '../components/StabilityPanel'
import { Card, EmptyState, ErrorBox, Loading } from '../components/ui'
import { useAssessment } from '../lib/assessment'
import { DEFAULT_SAMPLE, SAMPLES } from '../lib/samples'
import { useApi } from '../lib/useApi'
import { useTitle } from '../lib/useTitle'
import { validate } from '../lib/validate'
import { formatValue } from '../lib/format'
import { api } from '../services/api'

export default function Assessment() {
  useTitle('Credit Assessment')
  const schema = useApi(() => api.features(), [])
  const { assessment, setAssessment } = useAssessment()
  const [values, setValues] = useState(assessment.applicant || SAMPLES[DEFAULT_SAMPLE])
  const [fieldErrors, setFieldErrors] = useState({})
  const [busy, setBusy] = useState(false)
  const [stabBusy, setStabBusy] = useState(false)
  const [error, setError] = useState(null)
  const resultRef = useRef(null)

  const changeValues = (v) => {
    setValues(v)
    if (Object.keys(fieldErrors).length && schema.data) setFieldErrors(validate(schema.data, v))
  }

  const submit = async (applicant) => {
    const errs = validate(schema.data, applicant)
    setFieldErrors(errs)
    if (Object.keys(errs).length) {
      setError({ message: 'Please correct the highlighted fields.' })
      document.querySelector(`[name="${Object.keys(errs)[0]}"]`)?.focus()
      return
    }
    setBusy(true)
    setError(null)
    try {
      const [result, explanation] = await Promise.all([api.predict(applicant), api.explain(applicant)])
      setAssessment({ applicant, result, explanation, stability: null, assessedAt: new Date().toISOString() })
      // on narrow screens the result is below the form
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  const runStability = async () => {
    setStabBusy(true)
    setError(null)
    try {
      const stability = await api.stability(assessment.applicant)
      setAssessment((a) => ({ ...a, stability }))
    } catch (e) {
      setError(e)
    } finally {
      setStabBusy(false)
    }
  }

  const reset = () => {
    setValues(SAMPLES[DEFAULT_SAMPLE])
    setFieldErrors({})
    setError(null)
    setAssessment({ applicant: null, result: null, explanation: null, stability: null })
  }

  const { result, explanation, stability, assessedAt } = assessment
  return (
    <>
      <header className="print-only print-head">
        <h1>Credit Risk Assessment Report</h1>
        <p className="muted">CreditLens · {assessedAt ? new Date(assessedAt).toLocaleString() : ''}</p>
      </header>
      <header className="page-head row no-print">
        <div>
          <h1>Credit Assessment</h1>
          <p className="muted">Enter applicant details. Every field is a feature the trained model actually uses.</p>
        </div>
        <div className="head-actions no-print">
          {result && <button className="btn btn-ghost" onClick={() => window.print()}>Print report</button>}
          <button className="btn btn-ghost" onClick={reset}>New assessment</button>
        </div>
      </header>
      <ErrorBox error={error} />
      <div className="two-col">
        <Card title="Applicant details" className="no-print">
          {schema.loading && <Loading text="Loading model features…" />}
          <ErrorBox error={schema.error} onRetry={schema.reload} />
          {schema.data && (
            <ApplicantForm
              schema={schema.data} values={values} onChange={changeValues}
              onSubmit={submit} loading={busy} errors={fieldErrors}
            />
          )}
        </Card>
        <div className="stack" ref={resultRef}>
          {!result && !busy && (
            <Card title="Result">
              <EmptyState title="No assessment yet">
                Fill in the form (or load an example) and press <b>Assess credit risk</b> to see the decision,
                its explanation and a stability check.
              </EmptyState>
            </Card>
          )}
          {busy && <Card><Loading text="Running the model and computing the SHAP explanation…" /></Card>}
          {result && !busy && (
            <>
              <Card title="Decision">
                <RiskResult result={result} assessedAt={assessedAt} />
              </Card>
              {explanation && (
                <Card
                  title="Why was this decision made?"
                  subtitle="SHAP contributions: red pushes towards high risk, green towards low risk."
                  actions={<Link className="btn btn-ghost btn-sm no-print" to="/explain">Full explanation →</Link>}
                >
                  <ReasonList contributions={explanation.contributions} />
                  <ShapChart contributions={explanation.contributions} limit={8} />
                </Card>
              )}
              <Card
                title="Stability check"
                subtitle="Does the decision hold if the loan duration and amount change slightly?"
                actions={
                  <button className="btn btn-primary btn-sm no-print" onClick={runStability} disabled={stabBusy}>
                    {stabBusy ? 'Testing…' : stability ? 'Re-run test' : 'Run stability test'}
                  </button>
                }
              >
                {stabBusy && <Loading text="Testing 20 perturbed versions…" />}
                {!stabBusy && (stability ? <StabilityPanel data={stability} /> : <p className="muted">Not run yet.</p>)}
              </Card>
              {schema.data && assessment.applicant && (
                <Card title="Applicant details" className="print-only">
                  <table className="table">
                    <tbody>
                      {Object.entries(assessment.applicant).map(([k, v]) => (
                        <tr key={k}><td>{schema.data.labels[k] || k}</td><td>{formatValue(k, v)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </Card>
              )}
              <p className="muted small print-only">
                Generated by CreditLens — an academic demonstration model trained on the 1994 German Credit dataset.
                Not a real lending decision.
              </p>
            </>
          )}
        </div>
      </div>
    </>
  )
}
