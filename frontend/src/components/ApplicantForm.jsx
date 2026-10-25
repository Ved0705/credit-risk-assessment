/**
 * Applicant form generated from GET /model/features, so the fields always
 * match exactly the features the trained model uses.
 */
import { SAMPLES } from '../lib/samples'

const SECTIONS = [
  ['Loan request', ['credit_amount', 'duration', 'purpose', 'installment_rate']],
  ['Financial situation', ['checking_status', 'savings_status', 'credit_history', 'existing_credits', 'other_installment_plans', 'other_debtors', 'property']],
  ['Personal & employment', ['age', 'employment', 'job', 'housing', 'residence_since', 'num_dependents', 'telephone']],
]

export default function ApplicantForm({ schema, values, onChange, onSubmit, loading, errors = {} }) {
  const known = new Set(SECTIONS.flatMap(([, f]) => f))
  const all = [...Object.keys(schema.numeric), ...Object.keys(schema.categorical)]
  const extra = all.filter((f) => !known.has(f))
  const sections = SECTIONS.map(([t, f]) => [t, f.filter((x) => all.includes(x))])
  if (extra.length) sections.push(['Other', extra])

  const set = (k, v) => onChange({ ...values, [k]: v })

  const field = (name) => {
    const label = schema.labels[name] || name
    const err = errors[name]
    if (schema.numeric[name]) {
      const { min, max, description } = schema.numeric[name]
      return (
        <label key={name} className={`field ${err ? 'has-error' : ''}`}>
          <span>{label}</span>
          <input
            type="number" name={name} min={min} max={max} step="1" inputMode="numeric"
            value={values[name] ?? ''} aria-invalid={!!err}
            onChange={(e) => set(name, e.target.value === '' ? '' : Number(e.target.value))}
          />
          <small>{err || `${description} (${min}–${max})`}</small>
        </label>
      )
    }
    return (
      <label key={name} className={`field ${err ? 'has-error' : ''}`}>
        <span>{label}</span>
        <select name={name} value={values[name] ?? ''} onChange={(e) => set(name, e.target.value)}>
          {schema.categorical[name].map((o) => (
            <option key={o} value={o}>{o}</option>
          ))}
        </select>
        {err && <small>{err}</small>}
      </label>
    )
  }

  return (
    <form
      className="applicant-form"
      noValidate
      onSubmit={(e) => {
        e.preventDefault()
        onSubmit(values)
      }}
    >
      <div className="samples">
        <span className="muted small">Load example:</span>
        {Object.keys(SAMPLES).map((k) => (
          <button type="button" key={k} className="btn btn-ghost btn-sm" onClick={() => onChange(SAMPLES[k])}>
            {k}
          </button>
        ))}
      </div>
      {sections.map(([title, fields]) => (
        <fieldset key={title}>
          <legend>{title}</legend>
          <div className="grid-fields">{fields.map(field)}</div>
        </fieldset>
      ))}
      <p className="muted small">
        Sex/marital status and foreign-worker status are in the dataset but are deliberately <b>not</b> asked for:
        the model does not use them. They are only used for the fairness audit.
      </p>
      <div className="form-actions">
        <button className="btn btn-primary" type="submit" disabled={loading}>
          {loading ? 'Assessing…' : 'Assess credit risk'}
        </button>
      </div>
    </form>
  )
}
