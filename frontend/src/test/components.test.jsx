import { fireEvent, render, screen } from '@testing-library/react'
import ApplicantForm from '../components/ApplicantForm'
import { SAMPLES } from '../lib/samples'
import RiskResult from '../components/RiskResult'
import { ErrorBox } from '../components/ui'
import ReasonList from '../components/ReasonList'
import { formatValue } from '../lib/format'
import { validate } from '../lib/validate'

const schema = {
  numeric: { age: { min: 18, max: 75, description: 'Age' }, duration: { min: 4, max: 72, description: 'Months' } },
  categorical: { housing: ['rent', 'own', 'for free'] },
  labels: { age: 'Age', duration: 'Loan duration (months)', housing: 'Housing' },
}

test('form renders exactly the fields in the model schema', () => {
  render(<ApplicantForm schema={schema} values={{ age: 30, duration: 12, housing: 'own' }} onChange={() => {}} onSubmit={() => {}} />)
  expect(screen.getByLabelText(/^Age/)).toBeInTheDocument()
  expect(screen.getByLabelText(/Loan duration/)).toBeInTheDocument()
  expect(screen.getByLabelText(/Housing/)).toBeInTheDocument()
  expect(document.querySelectorAll('input, select')).toHaveLength(3)
})

test('form submits current values', () => {
  const onSubmit = vi.fn()
  const values = { age: 30, duration: 12, housing: 'own' }
  render(<ApplicantForm schema={schema} values={values} onChange={() => {}} onSubmit={onSubmit} />)
  fireEvent.click(screen.getByRole('button', { name: /assess credit risk/i }))
  expect(onSubmit).toHaveBeenCalledWith(values)
})

test('sample applicants use only known feature names', () => {
  const keys = Object.keys(SAMPLES['Typical good applicant']).sort()
  Object.values(SAMPLES).forEach((s) => expect(Object.keys(s).sort()).toEqual(keys))
  expect(keys).toHaveLength(18)
})

test('risk result shows verdict', () => {
  render(<RiskResult result={{ prediction: 'HIGH_RISK', risk_category: 'HIGH', probability_of_default: 0.7, decision_threshold: 0.27, risk_score: 70, confidence: 0.7, model_name: 'Random Forest', summary: 's', assessment_id: 'abc12345-x' }} />)
  expect(screen.getByTestId('verdict')).toHaveTextContent('HIGH RISK')
})

test('error box lists validation errors', () => {
  render(<ErrorBox error={{ message: 'Some inputs are invalid.', details: [{ field: 'age', message: 'too low' }] }} />)
  expect(screen.getByRole('alert')).toHaveTextContent('age')
})

test('validation flags out-of-range and missing values', () => {
  const errs = validate(schema, { age: 10, duration: '', housing: 'castle' })
  expect(Object.keys(errs).sort()).toEqual(['age', 'duration', 'housing'])
  expect(validate(schema, { age: 30, duration: 12, housing: 'own' })).toEqual({})
})

test('money values are formatted with unit', () => {
  expect(formatValue('credit_amount', 15000)).toBe('15,000 DM')
  expect(formatValue('duration', 24)).toBe('24 months')
  expect(formatValue('housing', 'own')).toBe('own')
})

test('reason list separates risk-raising and risk-lowering factors', () => {
  const c = [
    { feature: 'checking_status', label: 'Checking account', value: '< 0 DM', shap_value: 0.1 },
    { feature: 'age', label: 'Age', value: 40, shap_value: -0.05 },
  ]
  render(<ReasonList contributions={c} />)
  expect(screen.getByText('Factors increasing risk').parentElement).toHaveTextContent('Checking account')
  expect(screen.getByText('Factors reducing risk').parentElement).toHaveTextContent('Age')
})

test('confidence interval text and insufficient-data tone', async () => {
  const { ciText, statusTone } = await import('../lib/format')
  expect(ciText([0.7414, 0.874], { asPct: false, digits: 3 })).toBe('95% CI 0.741–0.874')
  expect(ciText([0.09, 0.4])).toBe('95% CI 9.0–40.0%')
  expect(ciText(null)).toBe('')
  expect(statusTone('INSUFFICIENT DATA')).toBe('neutral')
})
