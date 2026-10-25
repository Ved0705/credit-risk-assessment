// Formatting helpers shared across pages
export const pct = (v, digits = 0) => (v == null || Number.isNaN(v) ? '–' : `${(v * 100).toFixed(digits)}%`)

const MONEY_FEATURES = new Set(['credit_amount'])
const nf = new Intl.NumberFormat('en-IN')

/** Display an applicant value: thousands separators + unit for money. */
export function formatValue(feature, value) {
  if (typeof value === 'number' && MONEY_FEATURES.has(feature)) return `${nf.format(value)} DM`
  if (typeof value === 'number' && feature === 'duration') return `${value} months`
  return String(value)
}

export const prettyDecision = (p) => (p === 'HIGH_RISK' ? 'High risk' : 'Low risk')

export const statusTone = (s) =>
  ({
    ACCEPTABLE: 'good', STABLE: 'good', LOW: 'good', LOW_RISK: 'good',
    'NEEDS ATTENTION': 'warn', 'MODERATELY STABLE': 'warn', MEDIUM: 'warn',
    CONCERNING: 'bad', UNSTABLE: 'bad', HIGH: 'bad', HIGH_RISK: 'bad',
    'INSUFFICIENT DATA': 'neutral',
  })[s] || 'neutral'

/** "95% CI 74.1–87.4%" style text for a [low, high] interval; '' when missing. */
export function ciText(ci, { asPct = true, digits = 1 } = {}) {
  if (!Array.isArray(ci) || ci.length !== 2) return ''
  const f = (v) => (asPct ? (v * 100).toFixed(digits) : v.toFixed(digits))
  return `95% CI ${f(ci[0])}–${f(ci[1])}${asPct ? '%' : ''}`
}
