// Example applicants for quick demos. Keys must match the model's 18 features.
export const SAMPLES = {
  'Typical good applicant': {
    checking_status: 'no checking account', duration: 24,
    credit_history: 'existing credits paid back duly till now', purpose: 'car (used)',
    credit_amount: 3500, savings_status: '100 - 500 DM', employment: '1 - 4 years',
    installment_rate: 3, other_debtors: 'none', residence_since: 2,
    property: 'car or other property', age: 35, other_installment_plans: 'none',
    housing: 'own', existing_credits: 1, job: 'skilled employee / official',
    num_dependents: 1, telephone: 'none',
  },
  'Borderline applicant': {
    checking_status: '0 - 200 DM', duration: 24, credit_history: 'existing credits paid back duly till now',
    purpose: 'furniture/equipment', credit_amount: 2500, savings_status: '< 100 DM',
    employment: '1 - 4 years', installment_rate: 4, other_debtors: 'none', residence_since: 2,
    property: 'car or other property', age: 30, other_installment_plans: 'none',
    housing: 'own', existing_credits: 1, job: 'skilled employee / official',
    num_dependents: 1, telephone: 'none',
  },
  'Risky applicant': {
    checking_status: '< 0 DM', duration: 60, credit_history: 'no credits / all paid back duly',
    purpose: 'education', credit_amount: 15000, savings_status: '< 100 DM',
    employment: '< 1 year', installment_rate: 4, other_debtors: 'none', residence_since: 1,
    property: 'unknown / no property', age: 22, other_installment_plans: 'bank',
    housing: 'rent', existing_credits: 1, job: 'unskilled resident',
    num_dependents: 1, telephone: 'none',
  },
}
export const DEFAULT_SAMPLE = 'Typical good applicant'
