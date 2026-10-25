/** Returns {field: message} for values outside the model's allowed ranges. */
export function validate(schema, values) {
  const errors = {}
  Object.entries(schema.numeric).forEach(([k, { min, max }]) => {
    const v = values[k]
    if (v === '' || v == null || Number.isNaN(v)) errors[k] = 'Required'
    else if (v < min || v > max) errors[k] = `Must be between ${min} and ${max}`
  })
  Object.entries(schema.categorical).forEach(([k, opts]) => {
    if (!opts.includes(values[k])) errors[k] = 'Choose an option'
  })
  return errors
}
