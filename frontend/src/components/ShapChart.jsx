import { formatValue } from '../lib/format'
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const RED = '#d6453d'
const GREEN = '#1f9d6b'

/** Horizontal bar chart of SHAP contributions. Positive = towards HIGH risk (red). */
export default function ShapChart({ contributions, limit = 10 }) {
  const data = contributions.slice(0, limit).map((c) => ({
    name: c.label,
    value: c.shap_value,
    shown: formatValue(c.feature, c.value),
  }))
  return (
    <div className="shap-chart">
      <ResponsiveContainer width="100%" height={Math.max(220, data.length * 34)}>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30, top: 4, bottom: 4 }}>
          <XAxis type="number" tick={{ fontSize: 12 }} />
          <YAxis type="category" dataKey="name" width={210} tick={{ fontSize: 12 }} />
          <ReferenceLine x={0} stroke="#8892a6" />
          <Tooltip
            formatter={(v, _n, p) => [
              `${v > 0 ? '+' : ''}${v.toFixed(4)} (value: ${p.payload.shown})`,
              v > 0 ? 'Increases risk' : 'Reduces risk',
            ]}
          />
          <Bar isAnimationActive={false} dataKey="value" radius={[3, 3, 3, 3]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.value > 0 ? RED : GREEN} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="legend">
        <span><i style={{ background: RED }} /> pushes towards HIGH RISK</span>
        <span><i style={{ background: GREEN }} /> pushes towards LOW RISK</span>
      </div>
    </div>
  )
}

export function ShapTable({ contributions }) {
  return (
    <table className="table">
      <thead>
        <tr>
          <th>Factor</th>
          <th>Applicant value</th>
          <th className="num">SHAP</th>
          <th>Effect</th>
        </tr>
      </thead>
      <tbody>
        {contributions.map((c) => (
          <tr key={c.feature}>
            <td>{c.label}</td>
            <td>{formatValue(c.feature, c.value)}</td>
            <td className={`num ${c.shap_value > 0 ? 'txt-bad' : 'txt-good'}`}>
              {c.shap_value > 0 ? '+' : ''}
              {c.shap_value.toFixed(4)}
            </td>
            <td>{c.direction === 'increases_risk' ? '▲ increases risk' : '▼ reduces risk'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
