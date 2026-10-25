import { lazy, useState } from 'react'
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import { AssessmentContext } from './lib/assessment'

// Pages are loaded on demand (smaller initial bundle)
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Assessment = lazy(() => import('./pages/Assessment'))
const Explainability = lazy(() => import('./pages/Explainability'))
const Fairness = lazy(() => import('./pages/Fairness'))

function NotFound() {
  return (
    <div className="empty page-404">
      <b>Page not found</b>
      <p className="muted">The page you are looking for does not exist.</p>
      <Link className="btn btn-primary" to="/">Back to dashboard</Link>
    </div>
  )
}

export default function App() {
  const [assessment, setAssessment] = useState({ applicant: null, result: null, explanation: null, stability: null })
  return (
    <AssessmentContext.Provider value={{ assessment, setAssessment }}>
      <BrowserRouter>
        <Routes>
            <Route element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="assess" element={<Assessment />} />
              <Route path="explain" element={<Explainability />} />
              <Route path="fairness" element={<Fairness />} />
              <Route path="*" element={<NotFound />} />
            </Route>
        </Routes>
      </BrowserRouter>
    </AssessmentContext.Provider>
  )
}
