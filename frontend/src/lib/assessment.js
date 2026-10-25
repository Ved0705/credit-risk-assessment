import { createContext, useContext } from 'react'

// Holds the latest assessment so other pages (e.g. Explainability) can show it
export const AssessmentContext = createContext(null)
export const useAssessment = () => useContext(AssessmentContext)
