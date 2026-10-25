# Frontend (React + Vite)

```bash
npm install
cp .env.example .env   # optional; default API URL is http://localhost:8000
npm run dev            # http://localhost:5173
npm test               # component tests (Vitest)
npm run build          # production build in dist/
```

Pages: Dashboard, Credit Assessment, Explainability, Fairness, Model Performance.
The applicant form is generated from `GET /model/features`, so its fields always match the trained model.
