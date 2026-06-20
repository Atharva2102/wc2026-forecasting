# WC 2026 Prediction Frontend

React + TypeScript + Tailwind single-page demo for the FastAPI prediction service.

## Run

Start the backend first from the project root:

```bash
python -m uvicorn soccer_api.main:app --host 127.0.0.1 --port 8000
```

Then start the frontend:

```bash
npm install
npm run dev
```

The app reads `VITE_API_URL` and defaults to `http://127.0.0.1:8000`.

```bash
VITE_API_URL=http://127.0.0.1:8000 npm run dev
```
