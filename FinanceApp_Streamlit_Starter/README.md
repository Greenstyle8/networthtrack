# Personal Finance Command Center (Streamlit)

**What it is:** track bank balances, savings rates, investments, expenses, and forecasts.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Data saves to `finance.db` (SQLite).

## Deploy to Streamlit Community Cloud
1. Create a new GitHub repository and upload these files.
2. Go to https://share.streamlit.io/ → “New app” → pick your repo → Deploy.
3. Important: Cloud storage can reset. Use **Data Export** often to back up CSVs.

## CSV upload format
- Columns: `account,date,balance,rate`
- Date: `YYYY-MM-DD`
- `rate` optional; `4.5` = 4.5% APR (or `0.045`).

## Next step (optional, later)
Migrate to a hosted Postgres DB for permanent storage.
