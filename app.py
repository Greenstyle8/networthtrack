"""
app.py — Streamlit Personal Finance Command Center (single-file)

Features:
- Add/manage accounts (bank, savings, ISA, investments, pension)
- Enter balances weekly/monthly (manual form or CSV upload)
- Track savings rates/yields and their changes over time
- Track expenses by category
- Dashboard visuals: net worth trend, allocation, per-account growth
- Forecast: cash via latest rates, investments via historical CAGR or custom rate

How to run locally (easiest):
1) pip install -r requirements.txt
2) streamlit run app.py

How to deploy on Streamlit Cloud (simple):
1) Push these files to a new GitHub repo
2) Go to https://share.streamlit.io/ (Streamlit Community Cloud)
3) New app → pick your repo and branch → Deploy
(Heads-up: SQLite can reset on redeploys. Use the "Data Export" tab often.)
"""
import sqlite3
from datetime import datetime, date
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

DB_PATH = "finance.db"

# --------------------------
# DB Helpers
# --------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('bank','savings','isa','investment','pension')),
            currency TEXT DEFAULT 'GBP',
            target_allocation REAL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS balances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            balance REAL NOT NULL,
            FOREIGN KEY(account_id) REFERENCES accounts(id) ON DELETE CASCADE,
            UNIQUE(account_id, date)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            apr REAL NOT NULL,  -- 0.045 for 4.5%
            FOREIGN KEY(account_id) REFERENCES accounts(id) ON DELETE CASCADE,
            UNIQUE(account_id, date)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            amount REAL NOT NULL -- positive for spend
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS contributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            amount REAL NOT NULL, -- + deposit, - withdrawal
            FOREIGN KEY(account_id) REFERENCES accounts(id) ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    conn.close()


# --------------------------
# Data Access
# --------------------------
def fetch_df(query: str, params: Tuple = ()):
    conn = get_conn()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def upsert_account(name: str, type_: str, currency: str = "GBP", target_allocation: Optional[float] = None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO accounts(name, type, currency, target_allocation)
        VALUES(?,?,?,?)
        ON CONFLICT(name) DO UPDATE SET type=excluded.type, currency=excluded.currency, target_allocation=excluded.target_allocation
        """,
        (name.strip(), type_.strip(), currency.strip(), target_allocation),
    )
    conn.commit()
    conn.close()

def add_balance(account_id: int, d: date, balance: float):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO balances(account_id, date, balance)
        VALUES(?,?,?)
        ON CONFLICT(account_id, date) DO UPDATE SET balance=excluded.balance
        """,
        (account_id, d.isoformat(), float(balance)),
    )
    conn.commit()
    conn.close()

def add_rate(account_id: int, d: date, apr: float):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO rates(account_id, date, apr)
        VALUES(?,?,?)
        ON CONFLICT(account_id, date) DO UPDATE SET apr=excluded.apr
        """,
        (account_id, d.isoformat(), float(apr)),
    )
    conn.commit()
    conn.close()

def add_expense(d: date, category: str, description: str, amount: float):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO expenses(date, category, description, amount) VALUES(?,?,?,?)",
        (d.isoformat(), category.strip(), description.strip(), float(amount)),
    )
    conn.commit()
    conn.close()

def add_contribution(account_id: int, d: date, amount: float):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO contributions(account_id, date, amount) VALUES(?,?,?)",
        (account_id, d.isoformat(), float(amount)),
    )
    conn.commit()
    conn.close()


# --------------------------
# Analytics helpers
# --------------------------
def latest_balances_df() -> pd.DataFrame:
    q = """
    SELECT a.id as account_id, a.name, a.type, a.currency,
           b.date as last_date,
           b.balance as last_balance
    FROM accounts a
    LEFT JOIN (
      SELECT account_id, MAX(date) AS max_date FROM balances GROUP BY account_id
    ) lb ON a.id = lb.account_id
    LEFT JOIN balances b ON b.account_id = lb.account_id AND b.date = lb.max_date
    ORDER BY a.name
    """
    return fetch_df(q)

def net_worth_series() -> pd.DataFrame:
    q = """
    SELECT date, SUM(balance) as net_worth
    FROM balances
    GROUP BY date
    ORDER BY date
    """
    df = fetch_df(q)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

def allocation_snapshot() -> pd.DataFrame:
    df = latest_balances_df()
    if df.empty:
        return df
    groups = df.groupby('type')['last_balance'].sum().reset_index()
    groups['pct'] = groups['last_balance'] / groups['last_balance'].sum()
    return groups

def historical_cagr(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or len(df) < 2:
        return None
    df = df.sort_values('date')
    start_val = df['value'].iloc[0]
    end_val = df['value'].iloc[-1]
    if start_val <= 0 or end_val <= 0:
        return None
    days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    years = max(days / 365.25, 1e-6)
    return (end_val / start_val) ** (1 / years) - 1

def account_series(account_id: int) -> pd.DataFrame:
    q = """
    SELECT date, balance FROM balances WHERE account_id=? ORDER BY date
    """
    df = fetch_df(q, (account_id,))
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'balance': 'value'})
    return df

def latest_rate_for_account(account_id: int) -> Optional[float]:
    q = """
    SELECT apr FROM rates WHERE account_id=? ORDER BY date DESC LIMIT 1
    """
    df = fetch_df(q, (account_id,))
    if df.empty:
        return None
    return float(df['apr'].iloc[0])

def forecast_cash(balance: float, apr: Optional[float], months: int = 12) -> pd.DataFrame:
    if apr is None:
        apr = 0.0
    monthly_rate = apr / 12.0
    vals = []
    for _ in range(months):
        balance *= (1 + monthly_rate)
        vals.append(balance)
    dates = pd.date_range(date.today(), periods=months, freq='M')
    return pd.DataFrame({'date': dates, 'value': vals})

def forecast_investment(series: pd.DataFrame, months: int = 12, override_rate: Optional[float] = None) -> pd.DataFrame:
    if series is None or series.empty:
        return pd.DataFrame(columns=['date', 'value'])
    last_val = float(series['value'].iloc[-1])
    if override_rate is None:
        r = historical_cagr(series)
        if r is None:
            r = 0.0
    else:
        r = override_rate
    monthly_rate = r / 12.0
    vals = []
    balance = last_val
    for _ in range(months):
        balance *= (1 + monthly_rate)
        vals.append(balance)
    dates = pd.date_range(date.today(), periods=months, freq='M')
    return pd.DataFrame({'date': dates, 'value': vals})


# --------------------------
# UI
# --------------------------
def sidebar_nav() -> str:
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        [
            "Dashboard",
            "Accounts",
            "Balances & Rates",
            "Expenses",
            "Forecast",
            "Data Export"
        ],
    )

def page_dashboard():
    st.title("Personal Finance Command Center")
    st.caption("Simple, fast, and good enough to make smarter money moves.")

    lb = latest_balances_df()
    col1, col2, col3 = st.columns(3)
    total = lb['last_balance'].fillna(0).sum() if not lb.empty else 0
    cash = lb[lb['type'].isin(['bank', 'savings'])]['last_balance'].fillna(0).sum() if not lb.empty else 0
    inv = lb[lb['type'].isin(['isa', 'investment', 'pension'])]['last_balance'].fillna(0).sum() if not lb.empty else 0

    col1.metric("Total Net Worth", f"£{total:,.0f}")
    col2.metric("Cash", f"£{cash:,.0f}")
    col3.metric("Investments", f"£{inv:,.0f}")

    nw = net_worth_series()
    st.subheader("Net Worth Over Time")
    if nw.empty:
        st.info("Add some balances to see your trend.")
    else:
        fig = px.line(nw, x='date', y='net_worth', labels={'net_worth': '£'}, title=None)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Allocation Snapshot")
    alloc = allocation_snapshot()
    if alloc.empty:
        st.info("No accounts yet. Add accounts and balances.")
    else:
        fig2 = px.pie(alloc, names='type', values='last_balance', hole=0.35)
        st.plotly_chart(fig2, use_container_width=True)

    if not lb.empty:
        st.subheader("Accounts")
        for _, row in lb.iterrows():
            with st.expander(f"{row['name']} • {row['type'].upper()} • £{(row['last_balance'] or 0):,.0f}"):
                acc_id = int(row['account_id'])
                series = account_series(acc_id)
                if series.empty:
                    st.write("No history yet.")
                else:
                    fig = px.line(series, x='date', y='value', title=None, labels={'value': '£'})
                    st.plotly_chart(fig, use_container_width=True)
                latest_apr = latest_rate_for_account(acc_id)
                st.write(f"Latest rate: {f'{latest_apr*100:.2f}%' if latest_apr is not None else '—'}")

def page_accounts():
    st.title("Accounts")
    st.caption("Add bank, savings, ISA, investment, or pension accounts.")

    with st.form("add_acct"):
        name = st.text_input("Account name")
        type_ = st.selectbox("Type", ['bank', 'savings', 'isa', 'investment', 'pension'])
        currency = st.selectbox("Currency", ['GBP', 'USD', 'EUR'], index=0)
        target_allocation = st.number_input("Target allocation % (optional)", min_value=0.0, max_value=100.0, value=0.0)
        submitted = st.form_submit_button("Save / Update")
        if submitted and name:
            upsert_account(name, type_, currency, (target_allocation or None) / 100.0 if target_allocation else None)
            st.success("Saved.")

    st.divider()
    st.subheader("Existing Accounts")
    df = fetch_df("SELECT id, name, type, currency, target_allocation FROM accounts ORDER BY name")
    if df.empty:
        st.info("No accounts yet.")
    else:
        df['target_allocation'] = (df['target_allocation'] * 100).round(2)
        st.dataframe(df, use_container_width=True)

def page_balances_rates():
    st.title("Balances & Rates")
    st.caption("Enter snapshots weekly/monthly. Track savings rate changes.")

    accts = fetch_df("SELECT id, name, type FROM accounts ORDER BY name")
    acct_map = {f"{r['name']} ({r['type']})": int(r['id']) for _, r in accts.iterrows()} if not accts.empty else {}

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Add Balance Snapshot")
        if not acct_map:
            st.info("Add an account first.")
        else:
            acc_label = st.selectbox("Account", list(acct_map.keys()))
            d = st.date_input("Date", value=date.today())
            bal = st.number_input("Balance (£)", step=100.0, min_value=0.0)
            if st.button("Save Balance"):
                add_balance(acct_map[acc_label], d, bal)
                st.success("Balance saved.")

    with colB:
        st.subheader("Add/Update Rate (APR)")
        if not acct_map:
            st.info("Add an account first.")
        else:
            acc_label_r = st.selectbox("Account ", list(acct_map.keys()), key="rateacct")
            d_r = st.date_input("Date ", value=date.today(), key="ratedate")
            apr = st.number_input("APR (e.g., 4.5 for 4.5%)", step=0.1, min_value=0.0)
            if st.button("Save Rate"):
                add_rate(acct_map[acc_label_r], d_r, apr/100.0)
                st.success("Rate saved.")

    st.divider()
    st.subheader("Bulk CSV Upload")
    st.caption("Upload balances with columns: account,date,balance and (optional) rate. Dates as YYYY-MM-DD.")
    up = st.file_uploader("CSV file", type=['csv'])
    if up:
        try:
            df = pd.read_csv(up)
            needed = {'account','date','balance'}
            lower = {c.lower(): c for c in df.columns}
            if not needed.issubset(set(lower.keys())):
                st.error("CSV must include account,date,balance")
            else:
                for _, r in df.iterrows():
                    acc_name = r[lower['account']]
                    acc_row = fetch_df("SELECT id FROM accounts WHERE name=?", (acc_name,))
                    if acc_row.empty:
                        st.warning(f"Account '{acc_name}' not found. Skipping row.")
                        continue
                    acc_id = int(acc_row['id'].iloc[0])
                    d = pd.to_datetime(r[lower['date']]).date()
                    bal = float(r[lower['balance']])
                    add_balance(acc_id, d, bal)
                    if 'rate' in lower and not pd.isna(r[lower['rate']]):
                        apr = float(r[lower['rate']])/100.0 if r[lower['rate']] > 1.0 else float(r[lower['rate']])
                        add_rate(acc_id, d, apr)
                st.success("CSV processed.")
        except Exception as e:
            st.exception(e)

    st.divider()
    st.subheader("Recent Balances")
    lb = latest_balances_df()
    if lb.empty:
        st.info("No data yet.")
    else:
        st.dataframe(lb, use_container_width=True)

def page_expenses():
    st.title("Expenses")
    st.caption("Track monthly spending by category. Keep it simple.")

    with st.form("add_exp"):
        d = st.date_input("Date", value=date.today())
        cat = st.text_input("Category", value="General")
        desc = st.text_input("Description", value="")
        amt = st.number_input("Amount (£)", min_value=0.0, step=1.0)
        submitted = st.form_submit_button("Add Expense")
        if submitted and amt > 0:
            add_expense(d, cat, desc, amt)
            st.success("Saved.")

    st.divider()
    df = fetch_df("SELECT date, category, description, amount FROM expenses ORDER BY date DESC LIMIT 500")
    if df.empty:
        st.info("No expenses yet.")
        return

    df['date'] = pd.to_datetime(df['date'])
    st.dataframe(df, use_container_width=True)

    st.subheader("Monthly Spend by Category")
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    roll = df.groupby(['month','category'])['amount'].sum().reset_index()
    fig = px.bar(roll, x='month', y='amount', color='category', barmode='stack')
    st.plotly_chart(fig, use_container_width=True)

def page_forecast():
    st.title("Forecast")
    st.caption("Project cash with latest rates and investments using historical CAGR (or your override).")

    months = st.slider("Months ahead", 3, 36, 12)
    override_rate = st.number_input("Override investment annual rate (%) (optional)", value=0.0, step=0.5)
    use_override = st.checkbox("Use override for all investment accounts", value=False)

    lb = latest_balances_df()
    if lb.empty:
        st.info("Add some balances first.")
        return

    forecasts = []
    for _, row in lb.iterrows():
        acc_id = int(row['account_id'])
        series = account_series(acc_id)
        if row['type'] in ['bank','savings']:
            latest_apr = latest_rate_for_account(acc_id)
            fc = forecast_cash(float(row['last_balance'] or 0.0), latest_apr, months)
        else:
            fc = forecast_investment(series, months, (override_rate/100.0) if use_override else None)
        if not fc.empty:
            fc['account'] = row['name']
            forecasts.append(fc)

    if not forecasts:
        st.info("Not enough data to forecast.")
        return

    big = pd.concat(forecasts, ignore_index=True)
    fig = px.line(big, x='date', y='value', color='account', labels={'value':'£'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Projected Total")
    totals = big.groupby('date')['value'].sum().reset_index()
    fig2 = px.line(totals, x='date', y='value', labels={'value':'£'})
    st.plotly_chart(fig2, use_container_width=True)

def page_export():
    st.title("Data Export")
    st.caption("Download CSVs for backup or offline analysis.")

    conn = get_conn()
    balances = pd.read_sql_query("SELECT * FROM balances", conn)
    accounts = pd.read_sql_query("SELECT * FROM accounts", conn)
    rates = pd.read_sql_query("SELECT * FROM rates", conn)
    expenses = pd.read_sql_query("SELECT * FROM expenses", conn)
    conn.close()

    for name, df in {
        'accounts.csv': accounts,
        'balances.csv': balances,
        'rates.csv': rates,
        'expenses.csv': expenses,
    }.items():
        st.download_button(
            label=f"Download {name}",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=name,
            mime='text/csv'
        )

# --------------------------
# Main
# --------------------------

def main():
    st.set_page_config(page_title="Finance Command Center", layout="wide")
    init_db()

    page = sidebar_nav()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Accounts":
        page_accounts()
    elif page == "Balances & Rates":
        page_balances_rates()
    elif page == "Expenses":
        page_expenses()
    elif page == "Forecast":
        page_forecast()
    elif page == "Data Export":
        page_export()

if __name__ == "__main__":
    main()
