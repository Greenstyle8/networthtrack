"""
app.py — Streamlit Personal Finance Command Center (easy version)

What’s new vs your current one:
- CSV import now matches account names case/space-insensitively and can auto‑create missing accounts
- Manage accounts: rename, change type/currency, delete ALL data, or delete account
- Reports (PDF): download a clean PDF with a table of accounts + allocation pie + net worth line
- More stable database connection on Streamlit Cloud

How to deploy (no coding):
1) Replace your GitHub repo files with these (overwrite app.py and requirements.txt).
2) On Streamlit Cloud, click Redeploy (or New app if first time).
3) In the app, go to Accounts → add/rename as needed, then Balances & Rates → import CSV.
"""
import sqlite3
from datetime import date
from typing import Optional, Tuple
from io import BytesIO
import tempfile

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# --------------------------
# Settings
# --------------------------
DB_PATH = "finance.db"

# Utility: normalize names for matching (case/space tolerant)
def _norm(s: str) -> str:
    return "" if s is None else " ".join(str(s).strip().lower().split())

# --------------------------
# DB Helpers
# --------------------------
@st.cache_resource
def get_conn():
    # Shared connection; tuned pragmas reduce 'database is locked' issues
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
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
            apr REAL NOT NULL,
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
            amount REAL NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS contributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            amount REAL NOT NULL,
            FOREIGN KEY(account_id) REFERENCES accounts(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()

# --------------------------
# Data Access
# --------------------------
def fetch_df(query: str, params: Tuple = ()): 
    conn = get_conn()
    return pd.read_sql_query(query, conn, params=params)

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

def update_account(account_id: int, name: str, type_: str, currency: str, target_allocation: Optional[float]):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE accounts SET name=?, type=?, currency=?, target_allocation=? WHERE id=?",
        (name.strip(), type_.strip(), currency.strip(), target_allocation, account_id)
    )
    conn.commit()

def delete_account_data(account_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM balances WHERE account_id=?", (account_id,))
    cur.execute("DELETE FROM rates WHERE account_id=?", (account_id,))
    cur.execute("DELETE FROM contributions WHERE account_id=?", (account_id,))
    conn.commit()

def delete_account(account_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM accounts WHERE id=?", (account_id,))
    conn.commit()

def get_accounts_norm_map():
    df = fetch_df("SELECT id, name FROM accounts")
    return { _norm(n): i for i, n in zip(df['id'], df['name']) } if not df.empty else {}

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

def add_expense(d: date, category: str, description: str, amount: float):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO expenses(date, category, description, amount) VALUES(?,?,?,?)",
        (d.isoformat(), category.strip(), description.strip(), float(amount)),
    )
    conn.commit()

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
    q = "SELECT date, balance FROM balances WHERE account_id=? ORDER BY date"
    df = fetch_df(q, (account_id,))
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'balance': 'value'})
    return df

def latest_rate_for_account(account_id: int) -> Optional[float]:
    q = "SELECT apr FROM rates WHERE account_id=? ORDER BY date DESC LIMIT 1"
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
            "Reports (PDF)",
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
    st.caption("Add, edit, or delete accounts. Rename to fix CSV mismatches.")

    # Add / update
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
    st.subheader("Manage Existing Accounts")
    df = fetch_df("SELECT id, name, type, currency, target_allocation FROM accounts ORDER BY name")
    if df.empty:
        st.info("No accounts yet.")
        return

    edit_id = st.selectbox(
        "Choose account to edit",
        options=df['id'],
        format_func=lambda i: df.loc[df['id']==i, 'name'].values[0]
    )
    row = df[df['id']==edit_id].iloc[0]
    col1, col2 = st.columns(2)

    with col1:
        new_name = st.text_input("Rename", value=row['name'])
        new_type = st.selectbox("Type", ['bank','savings','isa','investment','pension'],
                                index=['bank','savings','isa','investment','pension'].index(row['type']))
        new_curr = st.selectbox("Currency", ['GBP','USD','EUR'],
                                index=['GBP','USD','EUR'].index(row['currency']))
        new_alloc = st.number_input("Target allocation %", min_value=0.0, max_value=100.0,
                                    value=float((row['target_allocation'] or 0)*100))
        if st.button("Save changes"):
            update_account(int(edit_id), new_name, new_type, new_curr, (new_alloc/100.0) if new_alloc else None)
            st.success("Updated. (If the table doesn't refresh, click Rerun in Streamlit.)")

    with col2:
        if st.button("Delete ALL data for this account"):
            delete_account_data(int(edit_id))
            st.warning("All balances/rates for this account deleted.")
        if st.button("Delete account entirely"):
            delete_account(int(edit_id))
            st.error("Account deleted. (Switch tabs or rerun to refresh.)")

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
    st.caption("Upload balances with columns: account,date,balance and (optional) rate. Names are matched case/space‑insensitively; missing accounts can be auto‑created.")
    create_missing = st.checkbox("Auto-create missing accounts as 'savings'", value=True)

    up = st.file_uploader("CSV file", type=['csv'])
    if up:
        try:
            dfcsv = pd.read_csv(up)
            needed = {'account','date','balance'}
            cols = {c.lower(): c for c in dfcsv.columns}
            if not needed.issubset(set(cols.keys())):
                st.error("CSV must include account,date,balance")
            else:
                norm_map = get_accounts_norm_map()
                created = updated = skipped = 0
                for _, r in dfcsv.iterrows():
                    acc_name_raw = str(r[cols['account']])
                    acc_norm = _norm(acc_name_raw)
                    acc_id = norm_map.get(acc_norm)
                    if acc_id is None:
                        if create_missing and acc_name_raw.strip():
                            upsert_account(acc_name_raw, 'savings', 'GBP', None)
                            norm_map = get_accounts_norm_map()
                            acc_id = norm_map.get(acc_norm)
                            created += 1
                        else:
                            skipped += 1
                            continue
                    d = pd.to_datetime(r[cols['date']]).date()
                    bal = float(r[cols['balance']])
                    add_balance(int(acc_id), d, bal)
                    updated += 1
                    if 'rate' in cols and not pd.isna(r[cols['rate']]):
                        rv = float(r[cols['rate']])
                        apr = rv/100.0 if rv > 1.0 else rv
                        add_rate(int(acc_id), d, apr)
                st.success(f"CSV processed. Rows added/updated: {updated}. Accounts created: {created}. Skipped: {skipped}.")
        except Exception as e:
            st.exception(e)

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

def page_reports():
    st.title("Reports (PDF)")
    st.caption("Download a clean PDF snapshot of your finances (balances + charts).")

    lb = latest_balances_df()
    if lb.empty:
        st.info("Add some balances first.")
        return

    # Build charts and export as PNG with kaleido
    imgs = {}
    nw = net_worth_series()
    if not nw.empty:
        fig_nw = px.line(nw, x='date', y='net_worth', title='Net Worth Over Time', labels={'net_worth':'£'})
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig_nw.write_image(tmp.name, scale=2)
            imgs['networth'] = tmp.name

    alloc = allocation_snapshot()
    if not alloc.empty:
        fig_alloc = px.pie(alloc, names='type', values='last_balance', hole=0.4, title='Allocation Snapshot')
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig_alloc.write_image(tmp.name, scale=2)
            imgs['alloc'] = tmp.name

    # Create PDF with reportlab
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.units import cm

    buf = BytesIO()
    c = Canvas(buf, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, height - 2*cm, "Personal Finance Snapshot")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, height - 2.6*cm, "Generated by your Streamlit app")

    y = height - 3.2*cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Latest Balances"); y -= 0.6*cm
    c.setFont("Helvetica", 10)
    for _, r in lb.fillna(0).iterrows():
        line = f"{r['name']}  ({str(r['type']).upper()})   £{float(r['last_balance'] or 0):,.0f}"
        c.drawString(2*cm, y, line)
        y -= 0.5*cm
        if y < 4*cm:
            c.showPage(); y = height - 3*cm

    if 'alloc' in imgs:
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, height - 2.5*cm, "Charts")
        c.drawImage(imgs['alloc'], 2*cm, height - 15*cm, width=12*cm, preserveAspectRatio=True, mask='auto')
    if 'networth' in imgs:
        c.drawImage(imgs['networth'], 2*cm, height - 27*cm, width=12*cm, preserveAspectRatio=True, mask='auto')

    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()

    st.download_button("Download PDF report", data=pdf_bytes, file_name="finance_report.pdf", mime="application/pdf")
    st.success("PDF generated.")

def page_export():
    st.title("Data Export")
    st.caption("Download CSVs for backup or offline analysis.")

    conn = get_conn()
    balances = pd.read_sql_query("SELECT * FROM balances", conn)
    accounts = pd.read_sql_query("SELECT * FROM accounts", conn)
    rates = pd.read_sql_query("SELECT * FROM rates", conn)
    expenses = pd.read_sql_query("SELECT * FROM expenses", conn)

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
def sidebar_and_route():
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
    elif page == "Reports (PDF)":
        page_reports()
    elif page == "Data Export":
        page_export()

if __name__ == "__main__":
    sidebar_and_route()
