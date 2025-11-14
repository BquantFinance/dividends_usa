import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import ftplib
from io import BytesIO, StringIO
import requests

# Page configuration
st.set_page_config(
    page_title="US Dividend Analyzer | @Gsnchez",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f3a 100%);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #1e2847 0%, #2a3454 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #3a4766;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        border: 1px solid #00d4ff;
        box-shadow: 0 12px 48px rgba(0, 212, 255, 0.2);
        transform: translateY(-2px);
    }
    
    h1 {
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        letter-spacing: -1px;
        font-size: 3.5rem !important;
    }
    
    h2 {
        color: #00d4ff;
        font-weight: 600;
        font-size: 2rem !important;
    }
    
    h3 {
        color: #00a8cc;
        font-weight: 500;
        font-size: 1.5rem !important;
    }
    
    h4 {
        color: #0088aa;
        font-weight: 500;
    }
    
    .search-box {
        background: linear-gradient(135deg, #1e2847 0%, #2a3454 100%);
        padding: 30px;
        border-radius: 20px;
        border: 2px solid #00d4ff;
        box-shadow: 0 20px 60px rgba(0, 212, 255, 0.3);
        margin: 20px 0;
    }
    
    .stock-card {
        background: linear-gradient(135deg, #1a2332 0%, #242f45 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #3a4766;
        margin: 15px 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #a0b0c0;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .highlight-card {
        background: linear-gradient(135deg, #1a2f3a 0%, #2a3f4a 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 4px solid #00ff88;
        margin: 10px 0;
    }
    
    .aristocrat-badge {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
        color: #000;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
        margin-left: 10px;
    }
    
    .sp500-badge {
        background: linear-gradient(135deg, #00d4ff 0%, #0088aa 100%);
        color: #fff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
        margin-left: 10px;
    }
    
    .footer {
        text-align: center;
        padding: 30px;
        color: #707070;
        border-top: 1px solid #3a4766;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_nasdaq_tickers():
    """Get NASDAQ listed tickers"""
    try:
        ftp = ftplib.FTP("ftp.nasdaqtrader.com")
        ftp.login()
        ftp.cwd("SymbolDirectory")
        
        r = BytesIO()
        ftp.retrbinary('RETR nasdaqlisted.txt', r.write)
        
        info = r.getvalue().decode()
        splits = info.split("|")
        tickers = [x for x in splits if "\r\n" in x]
        tickers = [x.split("\r\n")[1] for x in tickers if "NASDAQ" not in x != "\r\n"]
        tickers = [ticker for ticker in tickers if "File" not in ticker]
        
        ftp.close()
        return set(tickers)
    except Exception as e:
        st.warning(f"Could not fetch NASDAQ tickers: {e}")
        return set()

@st.cache_data(ttl=86400)
def get_nyse_amex_tickers():
    """Get NYSE and AMEX listed tickers"""
    try:
        ftp = ftplib.FTP("ftp.nasdaqtrader.com")
        ftp.login()
        ftp.cwd("SymbolDirectory")
        
        r = BytesIO()
        ftp.retrbinary('RETR otherlisted.txt', r.write)
        
        r.seek(0)
        df = pd.read_csv(r, sep="|")
        
        tickers_df = df[['ACT Symbol', 'Exchange']].copy()
        tickers_df.columns = ['ticker', 'exchange']
        tickers_df = tickers_df.dropna()
        tickers_df['ticker'] = tickers_df['ticker'].astype(str).str.strip()
        tickers_df = tickers_df[tickers_df['ticker'].str.len() > 0]
        tickers_df = tickers_df[~tickers_df['ticker'].str.contains('Symbol')]
        
        ftp.close()
        return tickers_df
    except Exception as e:
        st.warning(f"Could not fetch NYSE/AMEX tickers: {e}")
        return pd.DataFrame(columns=['ticker', 'exchange'])

@st.cache_data(ttl=86400)
def get_sp500_data():
    """Scrape S&P 500 companies from Wikipedia with proper headers"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        # Use requests with user agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML tables
        tables = pd.read_html(StringIO(response.text))
        sp500_df = tables[0]
        
        # Clean column names - the first table has the S&P 500 constituents
        sp500_df.columns = sp500_df.columns.str.strip()
        
        # Map to standard names
        column_mapping = {}
        for col in sp500_df.columns:
            if 'Symbol' in col or 'Ticker' in col:
                column_mapping[col] = 'ticker'
            elif 'Security' in col or 'Company' in col:
                column_mapping[col] = 'company'
            elif 'GICS Sector' in col or 'Sector' in col:
                column_mapping[col] = 'sector'
            elif 'GICS Sub' in col or 'Sub-Industry' in col or 'Industry' in col:
                column_mapping[col] = 'industry'
        
        sp500_df = sp500_df.rename(columns=column_mapping)
        
        # Keep only relevant columns
        cols_to_keep = ['ticker', 'company', 'sector', 'industry']
        sp500_df = sp500_df[[col for col in cols_to_keep if col in sp500_df.columns]]
        
        # Clean ticker symbols (remove any trailing characters)
        if 'ticker' in sp500_df.columns:
            sp500_df['ticker'] = sp500_df['ticker'].astype(str).str.strip()
        
        return sp500_df
    except Exception as e:
        st.warning(f"Could not fetch S&P 500 data: {e}")
        return pd.DataFrame(columns=['ticker', 'company', 'sector', 'industry'])

@st.cache_data(ttl=86400)
def get_dividend_aristocrats():
    """Scrape Dividend Aristocrats from Wikipedia with proper headers"""
    try:
        url = "https://en.wikipedia.org/wiki/S%26P_500_Dividend_Aristocrats"
        
        # Use requests with user agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML tables
        tables = pd.read_html(StringIO(response.text))
        
        # Find the table with aristocrats (look for Symbol or Ticker column)
        aristocrats_set = set()
        for table in tables:
            # Check if table has ticker/symbol column
            for col in table.columns:
                col_str = str(col).lower()
                if 'symbol' in col_str or 'ticker' in col_str:
                    tickers = table[col].dropna().astype(str).str.strip().tolist()
                    aristocrats_set.update([t for t in tickers if t and t != 'nan' and len(t) <= 5])
                    break
        
        return aristocrats_set
    except Exception as e:
        st.warning(f"Could not fetch Dividend Aristocrats: {e}")
        return set()

@st.cache_data
def load_data():
    """Load dividend data"""
    csv_filename = 'all_usa_dividends_complete_20251114_1523.csv'
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        st.error(f"❌ Could not find {csv_filename}")
        st.stop()
    
    df['ex_dividend_date'] = pd.to_datetime(df['ex_dividend_date'])
    df['payout_date'] = pd.to_datetime(df['payout_date'])
    df['year'] = df['ex_dividend_date'].dt.year
    df['quarter'] = df['ex_dividend_date'].dt.quarter
    df['month'] = df['ex_dividend_date'].dt.month
    return df

@st.cache_data
def enrich_dividend_data(df, nasdaq_tickers, nyse_amex_df, sp500_df, aristocrats):
    """Enrich dividend data with exchange and index information"""
    
    # Add exchange information
    def get_exchange(ticker):
        if ticker in nasdaq_tickers:
            return 'NASDAQ'
        nyse_amex_match = nyse_amex_df[nyse_amex_df['ticker'] == ticker]
        if not nyse_amex_match.empty:
            exchange = nyse_amex_match.iloc[0]['exchange']
            if exchange == 'N':
                return 'NYSE'
            elif exchange == 'A':
                return 'AMEX'
        return 'Unknown'
    
    # Create ticker metadata
    unique_tickers = df['ticker'].unique()
    ticker_info = pd.DataFrame({'ticker': unique_tickers})
    
    ticker_info['exchange'] = ticker_info['ticker'].apply(get_exchange)
    ticker_info['is_sp500'] = ticker_info['ticker'].isin(sp500_df['ticker']) if 'ticker' in sp500_df.columns else False
    ticker_info['is_aristocrat'] = ticker_info['ticker'].isin(aristocrats)
    
    # Merge with S&P 500 data
    if 'ticker' in sp500_df.columns and len(sp500_df) > 0:
        ticker_info = ticker_info.merge(
            sp500_df[['ticker', 'company', 'sector', 'industry']], 
            on='ticker', 
            how='left'
        )
    else:
        ticker_info['company'] = None
        ticker_info['sector'] = None
        ticker_info['industry'] = None
    
    return ticker_info

@st.cache_data
def get_stock_metrics(df, ticker):
    """Calculate comprehensive metrics for a stock"""
    ticker_data = df[df['ticker'] == ticker].sort_values('ex_dividend_date')
    
    if len(ticker_data) == 0:
        return None
    
    total_payments = len(ticker_data)
    latest_amount = ticker_data.iloc[-1]['amount']
    first_date = ticker_data['ex_dividend_date'].min()
    last_date = ticker_data['ex_dividend_date'].max()
    years_active = (last_date - first_date).days / 365.25
    
    growth_records = ticker_data[ticker_data['pct_change'].notna()]
    avg_growth = growth_records['pct_change'].mean() if len(growth_records) > 0 else 0
    
    consecutive_growth = 0
    max_consecutive_growth = 0
    for _, row in ticker_data[::-1].iterrows():
        if pd.notna(row['pct_change']) and row['pct_change'] > 0:
            consecutive_growth += 1
            max_consecutive_growth = max(max_consecutive_growth, consecutive_growth)
        else:
            consecutive_growth = 0
    
    annual_frequency = total_payments / max(years_active, 1)
    total_paid = ticker_data['amount'].sum()
    
    if len(ticker_data) > 1:
        first_amount = ticker_data.iloc[0]['amount']
        growth_rate = ((latest_amount / first_amount) ** (1/years_active) - 1) * 100 if first_amount > 0 else 0
    else:
        growth_rate = 0
    
    return {
        'total_payments': total_payments,
        'latest_amount': latest_amount,
        'first_date': first_date,
        'last_date': last_date,
        'years_active': years_active,
        'avg_growth': avg_growth,
        'consecutive_growth': max_consecutive_growth,
        'annual_frequency': annual_frequency,
        'total_paid': total_paid,
        'cagr': growth_rate,
        'data': ticker_data
    }

# Load all data
with st.spinner('🚀 Loading data...'):
    df = load_data()
    nasdaq_tickers = get_nasdaq_tickers()
    nyse_amex_df = get_nyse_amex_tickers()
    sp500_df = get_sp500_data()
    aristocrats = get_dividend_aristocrats()
    ticker_info = enrich_dividend_data(df, nasdaq_tickers, nyse_amex_df, sp500_df, aristocrats)

# Header with branding
st.markdown("""
    <div style='text-align: center; padding: 40px 0 20px 0;'>
        <h1>💰 US DIVIDEND ANALYZER</h1>
        <p style='font-size: 20px; color: #a0b0c0; margin-top: 10px;'>
            Track dividend history, analyze growth, and discover the best dividend-paying stocks
        </p>
        <p style='font-size: 14px; color: #707070; margin-top: 10px;'>
            Made by <a href='https://bquantfinance.com' target='_blank' style='color: #00d4ff; text-decoration: none;'>@Gsnchez</a> | 
            <a href='https://bquantfinance.com' target='_blank' style='color: #00d4ff; text-decoration: none;'>bquantfinance.com</a>
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("---")

# Exchange filter
exchanges = ['All'] + sorted(ticker_info['exchange'].unique().tolist())
selected_exchange = st.sidebar.selectbox("Exchange", exchanges, help="Filter by stock exchange")

# S&P 500 filter
sp500_filter = st.sidebar.checkbox("S&P 500 Only", False, help="Show only S&P 500 companies")

# Aristocrats filter
aristocrats_filter = st.sidebar.checkbox("Dividend Aristocrats Only", False, help="Show only Dividend Aristocrats (25+ years)")

# Sector filter (only if S&P 500 selected)
if sp500_filter and 'sector' in ticker_info.columns:
    sectors = ['All'] + sorted(ticker_info['sector'].dropna().unique().tolist())
    selected_sector = st.sidebar.selectbox("Sector", sectors, help="Filter by GICS sector")
    
    if selected_sector != 'All':
        industries = ['All'] + sorted(ticker_info[ticker_info['sector'] == selected_sector]['industry'].dropna().unique().tolist())
        selected_industry = st.sidebar.selectbox("Industry", industries, help="Filter by GICS industry")
    else:
        selected_industry = 'All'
else:
    selected_sector = 'All'
    selected_industry = 'All'

# Apply filters
filtered_tickers = ticker_info.copy()

if selected_exchange != 'All':
    filtered_tickers = filtered_tickers[filtered_tickers['exchange'] == selected_exchange]

if sp500_filter:
    filtered_tickers = filtered_tickers[filtered_tickers['is_sp500'] == True]

if aristocrats_filter:
    filtered_tickers = filtered_tickers[filtered_tickers['is_aristocrat'] == True]

if selected_sector != 'All' and 'sector' in filtered_tickers.columns:
    filtered_tickers = filtered_tickers[filtered_tickers['sector'] == selected_sector]

if selected_industry != 'All' and 'industry' in filtered_tickers.columns:
    filtered_tickers = filtered_tickers[filtered_tickers['industry'] == selected_industry]

available_tickers = sorted(filtered_tickers['ticker'].tolist())

# Show aggregated stats
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Filtered Stats")
st.sidebar.markdown(f"""
<div class='highlight-card'>
    <p>📈 <strong>Stocks:</strong> {len(available_tickers):,}</p>
    <p>💰 <strong>S&P 500:</strong> {filtered_tickers['is_sp500'].sum()}</p>
    <p>👑 <strong>Aristocrats:</strong> {filtered_tickers['is_aristocrat'].sum()}</p>
</div>
""", unsafe_allow_html=True)

# Database overview
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Database Overview")
total_tickers = df['ticker'].nunique()
total_sp500 = ticker_info['is_sp500'].sum()
total_aristocrats = ticker_info['is_aristocrat'].sum()

st.sidebar.markdown(f"""
<div class='highlight-card'>
    <p>📊 <strong>Total Stocks:</strong> {total_tickers:,}</p>
    <p>💰 <strong>Total Payments:</strong> {len(df):,}</p>
    <p>📅 <strong>Date Range:</strong> {df['year'].min()}-{df['year'].max()}</p>
    <p>🏢 <strong>S&P 500:</strong> {total_sp500}</p>
    <p>👑 <strong>Aristocrats:</strong> {total_aristocrats}</p>
</div>
""", unsafe_allow_html=True)

# Exchange breakdown
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏦 By Exchange")
exchange_counts = ticker_info['exchange'].value_counts()
for exchange, count in exchange_counts.items():
    st.sidebar.markdown(f"**{exchange}**: {count}")

# Main Search Section
st.markdown("<div class='search-box'>", unsafe_allow_html=True)
st.markdown("## 🔍 Search Dividend History")

col_search1, col_search2, col_search3 = st.columns([2, 2, 1])

with col_search1:
    if len(available_tickers) == 0:
        st.warning("⚠️ No tickers match your filters")
        selected_ticker = None
    else:
        selected_ticker = st.selectbox(
            "Select Stock Ticker",
            options=[''] + available_tickers,
            help=f"Search from {len(available_tickers):,} dividend-paying stocks",
            key='main_search'
        )

with col_search2:
    if selected_ticker:
        stock_metrics = get_stock_metrics(df, selected_ticker)
        ticker_meta = ticker_info[ticker_info['ticker'] == selected_ticker].iloc[0]
        
        if stock_metrics:
            badges = ""
            if ticker_meta['is_sp500']:
                badges += "<span class='sp500-badge'>S&P 500</span>"
            if ticker_meta['is_aristocrat']:
                badges += "<span class='aristocrat-badge'>👑 ARISTOCRAT</span>"
            
            st.markdown(f"### 📌 {selected_ticker} {badges}", unsafe_allow_html=True)
            
            company_info = f"<p style='color: #a0b0c0;'>"
            company_info += f"<strong>Exchange:</strong> {ticker_meta['exchange']} | "
            company_info += f"Latest: <span style='color: #00ff88; font-size: 24px; font-weight: 700;'>${stock_metrics['latest_amount']:.3f}</span></p>"
            
            if pd.notna(ticker_meta.get('sector')):
                company_info += f"<p style='color: #909090; font-size: 12px;'>{ticker_meta['sector']} - {ticker_meta['industry']}</p>"
            
            st.markdown(company_info, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Display stock analysis
if selected_ticker and 'stock_metrics' in locals() and stock_metrics:
    st.markdown("---")
    
    # Key Metrics
    st.markdown("### 📊 Key Dividend Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Payments", f"{stock_metrics['total_payments']:,}")
    
    with col2:
        st.metric("Latest Dividend", f"${stock_metrics['latest_amount']:.3f}", 
                 delta=f"{stock_metrics['avg_growth']:.1f}% avg growth")
    
    with col3:
        st.metric("Years Active", f"{stock_metrics['years_active']:.1f}")
    
    with col4:
        st.metric("Growth Streak", f"{stock_metrics['consecutive_growth']} yrs")
    
    with col5:
        freq_label = "Monthly" if stock_metrics['annual_frequency'] >= 11 else "Quarterly" if stock_metrics['annual_frequency'] >= 3.5 else "Annual"
        st.metric("Frequency", freq_label, delta=f"{stock_metrics['annual_frequency']:.1f}/yr")
    
    st.markdown("---")
    
    # Main Chart
    st.markdown("### 📈 Dividend History Timeline")
    
    ticker_data = stock_metrics['data']
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        subplot_titles=('Dividend Amount Over Time', 'Year-over-Year Growth Rate'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(
            x=ticker_data['ex_dividend_date'],
            y=ticker_data['amount'],
            mode='lines+markers',
            name='Dividend Amount',
            line=dict(color='#00d4ff', width=3),
            marker=dict(size=8, color='#00ff88', line=dict(color='#00d4ff', width=2)),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.2)',
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Amount: $%{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    if len(ticker_data) > 1:
        z = np.polyfit(range(len(ticker_data)), ticker_data['amount'], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=ticker_data['ex_dividend_date'],
                y=p(range(len(ticker_data))),
                mode='lines',
                name='Trend',
                line=dict(color='#ff6b6b', width=2, dash='dash'),
                hovertemplate='<b>Trend:</b> $%{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    growth_data = ticker_data[ticker_data['pct_change'].notna()]
    colors = ['#00ff88' if x > 0 else '#ff6b6b' for x in growth_data['pct_change']]
    
    fig.add_trace(
        go.Bar(
            x=growth_data['ex_dividend_date'],
            y=growth_data['pct_change'],
            name='Growth Rate',
            marker_color=colors,
            hovertemplate='<b>%{x|%b %Y}</b><br>Growth: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=700,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(30, 40, 71, 0.8)',
            bordercolor='#3a4766',
            borderwidth=1
        )
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', title_text="Date", row=2, col=1)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', title_text="Amount ($)", row=1, col=1)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', title_text="Growth (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analytics Section
    st.markdown("---")
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 💡 Dividend Insights")
        
        avg_amount = ticker_data['amount'].mean()
        median_amount = ticker_data['amount'].median()
        volatility = ticker_data['amount'].std()
        
        st.markdown(f"""
        <div class='stock-card'>
            <h4 style='color: #00d4ff; margin-bottom: 15px;'>Statistical Analysis</h4>
            <p>📊 <strong>Average Dividend:</strong> <span style='color: #00ff88;'>${avg_amount:.3f}</span></p>
            <p>📊 <strong>Median Dividend:</strong> <span style='color: #00ff88;'>${median_amount:.3f}</span></p>
            <p>📊 <strong>Volatility (Std Dev):</strong> <span style='color: #00ff88;'>${volatility:.3f}</span></p>
            <p>📈 <strong>CAGR:</strong> <span style='color: #00ff88;'>{stock_metrics['cagr']:.2f}%</span></p>
            <p>💰 <strong>Total Paid (History):</strong> <span style='color: #00ff88;'>${stock_metrics['total_paid']:.2f}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📅 Payment Frequency Analysis")
        yearly_counts = ticker_data.groupby('year').size()
        
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Bar(
            x=yearly_counts.index,
            y=yearly_counts.values,
            marker=dict(color=yearly_counts.values, colorscale='Viridis', showscale=False),
            hovertemplate='<b>%{x}</b><br>Payments: %{y}<extra></extra>'
        ))
        
        fig_freq.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            xaxis_title="Year",
            yaxis_title="Payments per Year",
            showlegend=False
        )
        fig_freq.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
        fig_freq.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig_freq, use_container_width=True)
    
    with col_right:
        st.markdown("### 📋 Recent Dividend History")
        
        recent_payments = ticker_data.tail(12)[::-1][['ex_dividend_date', 'payout_date', 'amount', 'pct_change']].copy()
        recent_payments['ex_dividend_date'] = recent_payments['ex_dividend_date'].dt.strftime('%Y-%m-%d')
        recent_payments['payout_date'] = recent_payments['payout_date'].dt.strftime('%Y-%m-%d')
        
        def format_growth(val):
            if pd.isna(val):
                return '-'
            color = '#00ff88' if val > 0 else '#ff6b6b' if val < 0 else '#ffffff'
            return f'<span style="color: {color}; font-weight: 600;">{val:.2f}%</span>'
        
        recent_payments['amount_fmt'] = recent_payments['amount'].apply(lambda x: f'${x:.3f}')
        recent_payments['growth_fmt'] = recent_payments['pct_change'].apply(format_growth)
        
        table_html = "<div class='dataframe'><table style='width: 100%; border-collapse: collapse;'>"
        table_html += """
        <thead>
            <tr style='background: rgba(0, 212, 255, 0.1); border-bottom: 2px solid #00d4ff;'>
                <th style='padding: 12px; text-align: left; color: #a0b0c0;'>Ex-Date</th>
                <th style='padding: 12px; text-align: left; color: #a0b0c0;'>Pay Date</th>
                <th style='padding: 12px; text-align: right; color: #a0b0c0;'>Amount</th>
                <th style='padding: 12px; text-align: right; color: #a0b0c0;'>Growth</th>
            </tr>
        </thead>
        <tbody>
        """
        
        for _, row in recent_payments.iterrows():
            table_html += f"""
            <tr style='border-bottom: 1px solid rgba(128, 128, 128, 0.2);'>
                <td style='padding: 10px; color: #ffffff;'>{row['ex_dividend_date']}</td>
                <td style='padding: 10px; color: #ffffff;'>{row['payout_date']}</td>
                <td style='padding: 10px; text-align: right; color: #00ff88; font-weight: 600;'>{row['amount_fmt']}</td>
                <td style='padding: 10px; text-align: right;'>{row['growth_fmt']}</td>
            </tr>
            """
        
        table_html += "</tbody></table></div>"
        st.markdown(table_html, unsafe_allow_html=True)
        
        csv = ticker_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Full History (CSV)",
            data=csv,
            file_name=f"{selected_ticker}_dividend_history.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer with branding
st.markdown("""
    <div class='footer'>
        <p style='font-size: 18px;'><strong>💰 US Dividend Analyzer</strong></p>
        <p style='font-size: 14px; margin-top: 10px;'>
            Tracking {total:,} dividend payments across {tickers:,} stocks since {min_year}
        </p>
        <p style='font-size: 14px; margin-top: 15px;'>
            Made by <a href='https://twitter.com/Gsnchez' target='_blank' style='color: #00d4ff; text-decoration: none;'>@Gsnchez</a> | 
            <a href='https://bquantfinance.com' target='_blank' style='color: #00d4ff; text-decoration: none;'>bquantfinance.com</a>
        </p>
        <p style='font-size: 12px; margin-top: 10px; color: #505060;'>
            Built with Streamlit & Plotly | Data updated: November 2025
        </p>
    </div>
""".format(total=len(df), tickers=total_tickers, min_year=df['year'].min()), unsafe_allow_html=True)
