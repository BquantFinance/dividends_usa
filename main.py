import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import ftplib
from io import BytesIO, StringIO

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
def enrich_dividend_data(df, nasdaq_tickers, nyse_amex_df):
    """Enrich dividend data with exchange information"""
    
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
    ticker_info = enrich_dividend_data(df, nasdaq_tickers, nyse_amex_df)

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

# Apply filters
filtered_tickers = ticker_info.copy()

if selected_exchange != 'All':
    filtered_tickers = filtered_tickers[filtered_tickers['exchange'] == selected_exchange]

available_tickers = sorted(filtered_tickers['ticker'].tolist())

# Show aggregated stats
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Filtered Stats")
st.sidebar.markdown(f"""
<div class='highlight-card'>
    <p>📈 <strong>Stocks:</strong> {len(available_tickers):,}</p>
</div>
""", unsafe_allow_html=True)

# Database overview
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Database Overview")
total_tickers = df['ticker'].nunique()

st.sidebar.markdown(f"""
<div class='highlight-card'>
    <p>📊 <strong>Total Stocks:</strong> {total_tickers:,}</p>
    <p>💰 <strong>Total Payments:</strong> {len(df):,}</p>
    <p>📅 <strong>Date Range:</strong> {df['year'].min()}-{df['year'].max()}</p>
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
            st.markdown(f"### 📌 {selected_ticker}", unsafe_allow_html=True)
            
            company_info = f"<p style='color: #a0b0c0;'>"
            company_info += f"<strong>Exchange:</strong> {ticker_meta['exchange']} | "
            company_info += f"Latest: <span style='color: #00ff88; font-size: 24px; font-weight: 700;'>${stock_metrics['latest_amount']:.3f}</span></p>"
            
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
    
    # Date Range Filter
    st.markdown("#### 📅 Select Time Period")
    
    # Quick selection buttons
    col_q1, col_q2, col_q3, col_q4, col_q5, col_q6 = st.columns(6)
    
    min_date = ticker_data['ex_dividend_date'].min().date()
    max_date = ticker_data['ex_dividend_date'].max().date()
    today = datetime.now().date()
    
    # Initialize session state for dates if not exists
    if 'start_date' not in st.session_state:
        st.session_state.start_date = min_date
    if 'end_date' not in st.session_state:
        st.session_state.end_date = max_date
    
    # DEBUG: Show current session state
    st.markdown(f"""
    <div style='background: rgba(255,0,0,0.1); padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 12px;'>
        🔍 <strong>DEBUG:</strong> Session state dates: {st.session_state.start_date} to {st.session_state.end_date}
    </div>
    """, unsafe_allow_html=True)
    
    with col_q1:
        if st.button("📅 1Y", use_container_width=True, help="Last 1 year", key='btn_1y'):
            new_start = max(min_date, (today - timedelta(days=365)))
            st.session_state.start_date = new_start
            st.session_state.end_date = max_date
            st.write(f"DEBUG: 1Y clicked, setting {new_start} to {max_date}")  # Temp debug
            st.rerun()
    
    with col_q2:
        if st.button("📅 3Y", use_container_width=True, help="Last 3 years", key='btn_3y'):
            new_start = max(min_date, (today - timedelta(days=365*3)))
            st.session_state.start_date = new_start
            st.session_state.end_date = max_date
            st.write(f"DEBUG: 3Y clicked, setting {new_start} to {max_date}")  # Temp debug
            st.rerun()
    
    with col_q3:
        if st.button("📅 5Y", use_container_width=True, help="Last 5 years", key='btn_5y'):
            new_start = max(min_date, (today - timedelta(days=365*5)))
            st.session_state.start_date = new_start
            st.session_state.end_date = max_date
            st.write(f"DEBUG: 5Y clicked, setting {new_start} to {max_date}")  # Temp debug
            st.rerun()
    
    with col_q4:
        if st.button("📅 10Y", use_container_width=True, help="Last 10 years", key='btn_10y'):
            new_start = max(min_date, (today - timedelta(days=365*10)))
            st.session_state.start_date = new_start
            st.session_state.end_date = max_date
            st.write(f"DEBUG: 10Y clicked, setting {new_start} to {max_date}")  # Temp debug
            st.rerun()
    
    with col_q5:
        if st.button("📅 YTD", use_container_width=True, help="Year to date", key='btn_ytd'):
            new_start = max(min_date, datetime(today.year, 1, 1).date())
            st.session_state.start_date = new_start
            st.session_state.end_date = max_date
            st.write(f"DEBUG: YTD clicked, setting {new_start} to {max_date}")  # Temp debug
            st.rerun()
    
    with col_q6:
        if st.button("📅 ALL", use_container_width=True, help="Full history", key='btn_all'):
            st.session_state.start_date = min_date
            st.session_state.end_date = max_date
            st.write(f"DEBUG: ALL clicked, setting {min_date} to {max_date}")  # Temp debug
            st.rerun()
    
    # Date inputs
    col_date1, col_date2 = st.columns([2, 2])
    
    with col_date1:
        start_date_input = st.date_input(
            "Start Date",
            value=st.session_state.start_date,
            min_value=min_date,
            max_value=max_date,
            help="Filter dividend history from this date",
            key='date_input_start'
        )
    
    with col_date2:
        end_date_input = st.date_input(
            "End Date",
            value=st.session_state.end_date,
            min_value=min_date,
            max_value=max_date,
            help="Filter dividend history until this date",
            key='date_input_end'
        )
    
    # Only update session state if user manually changed the date inputs
    # (not if they were updated by buttons)
    if start_date_input != st.session_state.start_date:
        st.session_state.start_date = start_date_input
    if end_date_input != st.session_state.end_date:
        st.session_state.end_date = end_date_input
    
    # Use session state values for filtering (these are the source of truth)
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    
    # DEBUG: Show what dates are being used for filtering
    st.markdown(f"""
    <div style='background: rgba(0,255,0,0.1); padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 12px;'>
        🔍 <strong>DEBUG:</strong> Using dates for filtering: {start_date} to {end_date}
    </div>
    """, unsafe_allow_html=True)
    
    # Filter data based on selected date range
    ticker_data_filtered = ticker_data[
        (ticker_data['ex_dividend_date'].dt.date >= start_date) & 
        (ticker_data['ex_dividend_date'].dt.date <= end_date)
    ].copy()
    
    # DEBUG: Show filtering result
    st.markdown(f"""
    <div style='background: rgba(0,0,255,0.1); padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 12px;'>
        🔍 <strong>DEBUG:</strong> Filtered result: {len(ticker_data_filtered)} payments out of {len(ticker_data)} total
    </div>
    """, unsafe_allow_html=True)
    
    # ALWAYS set ticker_data_chart, even if empty
    ticker_data_chart = ticker_data_filtered
    
    if len(ticker_data_filtered) == 0:
        st.warning("⚠️ No dividend payments in the selected date range. Please adjust your dates.")
        st.stop()  # Stop execution if no data
    
    # Show filtered stats
    st.markdown(f"""
    <div style='padding: 10px; margin: 10px 0; background: rgba(0, 212, 255, 0.1); border-radius: 10px; border-left: 3px solid #00d4ff;'>
        <p style='margin: 0; color: #a0b0c0;'>
            📊 Showing <strong style='color: #00ff88;'>{len(ticker_data_filtered)}</strong> payments 
            from <strong style='color: #00d4ff;'>{start_date.strftime('%Y-%m-%d')}</strong> 
            to <strong style='color: #00d4ff;'>{end_date.strftime('%Y-%m-%d')}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        subplot_titles=('Dividend Amount Over Time', 'Year-over-Year Growth Rate'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(
            x=ticker_data_chart['ex_dividend_date'],
            y=ticker_data_chart['amount'],
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
    
    if len(ticker_data_chart) > 1:
        z = np.polyfit(range(len(ticker_data_chart)), ticker_data_chart['amount'], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=ticker_data_chart['ex_dividend_date'],
                y=p(range(len(ticker_data_chart))),
                mode='lines',
                name='Trend',
                line=dict(color='#ff6b6b', width=2, dash='dash'),
                hovertemplate='<b>Trend:</b> $%{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    growth_data = ticker_data_chart[ticker_data_chart['pct_change'].notna()]
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
        
        # Use filtered data for statistics
        avg_amount = ticker_data_chart['amount'].mean()
        median_amount = ticker_data_chart['amount'].median()
        volatility = ticker_data_chart['amount'].std()
        
        # Calculate CAGR for filtered period if we have enough data
        if len(ticker_data_chart) > 1:
            first_amount = ticker_data_chart.iloc[0]['amount']
            last_amount = ticker_data_chart.iloc[-1]['amount']
            years_span = (ticker_data_chart.iloc[-1]['ex_dividend_date'] - ticker_data_chart.iloc[0]['ex_dividend_date']).days / 365.25
            filtered_cagr = ((last_amount / first_amount) ** (1/years_span) - 1) * 100 if first_amount > 0 and years_span > 0 else 0
            filtered_total = ticker_data_chart['amount'].sum()
        else:
            filtered_cagr = 0
            filtered_total = ticker_data_chart['amount'].sum() if len(ticker_data_chart) > 0 else 0
        
        st.markdown(f"""
        <div class='stock-card'>
            <h4 style='color: #00d4ff; margin-bottom: 15px;'>Statistical Analysis (Filtered Period)</h4>
            <p>📊 <strong>Average Dividend:</strong> <span style='color: #00ff88;'>${avg_amount:.3f}</span></p>
            <p>📊 <strong>Median Dividend:</strong> <span style='color: #00ff88;'>${median_amount:.3f}</span></p>
            <p>📊 <strong>Volatility (Std Dev):</strong> <span style='color: #00ff88;'>${volatility:.3f}</span></p>
            <p>📈 <strong>CAGR:</strong> <span style='color: #00ff88;'>{filtered_cagr:.2f}%</span></p>
            <p>💰 <strong>Total Paid (Period):</strong> <span style='color: #00ff88;'>${filtered_total:.2f}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📅 Payment Frequency Analysis")
        yearly_counts = ticker_data_chart.groupby('year').size()
        
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
        st.markdown("### 📋 Recent Dividend History (Filtered Period)")
        
        # Use filtered data and show up to last 12 payments from filtered period
        recent_payments = ticker_data_chart.tail(12)[::-1][['ex_dividend_date', 'payout_date', 'amount', 'pct_change']].copy()
        recent_payments.columns = ['Ex-Date', 'Pay Date', 'Amount', 'Growth %']
        
        # Show info about displayed vs total payments
        total_in_period = len(ticker_data_chart)
        showing = len(recent_payments)
        
        # Show date range of the displayed data
        if len(recent_payments) > 0:
            first_shown = recent_payments.iloc[-1]['Ex-Date']  # Last in reversed list = earliest date
            last_shown = recent_payments.iloc[0]['Ex-Date']    # First in reversed list = latest date
            
            st.markdown(f"""
            <p style='color: #a0b0c0; font-size: 12px; margin-bottom: 10px;'>
                Showing last <strong style='color: #00ff88;'>{showing}</strong> of <strong style='color: #00ff88;'>{total_in_period}</strong> payments in filtered period
                <br>
                <span style='color: #707070;'>Date range: {first_shown.strftime('%Y-%m-%d')} to {last_shown.strftime('%Y-%m-%d')}</span>
            </p>
            """, unsafe_allow_html=True)
        
        # Format the dataframe for display
        def color_growth(val):
            """Color code growth values"""
            if pd.isna(val):
                return ''
            color = '#00ff88' if val > 0 else '#ff6b6b' if val < 0 else '#ffffff'
            return f'color: {color}; font-weight: 600'
        
        # Style the dataframe
        styled_df = recent_payments.style.format({
            'Ex-Date': lambda x: x.strftime('%Y-%m-%d'),
            'Pay Date': lambda x: x.strftime('%Y-%m-%d'),
            'Amount': '${:.3f}',
            'Growth %': lambda x: '-' if pd.isna(x) else f'{x:.2f}%'
        }).applymap(color_growth, subset=['Growth %'])
        
        # Display with custom CSS
        st.markdown("""
        <style>
        .dataframe {
            font-size: 14px;
        }
        .dataframe th {
            background: rgba(0, 212, 255, 0.1);
            color: #a0b0c0;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #00d4ff;
        }
        .dataframe td {
            padding: 10px;
            border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            recent_payments,
            hide_index=True,
            use_container_width=True,
            key=f'recent_history_{start_date}_{end_date}_{len(recent_payments)}',
            column_config={
                'Ex-Date': st.column_config.DateColumn('Ex-Date', format='YYYY-MM-DD'),
                'Pay Date': st.column_config.DateColumn('Pay Date', format='YYYY-MM-DD'),
                'Amount': st.column_config.NumberColumn('Amount', format='$%.3f'),
                'Growth %': st.column_config.NumberColumn('Growth %', format='%.2f%%')
            }
        )
        
        st.markdown("")  # Add spacing
        
        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            csv_full = ticker_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Full History",
                data=csv_full,
                file_name=f"{selected_ticker}_dividend_full_history.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download complete dividend history"
            )
        
        with col_dl2:
            csv_filtered = ticker_data_chart.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Period",
                data=csv_filtered,
                file_name=f"{selected_ticker}_dividend_{start_date}_{end_date}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download data for selected date range"
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
