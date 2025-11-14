import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="US Dividend Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3241;
    }
    .stMetric:hover {
        border: 1px solid #4e5361;
        transition: 0.3s;
    }
    h1 {
        color: #00d4ff;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h2 {
        color: #00a8cc;
        font-weight: 600;
    }
    h3 {
        color: #0088aa;
        font-weight: 500;
    }
    .highlight-box {
        background: linear-gradient(135deg, #1e2130 0%, #2a2d3e 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #3e4251;
        margin: 10px 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00ff88;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #a0a0a0;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    """Load data from CSV file in repository"""
    # The CSV file is in the root of the repository
    csv_filename = 'all_usa_dividends_complete_20251114_1523.csv'
    
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        st.error(f"❌ Could not find {csv_filename} in the repository")
        st.stop()
    
    df['ex_dividend_date'] = pd.to_datetime(df['ex_dividend_date'])
    df['payout_date'] = pd.to_datetime(df['payout_date'])
    df['year'] = df['ex_dividend_date'].dt.year
    df['quarter'] = df['ex_dividend_date'].dt.quarter
    df['month'] = df['ex_dividend_date'].dt.month
    return df

@st.cache_data
def calculate_metrics(df):
    """Calculate key metrics for the dashboard"""
    latest_year = df['year'].max()
    current_year_data = df[df['year'] == latest_year]
    
    # Get most recent dividend for each ticker
    latest_dividends = df.sort_values('ex_dividend_date').groupby('ticker').tail(1)
    
    # Calculate dividend aristocrats (companies with 25+ years of dividend growth)
    ticker_history = df.groupby('ticker').agg({
        'ex_dividend_date': ['min', 'max'],
        'amount': 'count',
        'pct_change': lambda x: (x > 0).sum()
    }).reset_index()
    ticker_history.columns = ['ticker', 'first_date', 'last_date', 'payment_count', 'increases']
    ticker_history['years'] = (ticker_history['last_date'] - ticker_history['first_date']).dt.days / 365.25
    
    aristocrats = ticker_history[ticker_history['years'] >= 25]
    
    # Calculate yield metrics
    avg_dividend = latest_dividends['amount'].mean()
    median_dividend = latest_dividends['amount'].median()
    
    # Growth metrics
    growth_data = df[df['pct_change'].notna()]
    avg_growth = growth_data['pct_change'].mean()
    
    return {
        'total_tickers': df['ticker'].nunique(),
        'total_payments': len(df),
        'avg_dividend': avg_dividend,
        'median_dividend': median_dividend,
        'avg_growth': avg_growth,
        'aristocrats_count': len(aristocrats),
        'latest_year': latest_year,
        'latest_dividends': latest_dividends,
        'ticker_history': ticker_history
    }

# Load data
with st.spinner('Loading dividend data...'):
    df = load_data()
    metrics = calculate_metrics(df)

# Header
st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1>💰 US Dividend Analyzer</h1>
        <p style='font-size: 18px; color: #a0a0a0;'>
            Comprehensive analysis of {total_records:,} dividend payments across {total_tickers:,} US stocks
        </p>
        <p style='font-size: 14px; color: #707070;'>
            Data range: {min_year} - {max_year}
        </p>
    </div>
""".format(
    total_records=len(df),
    total_tickers=metrics['total_tickers'],
    min_year=df['year'].min(),
    max_year=df['year'].max()
), unsafe_allow_html=True)

# Sidebar filters
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("---")

# Year range filter
min_year, max_year = int(df['year'].min()), int(df['year'].max())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_year, max_year,
    (max(min_year, max_year - 10), max_year),
    help="Filter data by year range"
)

# Filter data
filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

# Minimum dividend filter
min_dividend = st.sidebar.number_input(
    "Minimum Dividend Amount ($)",
    min_value=0.0,
    max_value=float(df['amount'].max()),
    value=0.0,
    step=0.1,
    help="Filter stocks with dividends above this amount"
)

# Top KPIs
st.markdown("## 📊 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Dividend Payers",
        f"{metrics['total_tickers']:,}",
        help="Number of unique tickers in database"
    )

with col2:
    st.metric(
        "Avg Dividend",
        f"${metrics['avg_dividend']:.2f}",
        help="Average dividend amount across all recent payments"
    )

with col3:
    st.metric(
        "Median Dividend",
        f"${metrics['median_dividend']:.2f}",
        help="Median dividend amount (less affected by outliers)"
    )

with col4:
    st.metric(
        "Avg Growth Rate",
        f"{metrics['avg_growth']:.2f}%",
        delta="YoY",
        help="Average year-over-year dividend growth rate"
    )

with col5:
    st.metric(
        "Dividend Aristocrats",
        f"{metrics['aristocrats_count']}",
        help="Companies with 25+ years of dividend history"
    )

st.markdown("---")

# Two-column layout for main visualizations
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📈 Dividend Payments Over Time")
    
    # Aggregate by year
    yearly_stats = filtered_df.groupby('year').agg({
        'amount': ['sum', 'mean', 'count'],
        'ticker': 'nunique'
    }).reset_index()
    yearly_stats.columns = ['year', 'total_amount', 'avg_amount', 'payment_count', 'unique_tickers']
    
    # Create dual-axis plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=yearly_stats['year'],
            y=yearly_stats['payment_count'],
            name='Total Payments',
            marker_color='rgba(0, 212, 255, 0.6)',
            hovertemplate='<b>Year:</b> %{x}<br><b>Payments:</b> %{y:,}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=yearly_stats['year'],
            y=yearly_stats['avg_amount'],
            name='Avg Amount',
            mode='lines+markers',
            line=dict(color='#00ff88', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Year:</b> %{x}<br><b>Avg:</b> $%{y:.2f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title_text="Year", gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Number of Payments", secondary_y=False, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Average Dividend ($)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("### 💎 Top 20 Dividend Payers (Latest)")
    
    # Get top payers
    top_payers = metrics['latest_dividends'].nlargest(20, 'amount')[['ticker', 'amount', 'ex_dividend_date']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_payers['ticker'][::-1],
        x=top_payers['amount'][::-1],
        orientation='h',
        marker=dict(
            color=top_payers['amount'][::-1],
            colorscale='Viridis',
            colorbar=dict(title="Amount ($)")
        ),
        text=top_payers['amount'][::-1].apply(lambda x: f'${x:.2f}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Dividend: $%{x:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        xaxis_title="Dividend Amount ($)",
        yaxis_title="",
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Dividend Growth Analysis
col_growth1, col_growth2 = st.columns([1, 1])

with col_growth1:
    st.markdown("### 📊 Dividend Growth Distribution")
    
    # Filter for meaningful growth data
    growth_data = filtered_df[
        (filtered_df['pct_change'].notna()) & 
        (filtered_df['pct_change'] > -50) & 
        (filtered_df['pct_change'] < 100)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=growth_data['pct_change'],
        nbinsx=50,
        marker_color='rgba(0, 255, 136, 0.7)',
        hovertemplate='<b>Growth Rate:</b> %{x:.1f}%<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    # Add mean line
    mean_growth = growth_data['pct_change'].mean()
    fig.add_vline(
        x=mean_growth, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Mean: {mean_growth:.1f}%",
        annotation_position="top"
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        xaxis_title="Year-over-Year Growth Rate (%)",
        yaxis_title="Frequency",
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig, use_container_width=True)

with col_growth2:
    st.markdown("### 🏆 Dividend Aristocrats Timeline")
    
    # Companies with longest dividend history
    aristocrats = metrics['ticker_history'].nlargest(15, 'years')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=aristocrats['ticker'][::-1],
        x=aristocrats['years'][::-1],
        orientation='h',
        marker=dict(
            color=aristocrats['years'][::-1],
            colorscale='Plasma',
            colorbar=dict(title="Years")
        ),
        text=aristocrats['years'][::-1].apply(lambda x: f'{x:.1f}y'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>History: %{x:.1f} years<extra></extra>'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        xaxis_title="Years of Dividend History",
        yaxis_title="",
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Seasonal Analysis
st.markdown("### 📅 Seasonal Dividend Patterns")

col_season1, col_season2 = st.columns([1, 1])

with col_season1:
    # Monthly distribution
    monthly_dist = filtered_df.groupby('month').size().reset_index(name='count')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_dist['month_name'] = monthly_dist['month'].apply(lambda x: month_names[x-1])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_dist['month_name'],
        y=monthly_dist['count'],
        marker=dict(
            color=monthly_dist['count'],
            colorscale='Turbo',
            colorbar=dict(title="Payments")
        ),
        text=monthly_dist['count'],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Payments: %{y:,}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Ex-Dividend Dates by Month",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        xaxis_title="Month",
        yaxis_title="Number of Payments",
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig, use_container_width=True)

with col_season2:
    # Quarterly distribution
    quarterly_dist = filtered_df.groupby('quarter').agg({
        'amount': 'mean',
        'ticker': 'count'
    }).reset_index()
    quarterly_dist.columns = ['quarter', 'avg_amount', 'count']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=['Q1', 'Q2', 'Q3', 'Q4'],
            y=quarterly_dist['count'],
            name='Payment Count',
            marker_color='rgba(0, 168, 204, 0.6)',
            hovertemplate='<b>%{x}</b><br>Payments: %{y:,}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=['Q1', 'Q2', 'Q3', 'Q4'],
            y=quarterly_dist['avg_amount'],
            name='Avg Amount',
            mode='lines+markers',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=10),
            hovertemplate='<b>%{x}</b><br>Avg: $%{y:.2f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Quarterly Dividend Patterns",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title_text="Quarter", gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Number of Payments", secondary_y=False, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Average Amount ($)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Stock Search and Details
st.markdown("### 🔎 Stock Dividend Search")

col_search1, col_search2 = st.columns([1, 3])

with col_search1:
    # Get list of all tickers
    all_tickers = sorted(df['ticker'].unique())
    selected_ticker = st.selectbox(
        "Select a ticker to analyze:",
        options=[''] + all_tickers,
        help="Search and select a stock ticker"
    )

if selected_ticker:
    with col_search2:
        st.markdown(f"#### 📌 {selected_ticker} Dividend History")
    
    # Filter data for selected ticker
    ticker_data = df[df['ticker'] == selected_ticker].sort_values('ex_dividend_date')
    
    col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
    
    with col_detail1:
        st.metric("Total Payments", f"{len(ticker_data):,}")
    
    with col_detail2:
        latest_amount = ticker_data.iloc[-1]['amount']
        st.metric("Latest Dividend", f"${latest_amount:.3f}")
    
    with col_detail3:
        years_active = (ticker_data['ex_dividend_date'].max() - ticker_data['ex_dividend_date'].min()).days / 365.25
        st.metric("Years Active", f"{years_active:.1f}")
    
    with col_detail4:
        growth_records = ticker_data[ticker_data['pct_change'].notna()]
        if len(growth_records) > 0:
            avg_growth_ticker = growth_records['pct_change'].mean()
            st.metric("Avg Growth", f"{avg_growth_ticker:.2f}%")
        else:
            st.metric("Avg Growth", "N/A")
    
    # Plot dividend history
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ticker_data['ex_dividend_date'],
        y=ticker_data['amount'],
        mode='lines+markers',
        name='Dividend Amount',
        line=dict(color='#00d4ff', width=2),
        marker=dict(size=6, color='#00ff88'),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)',
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Amount:</b> $%{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        xaxis_title="Ex-Dividend Date",
        yaxis_title="Dividend Amount ($)",
        hovermode='closest'
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show recent dividend table
    st.markdown("#### Recent Dividend Payments")
    recent_payments = ticker_data.tail(10)[::-1][['ex_dividend_date', 'payout_date', 'amount', 'pct_change']].copy()
    recent_payments['ex_dividend_date'] = recent_payments['ex_dividend_date'].dt.strftime('%Y-%m-%d')
    recent_payments['payout_date'] = recent_payments['payout_date'].dt.strftime('%Y-%m-%d')
    recent_payments['amount'] = recent_payments['amount'].apply(lambda x: f'${x:.3f}')
    recent_payments['pct_change'] = recent_payments['pct_change'].apply(
        lambda x: f'{x:.2f}%' if pd.notna(x) else '-'
    )
    recent_payments.columns = ['Ex-Dividend Date', 'Payout Date', 'Amount', 'YoY Change']
    
    st.dataframe(recent_payments, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #707070; padding: 20px;'>
        <p>💰 US Dividend Analyzer | Built with Streamlit & Plotly</p>
        <p style='font-size: 12px;'>Data includes {total:,} dividend payments from {min_y} to {max_y}</p>
    </div>
""".format(total=len(df), min_y=df['year'].min(), max_y=df['year'].max()), unsafe_allow_html=True)
