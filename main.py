import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from streamlit_tags import st_tags

# Configuration
st.set_page_config(page_title="EDI Analytics Dashboard", layout="wide")

# Mock Data Generation
@st.cache_data
def load_mock_data():
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-03-31')
    carriers = ['UPS', 'FedEx', 'DHL', 'USPS', 'XPO Logistics']
    statuses = ['AP', 'DP', 'AR', 'DD']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami']
    
    data = pd.DataFrame({
        'shipment_id': [f'SHPM{1000+i}' for i in range(500)],
        'date': np.random.choice(dates, 500),
        'carrier': np.random.choice(carriers, 500),
        'status': np.random.choice(statuses, 500),
        'location': np.random.choice(locations, 500),
        'value_usd': np.random.uniform(1000, 5000, 500),
        'weight_lbs': np.random.uniform(10, 500, 500),
        'delivery_time_hrs': np.random.normal(48, 12, 500),
        'is_delayed': np.random.choice([0, 1], 500, p=[0.7, 0.3])
    })
    
    # Add geolocation coordinates
    location_coords = {
        'New York': (40.7128, -74.0060),
        'Los Angeles': (34.0522, -118.2437),
        'Chicago': (41.8781, -87.6298),
        'Houston': (29.7604, -95.3698),
        'Miami': (25.7617, -80.1918)
    }
    
    data['lat'] = data['location'].apply(lambda x: location_coords[x][0])
    data['lon'] = data['location'].apply(lambda x: location_coords[x][1])
    
    return data

df = load_mock_data()

# Status mapping
SHIPMENT_STATUS_CODES = {
    "AP": "Arrived Pickup",
    "DP": "Departed Pickup",
    "AR": "Arrived Destination",
    "DD": "Delivered"
}

# Sidebar Filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Date Range", 
    [datetime(2024,1,1), datetime(2024,3,31)])

selected_carriers = st.sidebar.multiselect("Carriers", 
    df['carrier'].unique(), 
    default=df['carrier'].unique())

status_filter = st.sidebar.multiselect("Status", 
    df['status'].map(SHIPMENT_STATUS_CODES).unique(),
    default=df['status'].map(SHIPMENT_STATUS_CODES).unique())

# Apply filters
filtered_df = df[
    (df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))) &
    (df['carrier'].isin(selected_carriers)) &
    (df['status'].map(SHIPMENT_STATUS_CODES).isin(status_filter))
]

# Main Dashboard
st.title("📈 Logistics EDI Analytics Dashboard")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Shipments", filtered_df.shape[0], delta="+5% MoM")
with col2:
    pending = filtered_df[~filtered_df['status'].isin(['DD'])].shape[0]
    st.metric("Pending Actions", pending, delta="-2% MoM")
with col3:
    on_time_rate = (1 - filtered_df['is_delayed'].mean()) * 100
    st.metric("On-Time Rate", f"{on_time_rate:.1f}%", delta="+1.5% MoM")
with col4:
    error_rate = np.random.uniform(0.5, 2.5)
    st.metric("EDI Error Rate", f"{error_rate:.1f}%", delta="-0.3% MoM")

# Visualizations
st.subheader("Shipment Analytics")
c1, c2 = st.columns([2, 1])

with c1:
    # Time Series Analysis
    ts_data = filtered_df.groupby(pd.Grouper(key='date', freq='W'))['shipment_id'].count().reset_index()
    fig = px.line(ts_data, x='date', y='shipment_id', 
                 title="Weekly Shipment Volume",
                 labels={'shipment_id': 'Number of Shipments', 'date': 'Week'})
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Status Distribution
    status_counts = filtered_df['status'].value_counts().reset_index()
    status_counts['status'] = status_counts['status'].map(SHIPMENT_STATUS_CODES)
    fig = px.pie(status_counts, names='status', values='count',
                 title="Shipment Status Distribution",
                 hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

# Carrier Performance
st.subheader("Carrier Performance")
col1, col2 = st.columns(2)

with col1:
    carrier_metrics = filtered_df.groupby('carrier').agg({
        'delivery_time_hrs': 'mean',
        'is_delayed': 'mean',
        'value_usd': 'sum'
    }).reset_index()
    
    fig = px.bar(carrier_metrics, x='carrier', y='delivery_time_hrs',
                 title="Average Delivery Time by Carrier",
                 labels={'delivery_time_hrs': 'Hours', 'carrier': ''})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(filtered_df, x='weight_lbs', y='delivery_time_hrs',
                     color='carrier', size='value_usd',
                     title="Delivery Time vs Weight & Value",
                     labels={'weight_lbs': 'Weight (lbs)', 
                             'delivery_time_hrs': 'Delivery Time (hrs)'})
    st.plotly_chart(fig, use_container_width=True)

# Geographical Analysis
st.subheader("Geographical Distribution")
geo_data = filtered_df.groupby(['location', 'lat', 'lon']).agg({
    'shipment_id': 'count',
    'value_usd': 'sum'
}).reset_index()

fig = px.scatter_geo(geo_data, lat='lat', lon='lon',
                     size='shipment_id', color='value_usd',
                     hover_name='location', scope='north america',
                     title="Shipment Volume & Value by Location",
                     size_max=30)
st.plotly_chart(fig, use_container_width=True)

# Data Table
st.subheader("Recent Shipments")
st.dataframe(filtered_df.sort_values('date', ascending=False).head(10)[
    ['date', 'shipment_id', 'carrier', 'status', 'location', 'value_usd']
], use_container_width=True)

# Error Log Section
st.subheader("Recent EDI Exceptions")
error_log = pd.DataFrame({
    'timestamp': pd.date_range('2024-03-20', periods=5),
    'error_type': ['Missing BOL', 'Invalid Format', 'Date Mismatch', 
                  'Duplicate ID', 'Missing Segments'],
    'severity': ['High', 'Medium', 'Low', 'Medium', 'High'],
    'status': ['Resolved', 'Pending', 'Open', 'Open', 'Pending']
})
st.dataframe(error_log, use_container_width=True)

# Add this to imports
from streamlit_tags import st_tags  # pip install streamlit-tags

# Add this after the main data table section
st.header("🛑 EDI Exception Analytics")

# Add at the top of your imports
from streamlit_tags import st_tags

# Only run if user triggers it
if st.button("🔍 Load Exception Analytics") or st.session_state.get('show_exceptions', False):
    st.session_state.show_exceptions = True
    
    # Generate mock error data
    error_log = pd.DataFrame({
        'timestamp': pd.date_range('2024-03-01', periods=200, freq='H'),
        'error_type': np.random.choice([
            'Missing BOL', 'Invalid Format', 'Date Mismatch', 
            'Duplicate ID', 'Missing Segments', 'Invalid Carrier Code',
            'Data Type Mismatch', 'Routing Error', 'Price Discrepancy',
            'Missing Tracking Number', 'Invalid Weight Value'
        ], 200, p=[0.2,0.15,0.1,0.1,0.1,0.08,0.07,0.06,0.05,0.05,0.04]),
        'severity': np.random.choice(['High', 'Medium', 'Low'], 200, p=[0.3,0.5,0.2]),
        'status': np.random.choice(['Open', 'Resolved', 'In Progress'], 200, p=[0.4,0.4,0.2]),
        'resolution_time_hrs': np.random.exponential(24, 200),
        'carrier': np.random.choice(df['carrier'].unique(), 200),
        'source_system': np.random.choice(['TMS', 'WMS', 'OMS', 'ERP'], 200)
    })

    # Filters
    with st.expander("⚙️ Filter Exceptions"):
        col1, col2, col3 = st.columns(3)
        with col1:
            error_severity = st.multiselect("Severity Level", 
                                          error_log['severity'].unique(),
                                          default=['High', 'Medium'])
        with col2:
            error_status = st.multiselect("Resolution Status", 
                                        error_log['status'].unique(),
                                        default=['Open', 'In Progress'])
        with col3:
            selected_systems = st.multiselect("Source Systems", 
                                            error_log['source_system'].unique(),
                                            default=error_log['source_system'].unique())

    filtered_errors = error_log[
        (error_log['severity'].isin(error_severity)) &
        (error_log['status'].isin(error_status)) &
        (error_log['source_system'].isin(selected_systems))
    ]

    # Exception KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Exceptions", filtered_errors.shape[0], 
                 delta=f"{len(filtered_errors)-100} vs prev period")
    with col2:
        avg_resolution = filtered_errors['resolution_time_hrs'].mean()
        st.metric("Avg Resolution Time", f"{avg_resolution:.1f} hours")
    with col3:
        open_errors = filtered_errors[filtered_errors['status'] == 'Open'].shape[0]
        st.metric("Open Exceptions", open_errors, delta_color="inverse")
    with col4:
        recurrence_rate = filtered_errors['error_type'].value_counts().max()/len(filtered_errors)
        st.metric("Top Error Frequency", f"{recurrence_rate:.1%}")

    # Visualization Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Trend Analysis", 
        "Error Composition", 
        "Root Cause Analysis",
        "Exception Details"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            error_ts = filtered_errors.set_index('timestamp').resample('D').size()
            fig = px.line(error_ts, title="Daily Exception Trend",
                         labels={'value': 'Number of Exceptions'})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(filtered_errors, x='resolution_time_hrs',
                              nbins=20, title="Resolution Time Distribution",
                              labels={'resolution_time_hrs': 'Hours to Resolve'})
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns([2,1])
        with c1:
            error_counts = filtered_errors['error_type'].value_counts().reset_index()
            fig = px.bar(error_counts, x='count', y='error_type', 
                        orientation='h', title="Error Type Frequency")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(filtered_errors, names='severity', 
                        title="Exception Severity Distribution",
                        hole=0.3, color='severity',
                        color_discrete_map={'High':'red', 'Medium':'orange', 'Low':'green'})
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.sunburst(filtered_errors, path=['source_system', 'carrier', 'error_type'],
                             title="Exception Source Analysis")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            corr_data = pd.get_dummies(filtered_errors[['error_type', 'carrier', 'source_system']])
            corr_matrix = corr_data.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu'))
            fig.update_layout(title="Error Type Correlations")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        selected_errors = st_tags(
            label='Filter specific error types:',
            text='Start typing to search',
            value=[],
            suggestions=filtered_errors['error_type'].unique().tolist())
        
        display_errors = filtered_errors[
            filtered_errors['error_type'].isin(selected_errors)
        ] if selected_errors else filtered_errors
        
        st.dataframe(display_errors.sort_values('timestamp', ascending=False),
                    column_config={
                        "timestamp": "Timestamp",
                        "error_type": "Error Type",
                        "severity": st.column_config.SelectboxColumn(
                            "Severity",
                            options=["High", "Medium", "Low"]
                        ),
                        "status": st.column_config.SelectboxColumn(
                            "Status",
                            options=["Open", "Resolved", "In Progress"]
                        )
                    },
                    use_container_width=True,
                    hide_index=True)

    # Add clear button at the bottom
    if st.button("❌ Clear Analytics"):
        st.session_state.show_exceptions = False
        st.experimental_rerun()

else:
    st.info("Click the 'Load Exception Analytics' button above to view error diagnostics")
    st.stop()
