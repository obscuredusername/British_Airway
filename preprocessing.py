import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title='Flight Review Analysis',
    layout='wide',
    page_icon='✈️'
)

# Custom CSS for dark theme
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}
.stDataFrame {
    background-color: #1E2130;
    color: #FFFFFF;
}
.dashboard-title {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 20px;
    color: #80cbc4;
}
.section-title {
    color: #26a69a;
    font-size: 1.5rem;
    margin-top: 10px;
    margin-bottom: 10px;
}
.metric-container {
    background-color: #1E2130;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}
.big-number {
    font-size: 2.5rem;
    font-weight: bold;
    color: #80cbc4;
}
.metric-label {
    font-size: 1rem;
    color: #90a4ae;
}
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('british_airways.csv')
        
        # Convert columns to numeric, handling errors
        numeric_columns = ['Overall Rating', 'Legroom', 'Seat Comfort', 
                        'In-flight Entertainment', 'Customer Service', 
                        'Value for Money', 'Cleanliness', 
                        'Check-in and Boarding', 'Food and Beverage']
        
        # Filter to only include columns that exist in the dataframe
        existing_numeric_columns = [col for col in numeric_columns if col in data.columns]
        
        for col in existing_numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Extract year from Date of Travel
        def extract_year(date_str):
            if pd.isna(date_str):
                return None
            match = re.search(r'\b(20\d{2})\b', str(date_str))
            return int(match.group(1)) if match else None
            
        # Only add Year column if Date of Travel exists
        if 'Date of Travel' in data.columns:
            data['Year'] = data['Date of Travel'].apply(extract_year)
        else:
            data['Year'] = None
        
        # Clean review text if it exists
        if 'Review Text' in data.columns:
            data['Review Text'] = data['Review Text'].fillna('').astype(str).str.replace('"', '').str.strip()
        
        return data
    except FileNotFoundError:
        st.error("Error: Data file 'british_airways.csv' not found. Please make sure the file exists in the current directory.")
        # Return an empty DataFrame with expected columns to prevent errors
        return pd.DataFrame(columns=['Overall Rating', 'Origin', 'Destination', 'Year'])
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(columns=['Overall Rating', 'Origin', 'Destination', 'Year'])

# Load data
df = load_data()

# Check if dataframe is empty
if df.empty:
    st.error("No data available for analysis. Please check your data file.")
    st.stop()

# Dashboard title
st.markdown("<h1 class='dashboard-title'>✈️ British Airways Review Analysis</h1>", unsafe_allow_html=True)

# Sidebar for filters
st.sidebar.header('Filters')

# Add year filter if Year column has data
years = []
if 'Year' in df.columns and not df['Year'].isna().all():
    years = sorted(df['Year'].dropna().unique().astype(int).tolist())
    
if years:
    selected_years = st.sidebar.multiselect('Select Years', years, default=years)
else:
    selected_years = []

# Only show filters for columns that exist
if 'Origin' in df.columns:
    origin_options = sorted(df['Origin'].dropna().unique())
    origin_filter = st.sidebar.multiselect('Select Origin', origin_options)
else:
    origin_filter = []

if 'Destination' in df.columns:
    destination_options = sorted(df['Destination'].dropna().unique())
    destination_filter = st.sidebar.multiselect('Select Destination', destination_options)
else:
    destination_filter = []

if 'Flight Type' in df.columns:
    flight_type_options = sorted(df['Flight Type'].dropna().unique())
    flight_type_filter = st.sidebar.multiselect('Select Flight Type', flight_type_options)
else:
    flight_type_filter = []

# Rating range filter (only if Overall Rating exists)
if 'Overall Rating' in df.columns:
    min_possible = int(df['Overall Rating'].min()) if not df['Overall Rating'].isna().all() else 1
    max_possible = int(df['Overall Rating'].max()) if not df['Overall Rating'].isna().all() else 5
    
    min_rating, max_rating = st.sidebar.slider(
        'Rating Range',
        min_value=min_possible,
        max_value=max_possible,
        value=(min_possible, max_possible)
    )
else:
    min_rating, max_rating = 1, 5

# Apply filters
filtered_df = df.copy()

# Apply year filter if years are selected and Year column exists
if selected_years and 'Year' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]

if origin_filter and 'Origin' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Origin'].isin(origin_filter)]
    
if destination_filter and 'Destination' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Destination'].isin(destination_filter)]
    
if flight_type_filter and 'Flight Type' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Flight Type'].isin(flight_type_filter)]
    
# Apply rating filter if Overall Rating exists
if 'Overall Rating' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['Overall Rating'] >= min_rating) & 
        (filtered_df['Overall Rating'] <= max_rating)
    ]

# Check if filtered dataframe is empty
if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust your filter criteria.")
    st.stop()

# Key metrics at the top
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    if 'Overall Rating' in filtered_df.columns and not filtered_df['Overall Rating'].isna().all():
        avg_rating = filtered_df['Overall Rating'].mean()
        st.markdown(f"<div class='big-number'>{avg_rating:.1f}/5</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='big-number'>N/A</div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Average Rating</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    review_count = len(filtered_df)
    st.markdown(f"<div class='big-number'>{review_count}</div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Total Reviews</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    if 'Overall Rating' in filtered_df.columns and not filtered_df['Overall Rating'].isna().all():
        satisfaction_rate = (filtered_df['Overall Rating'] >= 4).sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        st.markdown(f"<div class='big-number'>{satisfaction_rate:.1f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='big-number'>N/A</div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Satisfaction Rate</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    if 'Year' in filtered_df.columns and not filtered_df['Year'].isna().all():
        most_recent_year = filtered_df['Year'].max()
        st.markdown(f"<div class='big-number'>{int(most_recent_year)}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='big-number'>N/A</div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Most Recent Year</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Create tab layout
tab1, tab2, tab3, tab4 = st.tabs(['Ratings Overview', 'Detailed Analysis', 'Route Analysis', 'Text Analysis'])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Rating Distribution (if Overall Rating exists)
        if 'Overall Rating' in filtered_df.columns and not filtered_df['Overall Rating'].isna().all():
            fig_rating_dist = px.histogram(
                filtered_df, 
                x='Overall Rating',
                nbins=5,
                color_discrete_sequence=['#26a69a'],
                template='plotly_dark',
                title='Distribution of Overall Ratings'
            )
            fig_rating_dist.update_layout(bargap=0.1)
            st.plotly_chart(fig_rating_dist, use_container_width=True)
        else:
            st.info("Overall Rating data not available for histogram.")
        
    with col2:
        # 2. Interactive Radar Chart of Average Ratings
        potential_rating_columns = ['Legroom', 'Seat Comfort', 'In-flight Entertainment', 
                        'Customer Service', 'Value for Money', 'Cleanliness', 
                        'Check-in and Boarding', 'Food and Beverage']
        
        # Filter to only include columns that exist in the dataframe
        rating_columns = [col for col in potential_rating_columns if col in filtered_df.columns]
        
        if rating_columns and not filtered_df[rating_columns].isna().all().all():
            avg_ratings = filtered_df[rating_columns].mean()
            
            # Only include non-NaN values
            valid_ratings = avg_ratings.dropna()
            
            if not valid_ratings.empty:
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=valid_ratings.values,
                    theta=valid_ratings.index,
                    fill='toself',
                    line_color='#80cbc4'
                ))
                fig_radar.update_layout(
                    title='Average Ratings Across Flight Aspects',
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 5]
                        )),
                    template='plotly_dark'
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Insufficient data for radar chart")
        else:
            st.info("Rating category data not available for radar chart")
    
    # 3. Yearly Trend
    if 'Year' in filtered_df.columns and 'Overall Rating' in filtered_df.columns and not filtered_df['Year'].isna().all():
        # Handle potential NaN values
        yearly_data = filtered_df.dropna(subset=['Year', 'Overall Rating'])
        
        if not yearly_data.empty and len(yearly_data['Year'].unique()) > 1:
            yearly_ratings = yearly_data.groupby('Year')['Overall Rating'].mean().reset_index()
            yearly_ratings = yearly_ratings.sort_values('Year')
            
            fig_yearly = px.line(
                yearly_ratings,
                x='Year',
                y='Overall Rating',
                markers=True,
                line_shape='linear',
                color_discrete_sequence=['#80cbc4'],
                template='plotly_dark',
                title='Average Rating Trend by Year'
            )
            fig_yearly.update_layout(
                xaxis_title='Year',
                yaxis_title='Average Rating',
                yaxis=dict(range=[0, 5])
            )
            st.plotly_chart(fig_yearly, use_container_width=True)
        else:
            st.info("Insufficient yearly data for trend analysis")

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # 4. Correlation Heatmap
        potential_rating_columns = ['Legroom', 'Seat Comfort', 'In-flight Entertainment', 
                        'Customer Service', 'Value for Money', 'Cleanliness', 
                        'Check-in and Boarding', 'Food and Beverage', 'Overall Rating']
        
        # Filter to only include columns that exist in the dataframe
        rating_columns = [col for col in potential_rating_columns if col in filtered_df.columns]
        
        if len(rating_columns) > 1 and not filtered_df[rating_columns].isna().all().all():
            # Drop NaN values for correlation calculation
            corr_data = filtered_df[rating_columns].dropna()
            
            if len(corr_data) > 1:  # Need at least 2 rows for correlation
                corr_matrix = corr_data.corr()
                fig_corr = px.imshow(
                    corr_matrix, 
                    color_continuous_scale='Teal',
                    title='Correlation Between Flight Aspects',
                    template='plotly_dark',
                    text_auto=True
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Insufficient data for correlation analysis")
        else:
            st.info("Insufficient rating data for correlation analysis")
    
    with col2:
        potential_rating_columns = ['Legroom', 'Seat Comfort', 'In-flight Entertainment', 
                        'Customer Service', 'Value for Money', 'Cleanliness', 
                        'Check-in and Boarding', 'Food and Beverage']
        
        rating_columns = [col for col in potential_rating_columns if col in filtered_df.columns]
        
        if rating_columns and not filtered_df[rating_columns].isna().all().all():
            fig_boxplot = go.Figure()
            
            for col in rating_columns:
                if not filtered_df[col].isna().all():
                    fig_boxplot.add_trace(go.Box(
                        y=filtered_df[col].dropna(),  
                        name=col,
                        boxmean=True,
                        marker_color='#80cbc4'
                    ))
            
            if len(fig_boxplot.data) > 0:  
                fig_boxplot.update_layout(
                    title='Rating Distribution Across Flight Aspects',
                    template='plotly_dark',
                    showlegend=False
                )
                st.plotly_chart(fig_boxplot, use_container_width=True)
            else:
                st.info("Insufficient data for box plot")
        else:
            st.info("Rating category data not available for box plot")
    

    potential_rating_columns = ['Legroom', 'Seat Comfort', 'In-flight Entertainment', 
                    'Customer Service', 'Value for Money', 'Cleanliness', 
                    'Check-in and Boarding', 'Food and Beverage']

    rating_columns = [col for col in potential_rating_columns if col in filtered_df.columns]
    
   
with tab3:
    if 'Origin' in filtered_df.columns and 'Destination' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            if not filtered_df['Origin'].isna().all():
                origin_counts = filtered_df['Origin'].value_counts().head(10).reset_index()
                origin_counts.columns = ['Origin', 'Count']
                
                if not origin_counts.empty:
                    fig_origins = px.bar(
                        origin_counts,
                        x='Count',
                        y='Origin',
                        orientation='h',
                        color='Count',
                        color_continuous_scale=px.colors.sequential.Teal,
                        template='plotly_dark',
                        title='Top 10 Origins'
                    )
                    fig_origins.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_origins, use_container_width=True)
                else:
                    st.info("No data available for origins chart")
            else:
                st.info("Origin data not available")
        
        with col2:
            if not filtered_df['Destination'].isna().all():
                dest_counts = filtered_df['Destination'].value_counts().head(10).reset_index()
                dest_counts.columns = ['Destination', 'Count']
                
                if not dest_counts.empty:
                    fig_dests = px.bar(
                        dest_counts,
                        x='Count',
                        y='Destination',
                        orientation='h',
                        color='Count',
                        color_continuous_scale='Teal',
                        template='plotly_dark',
                        title='Top 10 Destinations'
                    )
                    fig_dests.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_dests, use_container_width=True)
                else:
                    st.info("No data available for destinations chart")
            else:
                st.info("Destination data not available")
        

        if 'Overall Rating' in filtered_df.columns:
            st.markdown("<h3 class='section-title'>Route Ratings</h3>", unsafe_allow_html=True)
            
            # Drop rows with NaN values in required columns
            route_data = filtered_df.dropna(subset=['Origin', 'Destination', 'Overall Rating'])
            
            if not route_data.empty:
                # Calculate average ratings for each route
                route_ratings = route_data.groupby(['Origin', 'Destination'])['Overall Rating'].agg(['mean', 'count']).reset_index()
                route_ratings = route_ratings[route_ratings['count'] >= 1]  # Only routes with at least 1 review
                
                if not route_ratings.empty:
                    route_ratings = route_ratings.sort_values('mean', ascending=False)
                    
                    # Create a bubble chart for routes
                    fig_routes = px.scatter(
                        route_ratings,
                        x='Origin',
                        y='Destination',
                        size='count',
                        color='mean',
                        color_continuous_scale='Teal',
                        range_color=[1, 5],
                        size_max=30,
                        template='plotly_dark',
                        title='Route Ratings (Bubble Size: Number of Reviews, Color: Average Rating)'
                    )
                    fig_routes.update_layout(
                        xaxis_title='Origin',
                        yaxis_title='Destination'
                    )
                    st.plotly_chart(fig_routes, use_container_width=True)
                else:
                    st.info("Insufficient route data for analysis")
            else:
                st.info("No valid route rating data available")
    else:
        st.info("Origin and Destination data required for route analysis")

with tab4:
    # 10. Word Cloud from Review Text
    if 'Review Text' in filtered_df.columns:
        st.markdown("<h3 class='section-title'>Word Cloud from Reviews</h3>", unsafe_allow_html=True)
        
        # Filter out empty reviews
        text_data = filtered_df[filtered_df['Review Text'].str.strip() != '']
        
        if not text_data.empty:
            # Combine all review text
            all_text = ' '.join(text_data['Review Text'].astype(str))
            
            if all_text and all_text.strip():
                try:
                    # Generate word cloud
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='#0E1117',
                        colormap='viridis',
                        max_words=100
                    ).generate(all_text)
                    
                    # Display the word cloud
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.tight_layout(pad=0)
                    st.pyplot(plt)
                except Exception as e:
                    st.error(f"Error generating word cloud: {str(e)}")
            else:
                st.info("No meaningful text content found for word cloud")
                
            # 11. Sample reviews
            if 'Overall Rating' in filtered_df.columns:
                st.markdown("<h3 class='section-title'>Route Ratings</h3>", unsafe_allow_html=True)
                
                route_data = filtered_df.dropna(subset=['Origin', 'Destination', 'Overall Rating'])
                
                if not route_data.empty:
                    route_ratings = route_data.groupby(['Origin', 'Destination'])['Overall Rating'].agg(['mean', 'count']).reset_index()
                    route_ratings = route_ratings[route_ratings['count'] >= 1]
                    
                    if not route_ratings.empty:
                        route_ratings = route_ratings.sort_values('mean', ascending=False)
                        
                        # Normalize bubble sizes
                        route_ratings['scaled_size'] = (route_ratings['count'] - route_ratings['count'].min()) / (route_ratings['count'].max() - route_ratings['count'].min()) * 40 + 10

                        fig_routes = px.scatter(
                            route_ratings,
                            x='Origin',
                            y='Destination',
                            size='scaled_size',
                            color='mean',
                            color_continuous_scale='Blues',
                            range_color=[1, 5],
                            size_max=50,
                            template='ggplot2',
                            title='Route Ratings (Bubble Size: Reviews, Color: Avg Rating)'
                        )

                        # Improve visibility
                        fig_routes.update_layout(
                            xaxis={'categoryorder': 'total descending', 'title': 'Origin', 'tickangle': -45},
                            yaxis={'categoryorder': 'total descending', 'title': 'Destination'},
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            margin=dict(l=50, r=50, t=50, b=50)
                        )
                        
                        # Annotate top 5 routes
                        fig_routes.update_traces(marker=dict(line=dict(width=2, color='white')))
                        for _, row in route_ratings.head(5).iterrows():
                            fig_routes.add_annotation(
                                x=row['Origin'], y=row['Destination'],
                                text=f"{row['mean']:.2f} ⭐",
                                showarrow=True, arrowhead=2,
                                font=dict(color='black', size=10),
                                bgcolor='white', opacity=0.8
                            )

                        st.plotly_chart(fig_routes, use_container_width=True)
                    else:
                        st.info("Insufficient route data for analysis")
                else:
                    st.info("No valid route rating data available")
            else:
                st.info("Origin and Destination data required for route analysis")

# 12. Year-wise comparison of ratings (if year data is available)
if ('Year' in filtered_df.columns and 'Overall Rating' in filtered_df.columns and 
    not filtered_df['Year'].isna().all() and len(filtered_df['Year'].unique()) > 1):
    
    st.markdown("<h3 class='section-title'>Year-wise Comparison of Ratings</h3>", unsafe_allow_html=True)
    
    potential_rating_columns = ['Legroom', 'Seat Comfort', 'In-flight Entertainment', 
                    'Customer Service', 'Value for Money', 'Cleanliness', 
                    'Check-in and Boarding', 'Food and Beverage']
    
    # Filter to only include columns that exist in the dataframe
    rating_columns = [col for col in potential_rating_columns if col in filtered_df.columns]
    
    # Filter for columns with actual data
    valid_rating_columns = [col for col in rating_columns if not filtered_df[col].isna().all()]
    
    if valid_rating_columns:
        # Drop rows with NaN in Year
        yearly_data = filtered_df.dropna(subset=['Year'])
        
        if not yearly_data.empty:
            # Group by year and calculate average ratings
            yearly_metrics = yearly_data.groupby('Year')[valid_rating_columns].mean().reset_index()
            
            if not yearly_metrics.empty and len(yearly_metrics) > 1:
                # Create a multi-line chart
                fig_yearly_metrics = go.Figure()
                
                for col in valid_rating_columns:
                    # Only add traces for columns with non-NaN values
                    if not yearly_metrics[col].isna().all():
                        fig_yearly_metrics.add_trace(go.Scatter(
                            x=yearly_metrics['Year'],
                            y=yearly_metrics[col],
                            mode='lines+markers',
                            name=col
                        ))
                
                if len(fig_yearly_metrics.data) > 0:  # Check if we added any traces
                    fig_yearly_metrics.update_layout(
                        title='Year-wise Comparison of Rating Metrics',
                        xaxis_title='Year',
                        yaxis_title='Average Rating',
                        yaxis=dict(range=[0, 5]),
                        template='plotly_dark',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig_yearly_metrics, use_container_width=True)
                else:
                    st.info("Insufficient yearly rating data for comparison")
            else:
                st.info("Insufficient yearly data for metrics comparison")
        else:
            st.info("No valid yearly data available")
    else:
        st.info("No valid rating categories available for yearly comparison")

# Display raw data
st.header('Raw Data')
if not filtered_df.empty:
    st.dataframe(filtered_df, height=300)
else:
    st.info("No data to display")

# Save button for data
if not filtered_df.empty:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='british_airways.csv',
        mime='text/csv',
    )