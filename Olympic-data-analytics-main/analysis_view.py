import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import base64 # Import base64 for image handling, though maybe not needed here

# --- UTILITY FUNCTIONS (Place all your helper functions here) ---

def find_region_column(df):
    # ... (Keep find_region_column logic) ...
    candidates = [c for c in df.columns if c.lower() in ('region','country','region_name','country_name','name')]
    candidates = [c for c in candidates if c.upper() != 'NOC']
    if candidates:
        return candidates[0]
    other = [c for c in df.columns if c.upper() != 'NOC']
    return other[0] if other else None

def safe_read_csv(uploaded_file):
    # ... (Keep safe_read_csv logic) ...
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file: {e}")

def safe_preprocess(ath_df, region_df):
    # ... (Keep safe_preprocess logic) ...
    if ath_df is None or region_df is None:
        raise ValueError("Both athlete_events and noc_regions must be provided.")

    df = ath_df.copy()

    if 'Season' in df.columns:
        try:
            df = df[df['Season'] == 'Summer'].copy()
        except Exception:
            pass

    preferred = ['ID','Name','Sex','Age','Height','Weight','Team','NOC','Games','Year','City','Sport','Event','Medal']
    keep_cols = [c for c in preferred if c in df.columns]
    df = df[keep_cols].copy()

    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')

    if 'NOC' not in df.columns:
        raise ValueError("Uploaded athlete_events CSV does not contain 'NOC' column. This column is required.")

    r = region_df.copy()
    if 'NOC' in r.columns:
        r = r.drop_duplicates(subset=['NOC']).copy()

    region_col = find_region_column(r) if isinstance(r, pd.DataFrame) else None
    if region_col is None:
        if 'NOC' in r.columns:
            r['region'] = r['NOC']
        else:
            r['region'] = np.nan
    else:
        if region_col != 'region':
            r = r.rename(columns={region_col: 'region'})

    if 'region' not in r.columns:
        if 'Team' in df.columns:
            r['region'] = r.get('Team', np.nan)
        elif 'NOC' in r.columns:
            r['region'] = r['NOC']
        else:
            r['region'] = np.nan

    df = df.merge(r[['NOC','region']], on='NOC', how='left')

    if 'region' not in df.columns or df['region'].isna().all():
        if 'Team' in df.columns and df['Team'].notna().any():
            df['region'] = df['Team']
        else:
            df['region'] = df['NOC']

    if 'Medal' in df.columns:
        dummies = pd.get_dummies(df['Medal'], dtype='int8')
    else:
        dummies = pd.DataFrame(index=df.index)

    for c in ['Gold','Silver','Bronze']:
        if c not in dummies.columns:
            dummies[c] = 0

    df = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    if df.empty:
        raise ValueError("After preprocessing the dataframe is empty. Please check the CSV contents.")

    return df


def country_year_list(df):
    # ... (Keep country_year_list logic) ...
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return ['Overall'], ['Overall']
        years = []
        if 'Year' in df.columns:
            vals = df['Year'].dropna().unique().tolist()
            years_ints = []
            for v in vals:
                try:
                    years_ints.append(int(v))
                except Exception:
                    try:
                        years_ints.append(int(str(v)))
                    except Exception:
                        pass
            years_ints = sorted(list(set(years_ints)))
            years = [str(y) for y in years_ints]
        years = ['Overall'] + years if years else ['Overall']
        if 'region' in df.columns:
            countries = sorted([str(x) for x in df['region'].dropna().unique().tolist()])
        elif 'Team' in df.columns:
            countries = sorted([str(x) for x in df['Team'].dropna().unique().tolist()])
        else:
            countries = []
        countries = ['Overall'] + countries if countries else ['Overall']
        return years, countries
    except Exception as e:
        st.error(f"Internal error building dropdown lists: {e}")
        return ['Overall'], ['Overall']

def fetch_medal_tally(df, year, country):
    # ... (Keep fetch_medal_tally logic) ...
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    uniq = df.drop_duplicates(subset=[c for c in ['Team','NOC','Games','Year','City','Sport','Event','Medal'] if c in df.columns])
    if year != 'Overall' and 'Year' in uniq.columns:
        try:
            uniq = uniq[uniq['Year'] == int(year)]
        except Exception:
            return pd.DataFrame()
    if country != 'Overall':
        if 'region' in uniq.columns:
            uniq = uniq[uniq['region'] == country]
        else:
            return pd.DataFrame()
    if uniq.empty:
        return pd.DataFrame()
    numeric = uniq.select_dtypes(include=[np.number])
    for c in ['Gold','Silver','Bronze']:
        if c not in numeric.columns:
            uniq[c] = 0
    grouped = uniq.groupby('region' if 'region' in uniq.columns else 'Team')[['Gold','Silver','Bronze']].sum(numeric_only=True).reset_index()
    grouped['total'] = grouped[['Gold','Silver','Bronze']].sum(axis=1)
    return grouped.sort_values('Gold', ascending=False).reset_index(drop=True)

def yearwise_medal_tally(df, country):
    # ... (Keep yearwise_medal_tally logic) ...
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if country == 'Overall':
        return pd.DataFrame()
    if 'region' not in df.columns or 'Medal' not in df.columns:
        return pd.DataFrame()
    temp = df.dropna(subset=['Medal']).drop_duplicates(subset=[c for c in ['Team','NOC','Games','Year','City','Sport','Event','Medal'] if c in df.columns])
    temp = temp[temp['region'] == country]
    if temp.empty or 'Year' not in temp.columns:
        return pd.DataFrame()
    out = temp.groupby('Year').count()['Medal'].reset_index()
    out = out.sort_values('Year').reset_index(drop=True)
    return out

def country_event_heatmap(df, country):
    # ... (Keep country_event_heatmap logic) ...
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if country == 'Overall' or 'Sport' not in df.columns or 'Medal' not in df.columns:
        return pd.DataFrame()
    temp = df.dropna(subset=['Medal']).drop_duplicates(subset=[c for c in ['Team','NOC','Games','Year','City','Sport','Event','Medal'] if c in df.columns])
    temp = temp[temp['region'] == country]
    if temp.empty or 'Year' not in temp.columns:
        return pd.DataFrame()
    pt = temp.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)
    return pt

# --- MAIN DASHBOARD FUNCTION ---

def render_analysis_view(df):
    """Renders the full analytics dashboard with filtering and charts."""
    
    # Show debug info in sidebar (Moved from main.py)
    # Note: raw_ath_df/raw_reg_df are not available here. You may need to pass them 
    # if you want to keep the debug panel, but for now, we'll show processed info.
    with st.sidebar.expander("✅ Processed dataframe info"):
        try:
            st.write("Processed columns:", list(df.columns))
            st.write("Processed dtypes:")
            st.dataframe(pd.DataFrame(df.dtypes, columns=['dtype']).astype(str))
            st.write("Sample rows:")
            st.dataframe(df.head(5))
        except Exception as e:
            st.write("Could not show processed debug info:", e)

    try:
        years, countries = country_year_list(df)
    except Exception as e:
        st.error("Failed to build year/country dropdown lists.")
        st.text(traceback.format_exc())
        return

    # Wrap main content in overlay for better readability (Uses global .app-overlay CSS)
    st.markdown('<div class="app-overlay">', unsafe_allow_html=True)

    # UI selections
    st.title("🏆 Olympic Data Analytics")
    left_col, right_col = st.columns(2)
    with left_col:
        selected_year = st.selectbox("Select Year", years, index=0)
    with right_col:
        selected_country = st.selectbox("Select Country", countries, index=0)

    # ... (Keep all chart/tally rendering logic as is from bda.py) ...
    # Basic medal tally
    st.subheader("🥇 Medal Tally")
    try:
        tally = fetch_medal_tally(df, selected_year, selected_country)
        if tally is None or tally.empty:
            st.info("No medal data for the chosen filters.")
        else:
            st.dataframe(tally)
    except Exception as e:
        st.error("Error computing medal tally:")
        st.text(traceback.format_exc())

    # Year-wise trend if country selected
    if selected_country != 'Overall':
        st.subheader(f"📈 Medal Trend for {selected_country}")
        try:
            trend = yearwise_medal_tally(df, selected_country)
            if trend is None or trend.empty:
                st.info("No yearwise medal trend data available for this country.")
            else:
                fig = px.line(trend, x='Year', y='Medal', markers=True, title=f"Medals per Year — {selected_country}")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Error computing yearwise trend:")
            st.text(traceback.format_exc())

    # Additional features (guarded)
    st.markdown("---")
    st.header("Additional Analytics")

    # Top 10 countries
    try:
        if 'region' in df.columns and all(c in df.columns for c in ['Gold','Silver','Bronze']):
            totals = df.groupby('region')[['Gold','Silver','Bronze']].sum(numeric_only=True)
            totals['Total'] = totals[['Gold','Silver','Bronze']].sum(axis=1)
            top10 = totals.sort_values('Total', ascending=False).head(10).reset_index()
            fig = px.bar(top10, x='region', y='Total', color='Gold', title='Top 10 Countries by Total Medals')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Top 10: required columns ('region','Gold','Silver','Bronze') not available.")
    except Exception:
        st.error("Error building Top 10 chart.")
        st.text(traceback.format_exc())

    # Gender distribution
    try:
        if 'Medal' in df.columns and 'Sex' in df.columns:
            gender_df = df.dropna(subset=['Medal']).groupby('Sex').count()['Medal'].reset_index()
            if not gender_df.empty:
                fig = px.pie(gender_df, names='Sex', values='Medal', title='Medal distribution by gender')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No gender-medal records found.")
        else:
            st.info("Gender chart: 'Sex' or 'Medal' column missing.")
    except Exception:
        st.error("Error building gender chart.")
        st.text(traceback.format_exc())

    # Age distribution
    try:
        if 'Age' in df.columns and 'Medal' in df.columns:
            age_df = df.dropna(subset=['Age','Medal'])
            if not age_df.empty:
                fig = px.histogram(age_df, x='Age', nbins=30, color='Medal', title='Age distribution of medal winners')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No age data for medal winners.")
        else:
            st.info("Age histogram: 'Age' or 'Medal' column missing.")
    except Exception:
        st.error("Error building age histogram.")
        st.text(traceback.format_exc())

    # Sports popularity
    try:
        if 'Sport' in df.columns and 'Year' in df.columns:
            sports_df = df.drop_duplicates(subset=['Year','Sport','Name'])
            sport_counts = sports_df.groupby(['Year','Sport']).count()['Name'].reset_index()
            top_k = st.slider("Number of sports to plot (by participants)", 3, 12, 6)
            totals_by_sport = sport_counts.groupby('Sport')['Name'].sum().sort_values(ascending=False).head(top_k).index.tolist()
            filt = sport_counts[sport_counts['Sport'].isin(totals_by_sport)]
            fig = px.line(filt, x='Year', y='Name', color='Sport', markers=True, title='Participation Trend for Top Sports')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sports trend: need 'Sport', 'Year', and 'Name' columns.")
    except Exception:
        st.error("Error building sports popularity chart.")
        st.text(traceback.format_exc())

    # Country heatmap
    try:
        if selected_country != 'Overall':
            heat = country_event_heatmap(df, selected_country)
            if not heat.empty:
                plt.figure(figsize=(12,5))
                # Use st.pyplot wrapper for matplotlib plots
                st.pyplot(sns.heatmap(heat, annot=True, fmt="g", cmap="YlGnBu").figure) 
            else:
                st.info("No heatmap data for selected country.")
        else:
            st.info("Select a specific country to see its heatmap.")
    except Exception:
        st.error("Error building heatmap.")
        st.text(traceback.format_exc())

    # Country comparison (simple)
    try:
        regions = sorted([r for r in df['region'].dropna().unique().tolist()]) if 'region' in df.columns else []
        if len(regions) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                country_a = st.selectbox("Country A", ['--'] + regions, index=0)
            with c2:
                country_b = st.selectbox("Country B", ['--'] + regions, index=1 if len(regions)>1 else 0)
            if country_a != '--' and country_b != '--' and country_a != country_b:
                comp = df[df['region'].isin([country_a, country_b])]
                comp_medals = comp.groupby('region')[['Gold','Silver','Bronze']].sum(numeric_only=True).reset_index()
                comp_medals['Total'] = comp_medals[['Gold','Silver','Bronze']].sum(axis=1)
                fig = px.bar(comp_medals, x='region', y=['Gold','Silver','Bronze'], title=f"{country_a} vs {country_b} medals")
                st.plotly_chart(fig, use_container_width=True)
                ta = yearwise_medal_tally(df, country_a)
                tb = yearwise_medal_tally(df, country_b)
                if not ta.empty or not tb.empty:
                    ta['Country'] = country_a
                    tb['Country'] = country_b
                    combined = pd.concat([ta, tb], ignore_index=True)
                    fig2 = px.line(combined, x='Year', y='Medal', color='Country', markers=True, title='Medal trend comparison')
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Pick two different countries to compare.")
        else:
            st.info("Not enough countries present to compare.")
    except Exception:
        st.error("Error in country comparison.")
        st.text(traceback.format_exc())

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("If you still see an error, copy the full traceback (from the top of the terminal) and paste it here.")
    
# End of analysis_view.py