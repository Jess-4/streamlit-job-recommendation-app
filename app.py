import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# load the dataset
@st.cache_data
def load_data():
    file_path = "job_recommendation_dataset.csv"
    df = pd.read_csv(file_path)
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    return df

df = load_data()

# creat the label
tab1, tab2 = st.tabs(["📊 Data Analysis", "🌍 HeatMap"])

# data analysis
with tab1:
    st.title("💼 AI Job Recommendation")
    st.markdown("### 📊 Data Analysis - Salary & Industry Screening")

# sidebar: filter
    st.sidebar.header("🔍 Filter criteria")

    min_salary, max_salary = int(df["Salary"].min()), int(df["Salary"].max())

    salary_range = st.sidebar.slider(
        "💰 Choose the range of salary:",
        min_value=min_salary,
        max_value=max_salary,
        value=(min_salary, max_salary),
        step=1000
    )

    industries = df["Industry"].unique().tolist()
    industries.insert(0, "All")

    selected_industry = st.sidebar.selectbox("🏢 Select industry:", industries)

# filter the data
    filtered_df = df[(df["Salary"] >= salary_range[0]) & (df["Salary"] <= salary_range[1])]
    if selected_industry != "All":
        filtered_df = filtered_df[filtered_df["Industry"] == selected_industry]

# show the visuals
    st.subheader("📋 Filtered Data")
    st.write(filtered_df[["Company", "Required Skills", "Industry", "Job Title", "Salary"]])

# bar chart
    st.subheader("📊 Average salary in different industries")
    industry_salary = filtered_df.groupby("Industry")["Salary"].mean().reset_index()
    fig1 = px.bar(industry_salary, x="Industry", y="Salary", title="Average Salary by Industry", color="Salary")
    st.plotly_chart(fig1)

# scatter 
    st.subheader("📈 Salary distribution by company")
    fig2 = px.scatter(filtered_df, x="Company", y="Salary", color="Industry", title="Salary by Company")
    st.plotly_chart(fig2)

# skill bar chart
    skills = df["Required Skills"].dropna().str.split(", ")
    flat_skills = [s for sublist in skills for s in sublist]
    skill_counts = Counter(flat_skills)   
    df_skill = pd.DataFrame(skill_counts.items(), columns=["Skill", "Count"])
    fig = px.bar(
        df_skill.sort_values("Count", ascending=True).tail(10),  
        x="Count",
        y="Skill",
        orientation='h', 
        color="Skill",  
        color_discrete_sequence=px.colors.qualitative.Pastel,  # 颜色美化：Pastel风格
        title="Top 10 Most Required Skills"
    )
    
    # layout beauty
    fig.update_layout(
        xaxis_title="Frequency",
        yaxis_title="Skill",
        template="plotly_white",  
        title_font_size=20,
        margin=dict(t=60, b=40)
    )
        
    st.plotly_chart(fig)


# word cloud
    st.subheader("☁️ Most Common Job Titles (Word Cloud)")

    text = " ".join(title for title in df["Job Title"].dropna())

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",  # 或 "plasma", "inferno"
        max_words=100
    ).generate(text)  
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)



# heatmap
with tab2:
        # page title
    st.title("📊 Job Market Heatmap")
    
    # upload file again
    file_path = "job_recommendation_dataset.csv"
    df = pd.read_csv(file_path)
    
        # comsistent the column name
    df.columns = df.columns.str.strip().str.lower()
    
        # show the column name
    st.write("✅ Processed Columns:")
    st.write(df.columns.tolist())
    
        # city coordinate point
    location_coords = {
        "new york": (40.7128, -74.0060),
        "san francisco": (37.7749, -122.4194),
        "berlin": (52.52, 13.4050),
        "sydney": (-33.8688, 151.2093),
        "london": (51.5074, -0.1278),
        "paris": (48.8566, 2.3522),
        "toronto": (43.651070, -79.347015),
        "chicago": (41.8781, -87.6298),
        "tokyo": (35.6762, 139.6503),
        "singapore": (1.3521, 103.8198),
        "bangalore": (12.9716, 77.5946)
    }
    
        # standardization the location
    df["location"] = df["location"].str.strip().str.lower()
    
        # add latitude and longtitude
    df["latitude"] = df["location"].map(lambda x: location_coords.get(x, (None, None))[0])
    df["longitude"] = df["location"].map(lambda x: location_coords.get(x, (None, None))[1])
    
        # Mapping experience levels
    experience_mapping = {
        "entry level": 1,
        "mid level": 2,
        "senior level": 3,
        "executive": 4
    }
    df["experience level"] = df["experience level"].str.strip().str.lower()
    df["experience_level_numeric"] = df["experience level"].map(experience_mapping).fillna(0)
    
        # Aggregate data
    agg_df = df.groupby("location").agg({
        "job title": "count",
        "salary": "mean",
        "experience_level_numeric": "mean",
        "latitude": "first",
        "longitude": "first"
    }).reset_index()
    
    agg_df.rename(columns={
        "job title": "job quantity",
        "experience_level_numeric": "experience level"
    }, inplace=True)
    metric = st.selectbox(
        "Select a metric for the heatmap",
        ["salary", "experience level", "job quantity"]
    )
    agg_df["size"] = agg_df[metric] * 2
    
        # draw the heatmap
    fig = px.scatter_geo(
        agg_df,
        lat="latitude",
        lon="longitude",
        size="size",
        color=metric,
        hover_name="location",
        projection="natural earth",
        title=f"Heatmap of {metric.title()} by Location",
        color_continuous_scale="Blues"
    )
    
    fig.update_traces(marker=dict(line=dict(width=0)))
    
    st.plotly_chart(fig, use_container_width=True)
        
# footer
st.markdown("**📌 Data source: [Kaggle Dataset](https://www.kaggle.com/datasets/samayashar/ai-powered-job-recommendations/data)**")
