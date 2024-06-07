import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title, hide_pages

add_page_title()

show_pages(
    [   
        Page("dezoomcamp/DE_Zoomcamp.py", "Data  Visualization", "📊"),

        # 2024 Content
        #Section("DE Zoomcamp 2024", "🧙‍♂️"),
        Page("dezoomcamp/2024_cohort/Course_Overview.py", "Property Price Prediction Project", "🏠", in_section=True),
        #Page("dezoomcamp/2024_cohort/Module_1_Introduction_&_Prerequisites.py", "Module 1 Introduction & Prerequisites", "1️⃣", in_section=True),
        #Page("dezoomcamp/2024_cohort/Module_2_Workflow_Orchestration.py", "Module 2 Workflow Orchestration", "2️⃣", in_section=True),
        #Page("dezoomcamp/2024_cohort/Workshop_1_Data_Ingestion.py", "Workshop 1 Data Ingestion", "🛠️", in_section=True),
        #Page("dezoomcamp/2024_cohort/Module_3_Data_Warehouse.py", "Module 3 Data Warehouse and BigQuery", "3️⃣", in_section=True),
        #Page("dezoomcamp/2024_cohort/Module_4_Analytics_Engineering.py", "Module 4 Analytics Engineering", "4️⃣", in_section=True),
        #Page("dezoomcamp/2024_cohort/Module_5_Batch_Processing.py", "Module 5 Batch Processing", "5️⃣", in_section=True),
        #Page("dezoomcamp/2024_cohort/Workshop_2_Stream_Processing_with_SQL.py", "Workshop 2 Stream Processing with SQL", "🛠️", in_section=True),
        #Page("dezoomcamp/2024_cohort/Module_6_Stream_Processing.py", "Module 6 Stream Processing", "6️⃣", in_section=True),
        #Page("dezoomcamp/2024_cohort/Course_Project.py", "Course Project", "🏆", in_section=True),

        # 2023 Content
        #Section("DE Zoomcamp 2023", "👨‍🔧"),
        #Page("dezoomcamp/2023_cohort/Course_Overview.py", "Course Overview", "📚", in_section=True),
        #Page("dezoomcamp/2023_cohort/Week_1_Introduction_&_Prerequisites.py", "Week 1 Introduction & Prerequisites", "1️⃣", in_section=True),
        #Page("dezoomcamp/2023_cohort/Week_2_Workflow_Orchestration.py", "Week 2 Workflow Orchestration", "2️⃣", in_section=True),
        #Page("dezoomcamp/2023_cohort/Week_3_Data_Warehouse.py", "Week 3 Data Warehouse", "3️⃣", in_section=True),
        #Page("dezoomcamp/2023_cohort/Week_4_Analytics_Engineering.py", "Week 4 Analytics Engineering", "4️⃣", in_section=True),
        #Page("dezoomcamp/2023_cohort/Week_5_Batch_Processing.py", "Week 5 Batch Processing", "5️⃣", in_section=True),
        #Page("dezoomcamp/2023_cohort/Week_6_Stream_Processing.py", "Week 6 Stream Processing", "6️⃣", in_section=True),
        #Page("dezoomcamp/2023_cohort/Week_7_Project.py", "Week 7 Project", "7️⃣", in_section=True),
        #Page("dezoomcamp/2023_cohort/Homework_Quizzes.py", "Homework Quizzes", "📝", in_section=True),
        #
        #Page("dezoomcamp/Datasets.py", "Datasets", icon="💾", in_section=False),
        #Page("dezoomcamp/Certificate.py", "Certificate", "📜", in_section=False),
        #Page("dezoomcamp/FAQ.py", "FAQ", "❔", in_section=False),
        #Page("dezoomcamp/Contact.py", "Contact", icon="📩", in_section=False),   
        #Page("dezoomcamp/Contact_thanks.py", "Thank you", icon="💌"),   
        #Page("dezoomcamp/About.py", "About", icon="🖼️", in_section=False) 
    ]
)

hide_pages(["Thank you"])

st.markdown("### 📈 Data Science Projects by [Chibudom Obasi](https://journeygenius.pythonanywhere.com/ml)")

#st.image("https://pbs.twimg.com/media/FmmYA2YWYAApPRB.png")

st.info("You can find me on [TikTok](https://tiktok.com/@mobile.desk) |  [YouTube](https://www.youtube.com/c/Chibudomobasi?app=desktop&sub_confirmation=1) | [LinkedIn](https://www.linkedin.com/in/praise-obasi)")

st.markdown("---")

with st.expander("Sign up here for Data Decoded"):
    st.markdown("""
    
    <a href="https://www.udemy.com/course/data-decoded/?referralCode=B51A455EEDEF74E4DB12"><img src="https://user-images.githubusercontent.com/875246/185755203-17945fd1-6b64-46f2-8377-1011dcb1a444.png" height="50" /></a>

    

    - Register and get amazing discounts
    - The videos are published on [Chibudom Obasi's YouTube channel](https://www.youtube.com/c/Chibudomobasi?app=desktop&sub_confirmation=1) 
    #""", unsafe_allow_html=True)

st.markdown("""
### 👨‍🎓 About the projects

##### Write Ups

Details about each projects are on my [portfolio](https://journeygenius.pythonanywhere.com/ml) .

This site is used to display an interactive dashboard for each project.


""", unsafe_allow_html=True)


#st.image("https://raw.githubusercontent.com/DataTalksClub/data-engineering-zoomcamp/main/images/architecture/photo1700757552.jpeg")


