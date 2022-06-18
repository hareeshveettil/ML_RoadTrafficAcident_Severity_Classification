
import streamlit as st
from PIL import Image
from src.predict import get_prediction, ordinal_encoder
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
import warnings
from sklearn.exceptions import DataConversionWarning
from src.load_model import get_model
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

st.set_page_config(page_title="Road Accident Severity Predictor",             
                layout="wide")

st.markdown("<h1 align='center'>Road Accident Severity Prediction </h1>", unsafe_allow_html=True)
st.image("https://images.news18.com/ibnlive/uploads/2021/11/road-accident-163747048616x9.jpg", use_column_width=True)

st.sidebar.title("About this application")
st.sidebar.write("""
        The app is aimed at predicting the Road accident severity based on the features of the accident.
        This data set is collected from Addis Ababa Sub-city police departments for master's research work.
        The data set has been prepared from manual records of road traffic accidents of the year 2017-20. All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident. Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms
    
        """)

st.sidebar.info("### Made by:    Bhanumathi Ramesh")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/bhanumathiramesh)")
st.sidebar.markdown("[Github](https://github.com/bhanu0925/ML_RoadTrafficAcident_Severity_Classification)")


days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
cause = ['Changing lane to the left', 'Changing lane to the right', 'Driving at high speed', 'Driving carelessly', 'Driving to the left', 'Driving under the influence of drugs', 'Drunk driving', 'Getting off the vehicle improperly', 'Improper parking', 'Moving Backward', 'No distancing', 'No priority to pedestrian', 'No priority to vehicle', 'Other', 'Overloading', 'Overspeed', 'Overtaking', 'Overturning', 'Turnover', 'Unknown']
lightcondition = ['Darkness - lights lit', 'Darkness - lights unlit', 'Darkness - no lighting', 'Daylight']
ageOfDriver = ['18-30', '31-50', 'Over 51', 'Under 18', 'Unknown']
educationOfDriver = ['Above high school', 'Elementary school', 'High school', 'Illiterate', 'Junior high school', 'Unknown', 'Writing & reading']
experienceOfDriver = ['1-2yr', '2-5yr', '5-10yr', 'Above 10yr', 'Below 1yr', 'No Licence', 'unknown']
typeOfVeh = ['Automobile', 'Bajaj', 'Bicycle', 'Long lorry', 'Lorry (11?40Q)', 'Lorry (41?100Q)', 'Motorcycle', 'Other', 'Pick up upto 10Q', 'Public (12 seats)', 'Public (13?45 seats)', 'Public (> 45 seats)', 'Ridden horse', 'Special vehicle', 'Stationwagen', 'Taxi', 'Turbo']
lanesOrMedians = ['Double carriageway (median)', 'One way', 'Two-way (divided with broken lines road marking)', 'Two-way (divided with solid lines road marking)', 'Undivided Two way', 'Unknown', 'other']
typesofjunction = ['Crossing', 'No junction', 'O Shape', 'Other', 'T Shape', 'Unknown', 'X Shape', 'Y Shape']
roadsurfacecondition = ['Dry', 'Flood over 3cm. deep', 'Snow', 'Wet or damp']
areaaccidentoccured = ['Residential areas', 'Office areas', '  Recreational areas',
       ' Industrial areas', 'Other', ' Church areas',
       '  Market areas', 'Unknown', 'Rural village areas',
       ' Outside rural areas', ' Hospital areas', 'School areas',
       'Rural village areasOffice areas', 'Recreational areas'] 
typeofcollision = ['Collision with animals', 'Collision with pedestrians', 'Collision with roadside objects', 'Collision with roadside-parked vehicles', 'Fall from vehicles', 'Other', 'Rollover', 'Unknown', 'Vehicle with vehicle collision', 'With Train']




st.markdown("<h2 align='center'> Road Accident Severity Prediction Output </h2>", unsafe_allow_html=True)



with st.form("Prediction_form"):
    col1,col2 = st.columns(2)
    day = col1.selectbox("Day of the Week: ", options = days )
    causeofaccident = col2.selectbox("Cause of Aaccident : ", options = cause )
    num_of_casualities = col1.number_input("No. of Casualities", min_value=0)
    vehicles_involved = col2.number_input("No. of vehicles Involved", min_value=0)
    light = col1.selectbox("Light Conditions : ", options = lightcondition )
    age = col2.selectbox("Age of Driver : ", options = ageOfDriver)
    education = col1.selectbox("Education of Driver : ", options = educationOfDriver)
    experience = col2.selectbox("Experience Of Driver :", options = experienceOfDriver)
    typeofveh = col1.selectbox("Type of Vehicle : ", options = typeOfVeh)
    lanes = col2.selectbox("Lanes or Medians : ", options = lanesOrMedians)
    junctions = col1.selectbox("Types of Junctions : ", options = typesofjunction)
    roadsurface = col2.selectbox("Road Surface Conditions : ", options = roadsurfacecondition)
    areaaccoccured = col1.selectbox("Area Accident Occured : ", options = areaaccidentoccured)
    collision = col2.selectbox("Type of Collision : ", options = typeofcollision)


    if st.form_submit_button("Predict") :
        day_of_week = ordinal_encoder(day,days)
        Cause_Of_Accident = ordinal_encoder(causeofaccident,cause)
        light_conditions = ordinal_encoder(light,lightcondition)
        age_of_driver = ordinal_encoder(age,ageOfDriver)
        education_of_driver = ordinal_encoder(education,educationOfDriver)
        experience_of_driver = ordinal_encoder(experience,experienceOfDriver)
        type_of_vehicle = ordinal_encoder(typeofveh,typeOfVeh)
        lanes = ordinal_encoder(lanes,lanesOrMedians)
        type_of_junction = ordinal_encoder(junctions,typesofjunction)
        road_surface = ordinal_encoder(roadsurface,roadsurfacecondition)
        area_acc_Occured = ordinal_encoder(areaaccoccured,areaaccidentoccured)
        type_of_collision = ordinal_encoder(collision,typeofcollision)
        
        
        
        data = np.array([day_of_week,Cause_Of_Accident,num_of_casualities,vehicles_involved, 
                             light_conditions,age_of_driver,education_of_driver,experience_of_driver,
                             type_of_vehicle,lanes,type_of_junction,road_surface,area_acc_Occured,type_of_collision]).reshape(1,-1)

       # model = joblib.load(r'model/rt_reduced.joblib')
       
        #joblib.dump(model, open(r'model/RT_rePickle.joblib', 'wb'),compress=3)
        
        ## remodel = joblib.load(r'model/RT_rePickle.joblib')
        ##model = get_model2()
        model = get_model(model_path =r'model/RT_rePickle.joblib' )
        pred = get_prediction(data=data, model=model)
        
        st.write(f"The predicted severity is : {pred}")
     
        ## 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
        ## https://medium.com/analytics-vidhya/deploying-nlp-model-on-heroku-using-flask-nltk-and-git-lfs-eed7d1b22b11
        
        