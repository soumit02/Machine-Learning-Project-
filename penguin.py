import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier # pip install scikit-learn==1.2.2
import altair as alt
from sklearn.preprocessing import LabelEncoder
from pandas.core.dtypes.common import is_numeric_dtype
import pickle


st.title("Penguin Species Prediction")
df=pd.read_csv("penguins_data.csv")
button=st.sidebar.radio("Context",["Home","Prediction"])
if button=="Home":
    st.image("penguin.jpg",width=500)
    if st.checkbox("Show Dataset"):
        st.table(df)
    habitat = pd.DataFrame({
        'country': ['Antarctica', 'Angola', 'Argentina', 'Australia', 'Chile', 'Namibia', 'New Zealand',
                    'South Africa'],
        'lat': [-76.299965, -8.838333, -34.603722, -33.865143, -33.447487, -22.601255, -36.848461, -26.195246],
        'lon': [-148.003021, 13.234444, -58.381592, 151.209900, -70.673676, 17.092333, 174.763336, 28.034088]
    })
    st.map(habitat)

if button=="Prediction":
    st.header("Species Prediction")
    def user_input_features():
        island = st.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.selectbox('Sex',('male','female'))
        bill_length_mm = st.number_input('Bill length (mm)', 0.00,500.55,step=0.50)
        bill_depth_mm = st.number_input('Bill depth (mm)', 0.00,500.55,step=0.50)
        flipper_length_mm = st.number_input('Flipper length (mm)', 0.00,500.55,step=0.50)
        body_mass_g = st.number_input('Body mass (g)', 0.00,50000.55,step=0.50)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

    x=df.drop(columns=['species'])
    df2 = pd.concat([input_df, x], axis=0)

    le=LabelEncoder()
    for i in df2.columns:
        if is_numeric_dtype(df2[i]):
            continue
        else:
            df2[i] = le.fit_transform(df2[i])

    df2 = df2[:1]
    clf2=pickle.load(open("penguin_project","rb"))
    #st.subheader("Prediction")

    prediction = clf2.predict(df2)[0]
    prediction_proba = clf2.predict_proba(df2)

    penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
    if st.button("Predict"):
        st.success(f"Your predicted species is {penguins_species[prediction]}")
        #st.subheader('Prediction')

        #st.write(penguins_species[prediction])

        st.subheader('Prediction Probability')
        st.write(prediction_proba)


