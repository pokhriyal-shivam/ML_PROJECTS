import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np


st.set_page_config(
    page_title="AeroSense AI",
    page_icon="🌍",
    layout="wide"
)


st.markdown("""
<style>

.stApp{
    background:#050816;
}


.stApp:before{
    content:"";
    position:fixed;
    width:200%;
    height:200%;
    top:-50%;
    left:-50%;
    background:
    radial-gradient(circle,#0ea5e9 1px,transparent 1px);
    background-size:45px 45px;
    animation:move 25s linear infinite;
    opacity:0.15;
    z-index:-1;
}


@keyframes move{
    from{
        transform:translate(0,0);
    }
    to{
        transform:translate(100px,100px);
    }
}


.title{

font-size:55px;
font-weight:900;
text-align:center;

background:
linear-gradient(
90deg,
#00ffff,
#3b82f6,
#a855f7
);

-webkit-background-clip:text;
color:transparent;

}


.subtitle{

text-align:center;
font-size:20px;
color:#94a3b8;
margin-bottom:35px;

}


.glass{

background:rgba(15,23,42,0.75);
border:1px solid rgba(255,255,255,0.15);
backdrop-filter:blur(20px);

border-radius:25px;
padding:25px;

box-shadow:
0 0 40px rgba(0,200,255,0.15);

}



.aqi-box{

background:
linear-gradient(
135deg,
#2563eb,
#a855f7
);

border-radius:35px;
padding:40px;

text-align:center;

box-shadow:
0 0 50px rgba(59,130,246,.5);

}



.aqi-value{

font-size:85px;
font-weight:900;
color:white;

animation:pulse 2s infinite;

}



@keyframes pulse{

0%{
text-shadow:0 0 10px white;
}

50%{
text-shadow:0 0 40px cyan;
}

100%{
text-shadow:0 0 10px white;
}

}



.metric{

background:
rgba(255,255,255,0.06);

border-radius:20px;

padding:20px;

text-align:center;

border:
1px solid rgba(255,255,255,.1);

}



.metric-number{

font-size:35px;
font-weight:900;
color:#38bdf8;

}



.metric-name{

color:#cbd5e1;

}



.stButton button{

height:60px;

width:100%;

border-radius:20px;

background:
linear-gradient(
90deg,
#06b6d4,
#2563eb
);

color:white;

font-size:22px;

font-weight:800;

border:none;

}



.stButton button:hover{

transform:scale(1.03);

box-shadow:
0 0 30px cyan;

}

</style>

""",unsafe_allow_html=True)



class AQI_Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.network=nn.Sequential(

            nn.Linear(11,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,1)

        )


    def forward(self,x):

        return self.network(x)



@st.cache_resource
def load_model():

    model=AQI_Model()

    model.load_state_dict(
        torch.load(
            "aqi_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model



@st.cache_resource
def load_scaler():

    return joblib.load(
        "X_scaler.pkl"
    )



model=load_model()

scaler=load_scaler()



st.markdown(
"""
<div class="title">
🌍 AeroSense AI
</div>

<div class="subtitle">
Deep Learning Air Quality Intelligence System
</div>
""",
unsafe_allow_html=True
)



st.markdown(
"""
<div class="glass">

<h2>
🌫 Atmospheric Data Input
</h2>

</div>
""",
unsafe_allow_html=True
)



c1,c2,c3=st.columns(3)



with c1:

    pm25=st.number_input(
        "PM2.5",
        value=50.0
    )

    pm10=st.number_input(
        "PM10",
        value=80.0
    )



with c2:

    no2=st.number_input(
        "NO2",
        value=20.0
    )

    so2=st.number_input(
        "SO2",
        value=10.0
    )



with c3:

    co=st.number_input(
        "CO",
        value=1.0
    )

    ozone=st.number_input(
        "Ozone",
        value=50.0
    )



st.markdown(
"""
<div class="glass">

<h2>
📅 Time Information
</h2>

</div>
""",
unsafe_allow_html=True
)



a,b,c,d,e=st.columns(5)


with a:
    date=st.number_input("Date",1,31,1)

with b:
    month=st.number_input("Month",1,12,1)

with c:
    year=st.number_input("Year",2000,2100,2025)

with d:
    holidays=st.number_input("Holidays",0,30,0)

with e:
    days=st.number_input("Days",1,365,1)



st.write("")


predict=st.button(
    "🚀 RUN AI PREDICTION"
)



if predict:


    data=np.array([[
        date,
        month,
        year,
        holidays,
        days,
        pm25,
        pm10,
        no2,
        so2,
        co,
        ozone
    ]])


    data=scaler.transform(data)


    tensor=torch.tensor(
        data,
        dtype=torch.float32
    )


    with torch.no_grad():

        prediction=model(tensor)


    aqi=prediction.item()



    if aqi<=50:

        status="🟢 GOOD AIR QUALITY"
        advice="Air quality is clean. Outdoor activities are safe."

    elif aqi<=100:

        status="🟡 MODERATE AIR QUALITY"
        advice="Air quality is acceptable. Sensitive people should be careful."

    elif aqi<=200:

        status="🟠 POOR AIR QUALITY"
        advice="Reduce prolonged outdoor exposure."

    elif aqi<=300:

        status="🔴 VERY POOR AIR QUALITY"
        advice="Avoid unnecessary outdoor activities."

    else:

        status="☠️ HAZARDOUS AIR QUALITY"
        advice="Stay indoors and avoid exposure."



    st.markdown(
    f"""

    <div class="aqi-box">

    <h2>
    LIVE AIR QUALITY INDEX
    </h2>


    <div class="aqi-value">
    {aqi:.1f}
    </div>


    <h2>
    {status}
    </h2>


    </div>

    """,

    unsafe_allow_html=True
    )



    st.write("")


    x1,x2,x3,x4=st.columns(4)


    values=[
        ("PM2.5",pm25),
        ("PM10",pm10),
        ("NO2",no2),
        ("CO",co)
    ]


    for col,(name,value) in zip(
        [x1,x2,x3,x4],
        values
    ):

        with col:

            st.markdown(
            f"""

            <div class="metric">

            <div class="metric-number">
            {value}
            </div>

            <div class="metric-name">
            {name}
            </div>

            </div>

            """,

            unsafe_allow_html=True
            )



    st.write("")



    st.markdown(
    f"""

    <div class="glass">

    <h2>
    🤖 AI Environmental Analysis
    </h2>


    <p style="font-size:20px;color:#cbd5e1">

    The neural network analyzed 11 atmospheric
    parameters and generated the AQI prediction.

    </p>


    <p style="font-size:20px;color:#38bdf8">

    Prediction Status:

    {status}

    </p>


    <p style="font-size:20px">

    Recommendation:

    {advice}

    </p>


    </div>

    """,

    unsafe_allow_html=True
    )