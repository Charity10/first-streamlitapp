from multiprocessing.sharedctypes import Value
from optparse import Values
from re import X
from turtle import color, fillcolor
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

background_color = '#E8E8EC'
header= st.container()
data = st.container()
visual= st.container()
mod_train = st.container()


def get_data(filename):
    data = pd.read_csv(filename)

    return data

with header:
    st.title('Technical support data')
    image = Image.open('cus.jpg')
    st.image(image, use_column_width= True)
    st.write('This data is about the complains made by customers to a certain company')

with data:
    st.header('Our Dataset')
    tec = get_data('technical_support_data-2.csv')
    st.write(tec)
    col,row = st.columns(2)
    col.write('Columns in the dataset')
    cols =  col.write(tec.columns)
    rows = row.write(tec.iloc[:,:-7])
    row.write('Rows in the dataset')
    st.write(cols)
    st.write(rows)

with visual:
    fig = go.Figure(data = go.Table( header = dict(values = list(tec[['no_of_cases', 'Avg_pending_calls', 'Avg_resol_time', 'recurrence_freq']].columns)), cells = dict(values = [tec.no_of_cases, tec.Avg_pending_calls, tec.Avg_resol_time, tec.recurrence_freq],)))
    fig.update_layout(margin = dict(l = 50, r = 40, b =20 , t = 10 ), paper_bgcolor = background_color)
    st.write(fig)
    cht, pie = st.columns(2)
    ptec = tec.iloc[:,:2]

    top_n = cht.text_input('how many number of problems would you like to plot?', 10)
    slid_n = cht.slider('slide your desired num', 0,22 )
    sel = st.selectbox('pick', (top_n, slid_n))
    if sel == top_n:
        top_n = int(top_n)
        ptech = ptec.head(top_n)
        fig2 = px.pie(ptech, values = 'no_of_cases', names = 'PROBLEM_TYPE')
        fig2.update_layout()
        pie.write(fig2)
    elif sel == slid_n:
        #slid_n = int(top_n)
        ptech = ptec.head(slid_n)
        fig2 = px.pie(ptech, values = 'no_of_cases', names = 'PROBLEM_TYPE')
        fig2.update_layout()
        pie.write(fig2)

    

    #crosstab to campare two columns
    #fig.update_axis to update axis
    #fig3 = px.line(animation_frame ='', animation_graph ='' )


with mod_train:
    st.write('Output')
    x = tec.iloc[:,1:]
    y = tec['PROBLEM_TYPE']
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, random_state = 10)
    x2 = x.apply(zscore)
    mod = KMeans(4)
    mod.fit(x2)
    pred = mod.predict(x2)
    tec['clusters'] = pred
    st.write(tec)
    one = tec[tec['clusters']== 1]
    st.write('Cluster  ==  1')
    st.write(one)
    two = tec[tec['clusters']== 2]
    st.write('Cluster  ==  2')
    st.write(two)
    thr = tec[tec['clusters']== 3]
    st.write('Cluster  ==  3')
    st.write(thr)
    fou = tec[tec['clusters']== 4]
    st.write('Cluster  ==  4')
    st.write(fou)




