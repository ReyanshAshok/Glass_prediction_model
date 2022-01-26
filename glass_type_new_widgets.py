import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
 
# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
@st.cache()
def prediction(model,feature_list):
    glass_type = model.predict([feature_list])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "building windows float processed".upper()
    elif glass_type == 2:
        return "building windows non float processed".upper()
    elif glass_type == 3:
        return "vehicle windows float processed".upper()
    elif glass_type == 4:
        return "vehicle windows non float processed".upper()
    elif glass_type == 5:
        return "containers".upper()
    elif glass_type == 6:
        return "tableware".upper()
    else:
        return "headlamps".upper()
st.title("Glass Type prediction Web app")
st.sidebar.title("Glass Type prediction Web app")
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Glass Type Data set")
    st.dataframe(glass_df)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.subheader('Visualization Selector')
plot_list=st.sidebar.multiselect('Select tthe Charts/Plots',('HeatMap','LineChart','AreaChart','CountPlot','PieChart','BoxPlot'))
if 'HeatMap' in plot_list:
  # plot correlation heatmap
  st.subheader('Correlation HeatMap')
  plt.figure(figsize=(15,6))
  sns.heatmap(glass_df.corr(),annot=True)
  st.pyplot()
if 'LineChart' in plot_list:
  st.subheader('LineChart')
  st.line_chart(glass_df)
  # plot line chart 
if 'AreaChart' in plot_list:
  st.subheader('AreaChart')
  st.area_chart(glass_df)
  # plot area chart 
elif 'CountPlot' in plot_list:
  st.subheader('CountPlot')
  sns.countplot(glass_df['GlassType'])
  st.pyplot()
  # plot count plot
if 'PieChart' in plot_list:
  st.subheader('PiChart')
  plt.pie(glass_df['GlassType'].value_counts(),labels=glass_df['GlassType'].value_counts().index,autopct='%1.2f%%')
  st.pyplot()
if 'BoxPlot' in plot_list:
	st.subheader('BoxPlot')
	column=st.sidebar.selectbox('Select the Coloumn',list(glass_df.columns[:-1]))
	sns.boxplot(glass_df[column])
	st.pyplot()
feature_list=[]
for i in list(glass_df.columns[:-1]):
	user_input = st.sidebar.slider(f"Input {i}", float(glass_df[i].min()), float(glass_df[i].max()))
	feature_list.append(user_input)
st.sidebar.subheader('Classifier Options')
classifier=st.sidebar.selectbox('Choose Classifier',('Support Vector Machines','RandomForestClassifier','LogisticRegression'))
if classifier=='Support Vector Machines':
	st.subheader('Model Hyper Parameter')
	c_value=st.sidebar.number_input('Error_rate',1,100,step=1)
	gamma_value=st.sidebar.number_input('Gamma',1,100,step=1)
	kernel_value=st.sidebar.radio('Kernel Input',('linear','rbf','poly'))
	if st.sidebar.button('Predict'):
		svc_obj=SVC(C=c_value,kernel=kernel_value,gamma=gamma_value)
		svc_obj.fit(X_train,y_train)
		st.write(svc_obj.score(X_test,y_test))
		test_pred=svc_obj.predict(X_test)
		plot_confusion_matrix(svc_obj,X_test,y_test)
		st.pyplot()
		glass_type=prediction(svc_obj,feature_list)
		st.write(glass_type)
if classifier=='LogisticRegression':
	c_value=st.sidebar.number_input('Error_rate',1,100,step=1)
	max_t=st.sidebar.slider('Maximum Iterations',0,100)
	if st.sidebar.button('Predict'):
		log_obj=LogisticRegression(C=c_value,max_iter=max_t)
		log_obj.fit(X_train,y_train)
		st.write(log_obj.score(X_train,y_train))
		test_pred=log_obj.predict(X_test)
		plot_confusion_matrix(log_obj,X_test,y_test)
		st.pyplot()
		glass_type=prediction(log_obj,feature_list)
		st.write(glass_type)
if classifier=='RandomForestClassifier':
	n=st.sidebar.number_input('N estimator',1,100,step=1)
	max_d=st.sidebar.slider('Maximum Depth',0,100)
	if st.sidebar.button('Predict'):
		rfc_obj=RandomForestClassifier(n,max_depth=max_d)
		rfc_obj.fit(X_train,y_train)
		st.write(rfc_obj.score(X_train,y_train))
		test_pred=rfc_obj.predict(X_test)
		plot_confusion_matrix(rfc_obj,X_test,y_test)
		st.pyplot()
		glass_type=prediction(rfc_obj,feature_list)
		st.write(glass_type)


