import os
import flask
from flask import Flask, request, render_template, send_file, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
# from imageai.Detection import ObjectDetection
import traceback
from fbprophet import Prophet
import pickle
import pandas as pd
import numpy as np
#import seaborn as sns
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pylab import rcParams

#GENDER AND AGE DETECTION
import cv2
import math
import argparse
import face_recognition
import base64

#ner
import spacy
from spacy import displacy
import en_core_web_sm
from bs4 import BeautifulSoup
import requests
import re
from IPython.core.display import display, HTML
from flaskext.markdown import Markdown

nlp = spacy.load('en_core_web_sm')

#sentiment analyser

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re, string
stop_words = stopwords.words('english')
from nltk.tokenize import word_tokenize
import joblib

classifier = joblib.load('model/sentimental_model.pkl')



#Customer Segmentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import warnings

# load the model from disk
loaded_model__ = joblib.load("model/customer_segmentation.pkl")


#ocr nlp

import easyocr

#car price

# Load the model from the file
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
import joblib 
import warnings

carPrice_model = joblib.load('model/Car_price.pkl')

#anomaly detection
from sklearn.ensemble import IsolationForest
#import seaborn as sns
from io import BytesIO # not necessary?

#pytesseract ocr
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#-----------

WORKING_DIRECTORY = os.getcwd()
FILE_UPLOAD_PATH = os.path.join(WORKING_DIRECTORY, 'inputed')
ALLOWED_EXTENSIONS = set(['csv', 'xls', 'xlsx', "png", "jpg"])
#model_anomaly = joblib.load('bivariate_anomaly.pkl')
#model_uni_anomaly = joblib.load('univariate_anomaly.pkl')
simi=joblib.load('model/movieRecommendation.pkl')
indices=pd.read_csv("data/indices_recommendation.csv")["title"]
indices = indices.apply(lambda x: str(x).lower())
indices =pd.DataFrame(indices)

movies_df=pd.read_csv("data/movies_df.csv")

def allowed_file(filename):
    return '.' in filename and            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__, template_folder='html', static_folder="static")
Markdown(app)
#@app.route('/', methods=['GET', 'POST'])
#def login():
#    error = None
#   if request.method == 'POST':
#        print("Username********************", request.form['uname'])
#        print("password********************", request.form['psw'])
#        if request.form['uname'] != 'admin' or request.form['psw'] != 'admin':
#            error = 'Invalid Credentials. Please try again.'
#        else:
#            return redirect(url_for('dashboard'))
#    return render_template('loginfirst.html', error=error)

@app.route("/", methods=['GET', 'POST'])
def logs():
    error = None
    if request.method == 'POST':
        print("Username********************", request.form['username'])
        print("password********************", request.form['password'])
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('dashboard'))
    return flask.render_template('login.html', error=error)

@app.route("/dashboard")
def dashboard():
    return flask.render_template('dashboard.html')

@app.route("/demo/cancerPrediction")
def salesIventory():
    return flask.render_template('cancerprediction.html')


@app.route('/detect', methods=['POST'])
def make_detection():
    print('in make_detection')
    if request.method=='POST':
        print('in request.method')
        # get uploaded image file if it exists
        file = request.files['data']
        if not file: return render_template('cancerprediction.html', wrnMsg_label="No file")
        
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
#             flash('No selected file')
            return render_template('cancerprediction.html', wrnMsg_label="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('execution',execution_path)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

#             saving uploaded file
            upload = execution_path + '/inputed'
            print(upload)
            uploadPath = os.path.join(upload, filename)
            file.save(uploadPath)
            print('file uploaded!')
            
            uploadPath_1 = uploadPath
            
            print('uploadPath_1',uploadPath_1)
            
            print(uploadPath.split('.')[-1])

            if uploadPath.split('.')[-1] == 'csv':
                print('in cvs')
                df = pd.read_csv(uploadPath_1,header=None)
            else:
                df = pd.read_excel(uploadPath_1,header=None)
            print(df)

            print('detecting...')


            filename = 'model/test_finalized_model_10.sav'
            # load the model from disk
            loaded_model = pickle.load(open(filename, 'rb'))
            print('model loaded')

            predicted = loaded_model.predict(df)
            print(predicted)

            print(type(predicted))


            pred_lst = predicted.tolist() 
            print(pred_lst)

            m_count = pred_lst.count(1)
            b_count = pred_lst.count(0)
            total = len(pred_lst)
            print('condition')

            if ((m_count*100)/total) > 50:
                status =  'The patient needs to re-test.'
            elif ((b_count*100)/total) > 50:
                status =  'The patient does not need to re-test.'
            else:
                status = ''


            returnStr_Status = status

#             status = ''
            m_percent = round( ((m_count*100)/total), 2)
            b_percent = round( ((b_count*100)/total), 2)
        
            m_lbl  =  'Malignant' + ' (' + str(m_percent) + '%)'
            b_lbl  =  'Benign' + ' (' + str(b_percent) + '%)'
            labels = [
                m_lbl, b_lbl
            ]

            values = [
                m_percent, b_percent
            ]
            
            bar_labels=labels
            bar_values=values

            returnStr = 'The probalility of the paitent being Malignant is: ' + str(m_percent) + '% and Benign is: ' + str(b_percent) + ' %.'

            return render_template('cancerprediction.html', label=returnStr, label_Status=returnStr_Status, max=100, labels=bar_labels, values=bar_values)#, user_image = newName)
        else:
            return render_template('cancerprediction.html', wrnMsg_label="Only 'csv', 'xls' and 'xlsx' files are allowed.")

        
class Helper:
    """Helper functions for model training, eda and model prediction"""

    def __init__(self, file_path):
        self.file_name = ""
        self.train_df = None
        self.file_path = file_path

    def read_csv_files(self, radio_button_response):
        try:
            self.train_df = pd.read_csv(self.file_path)
            self.train_df = self.train_df.dropna()
            self.train_df.columns = self.train_df.columns.str.lower()
            # self.train_df = self.train_df.rename(
            #     columns={"weekly_sales": "sales", "dept": "item"})
            # self.train_df = self.train_df[['date', 'sales', 'item', 'store']]
            # print(self.train_df.head())
            # self.train_df = self.train_df[self.train_df['store'] == self.item_no][
            #     self.train_df['item'] == self.store_no]
            # self.train_df["year"] = pd.to_datetime(self.train_df["date"]).dt.year
            # self.train_df["month"] = pd.to_datetime(self.train_df["date"]).dt.month
            # self.train_df["day"] = pd.to_datetime(self.train_df["date"]).dt.day
            # self.train_df['dayOfWeek'] = pd.to_datetime(self.train_df["date"]).dt.dayofweek
            if 'sale' in radio_button_response:
                # self.train_df.date = pd.to_datetime(self.train_df.date, format='%Y-%m')
                self.train_df["sales"] = np.log1p(self.train_df["sales"])
                self.train_df = self.train_df[["date", "sales"]]

                self.train_df = self.train_df.rename(columns={"date": "ds", "sales": "y"})
            if 'inventory' in radio_button_response:
                # self.train_df.date = pd.to_datetime(self.train_df.date, format='%Y-%m')
                self.train_df["inventory"] = np.log1p(self.train_df["inventory"])
                self.train_df = self.train_df[["date", "inventory"]]
                self.train_df = self.train_df.rename(columns={"date": "ds", "inventory": "y"})
            print(self.train_df.head())

            return True
        except Exception as e:
            print(f"Error while reading csv. Error: {e}")
            return False

    def train_predict(self, radio_button_response, future_day):
        # print(self.train_df.tail(), len(self.train_df))
        # model = Prophet(weekly_seasonality=True)
        model = Prophet()
        model.fit(self.train_df)
        future = model.make_future_dataframe(periods=int(future_day))
        # print(future.head(), len(future))
        forecast_ = model.predict(future)
        print(forecast_.head())
        forecast_["yhat"] = np.expm1(forecast_["yhat"])
        forecast = forecast_.tail(int(future_day))
        # print("r2 score: ", r2_score(self.train_df.y, forecast_['yhat'][:int(future_day)]))
        # print("rmse: ", mean_squared_error(self.train_df.y, forecast_['yhat'][:int(future_day)], squared=False))
        # print("mean_absolute_error", mean_absolute_error(self.train_df.y, forecast_['yhat'][:int(future_day)]))
        print(forecast.head(), len(forecast_))
        future_predicted_value = forecast[['ds', 'yhat']]
        future_predicted_value.reset_index(drop=True, inplace=True)
        # print("len", len(forecast_['yhat'][:int(future_day)].tolist()),
        #       len(forecast_['yhat'][int(future_day):].tolist()))
        act_nan_lst = ('NaN ' * len(np.expm1(self.train_df['y']).round(2).tolist())).split()
        fore_nan_lst = ('NaN ' * len(forecast_['yhat'].tail(int(future_day)).round(2).tolist())).split()
        final_actual = np.expm1(self.train_df['y']).round(2).tolist() + fore_nan_lst
        final_forecast = act_nan_lst + np.expm1(self.train_df['y']).round(2).tolist()

        result_data = {'date': forecast_['ds'].tolist(),
                       'actual': final_actual,
                       'forecasted': final_forecast}
        return result_data  # flask.render_template('index.html',result = jsonify(result_data))
        """
        if 'sale' in radio_button_re sponse:
            return {'result_data': {'date': forecast_['ds'].tolist(),
                                    'actual': final_actual,
                                    'sale_forecasted': final_forecast}}
        if 'inventory' in radio_button_response:
            return {'result_data': {'date': forecast_['ds'].tolist(),
                                    'actual': final_actual,
                                    'inventory_forecasted': final_forecast}}
        """
        
@app.route("/demo/SalesInventoryforecast")
def forecast_index():
    return flask.render_template('salesinventory.html')


@app.route('/forecast', methods=['POST'])
def forecast():
    if request.method == 'POST':
        future_day = int(request.form.get("f_days"))
        if future_day is None or future_day == '':
            return render_template('index_forecast.html', wrnMsg_label="No value in 'No of days'")
        radio_button_response = request.form.getlist('prediction')
        file = request.files['data']
        if not file:
            return render_template('salesinventory.html', wrnMsg_label="No file")
        if file.filename == '':
            return render_template('salesinventory.html', wrnMsg_label="No selected file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(FILE_UPLOAD_PATH, filename)
            file.save(upload_path)
            # new_name_path = os.path.join(FILE_UPLOAD_PATH, 'test.csv')
            # os.rename(upload_path, new_name_path)
            class_obj = Helper(upload_path)
            check = class_obj.read_csv_files(radio_button_response)
            print(check)
            if not check:
                return render_template('salesinventory.html', wrnMsg_label="Mis-match in selected type and given file")
            result_data = class_obj.train_predict(radio_button_response, future_day)
            return flask.render_template('salesinventory.html', date_1=result_data['date'], actual=result_data['actual'],
                                         forecasted=result_data['forecasted'])

###### Movie Recommendation #######
@app.route("/demo/MovieRecommendation")
def movierecommendationPage():
    return flask.render_template('movieRecommendation.html')


@app.route('/recommendedMovies', methods=['POST'])

def get_recommend():
    """
    in this function,
        we take the cosine score of given movie
        sort them based on cosine score (movie_id, cosine_score)
        take the next 10 values because the first entry is itself
        get those movie indices
        map those indices to titles
        return title list
    """
    error = None
    if request.method == 'POST':
        print("Username********************", request.form['MovieName'])
        form_data = request.form['MovieName']
        if not form_data:
            return render_template('movieRecommendation.html', wrnMsg_label="No data")

        MovieName =str(form_data)
        
        serach_data = indices.loc[indices['title'] == MovieName]
        if  len(serach_data):
            idx = indices.loc[indices['title']==str(MovieName)].index[0]
            print(idx)
            sim_scores = list(enumerate(simi[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            # (a, b) where a is id of movie, b is sim_score
            
            movies_indices = [ind[0] for ind in sim_scores]
            movies = movies_df["title"].iloc[movies_indices]
            recc_movies=[]
            for i in movies_indices:
                names=indices['title'][i]
                recc_movies.append(names)
            print(recc_movies)
            return flask.render_template('movieRecommendation.html', results=recc_movies, movieName=request.form['MovieName'])

        else:
            print("not found")
            return flask.render_template('movieRecommendation.html', 
                 wrnMsg_label="return-entered movies is not in our database, pls enter different movie name")
                                         


#NER Prediction
###### Movie Recommendation #######
@app.route("/demo/PredNer")
def nerPage():
    return flask.render_template('ner.html')


@app.route('/nerPred', methods=['POST'])

def ner_():

    error = None
    if request.method == 'POST':
        print("Username********************", request.form['NerName'])
        form_data = request.form['NerName']
        if not form_data:
            return render_template('ner.html', wrnMsg_label="No data")
        
    
        text1= nlp(str(request.form['NerName']))
        print("000000:", text1)
        ner_data =[]
        for word in text1.ents:
            #print(word.text)
            ner_data.append(word)
        print("11111111:", ner_data)
       
        #return flask.render_template('ner.html', results=ner_data)
        #Output1 = displacy.render(text1,style="ent")
        Output1 = displacy.render(text1, style="ent")
    
        #print("2222:", Output1)
        return flask.render_template('ner.html', results=Output1)
  


#gender and age detction------------

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes
#gender and age detction

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

faceProto="model/opencv_face_detector.pbtxt"
faceModel="model/opencv_face_detector_uint8.pb"
ageProto="model/age_deploy.prototxt"
ageModel="model/age_net.caffemodel"
genderProto="model/gender_deploy.prototxt"
genderModel="model/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


padding=20

@app.route("/demo/DetectGenderAge")
def genderAgePage():
    return flask.render_template('genderAge.html')

@app.route('/GenderAgeDetection', methods=['POST'])
def GenderAgeDetect_():
    if request.method == 'POST':
        file = request.files['image']
        print("11111111111:", file)
         # get uploaded image file if it exists
        if not file:
            return render_template('genderAge.html', wrnMsg_label="No file")
        
    
        # Save file
        #filename = 'static/' + file.filename
        #file.save(filename)
    
        # Read image
        frame = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        print("222222222222:", file)
        resultImg,faceBoxes=highlightFace(faceNet,frame)
        print("3333333333333:", faceBoxes)
        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                       min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                       :min(faceBox[2]+padding, frame.shape[1]-1)]
        
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')
        
            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')
        
            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
           
        
            
            # In memory
            image_content = cv2.imencode('.jpg', resultImg)[1].tostring()
            encoded_image = base64.encodestring(image_content)
            to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    
        return render_template('genderAge.html', image_to_show=to_send)
    else:
        return render_template('genderAge.html', wrnMsg_label="Error: please upload input image.. only allowed:[png or jpg....]")





#sentiment analyser

###### sentiment analyser #######

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()


def predict_sentiment(text):
    score = ((sid.polarity_scores(str(text))))['compound']
    return score


@app.route("/demo/AnalyseSentiment")
def sentimentAnalysePage():
    return flask.render_template('sentimentAnalysis.html')


@app.route('/SentimentAnalyse', methods=['POST'])

def sentimentAnalyse_():

    error = None
    if request.method == 'POST':
        print('in request.method')
        print("Username********************", request.form['SentimentAnalyseName'])

        # get uploaded image file if it exists
        form_data = request.form['SentimentAnalyseName']
        if not form_data: 
            return render_template('sentimentAnalysis.html', wrnMsg_label="No data")
     
    
        test_sentence3= str(request.form['SentimentAnalyseName'])
        print("000000:", test_sentence3)
        score =predict_sentiment(test_sentence3)
        print("senti_pred:", score)
        if(score > 0):
            label = 'positive'
            print(label)

            return flask.render_template('sentimentAnalysis.html', results=label)

        elif(score == 0):
            label = 'neutral'
            print(label)

            return flask.render_template('sentimentAnalysis.html', results=label)

        else:
            label = 'negative'
        
            print(label)
        
            return flask.render_template('sentimentAnalysis.html', results=label)
    


#Customer Segmentation

def predict_custSegment(df):
    print("insideeeeeeeeeeeeeee")
    df_fix = df[df['CustomerID'].notna()]
    df_fix["InvoiceDate"] = pd.to_datetime(df_fix["InvoiceDate"],errors='coerce')
    df_fix["TotalSum"] = df_fix["Quantity"] * df_fix["UnitPrice"]
    snapshot_date = max(df_fix.InvoiceDate) + datetime.timedelta(days=1)
    customers = df_fix.groupby(['CustomerID']).agg({'InvoiceDate': lambda x: (snapshot_date - x.max()).days,'InvoiceNo': 'count','TotalSum': 'sum'})
    customers.rename(columns = {'InvoiceDate': 'Recency',
                                'InvoiceNo': 'Frequency',
                                'TotalSum': 'MonetaryValue'}, inplace=True)
    pd.Series(np.cbrt(customers['MonetaryValue'])).values
    customers_fix = pd.DataFrame()
    customers_fix["Recency"] = stats.boxcox(customers['Recency'])[0]
    customers_fix["Frequency"] = stats.boxcox(customers['Frequency'])[0]
    customers_fix["MonetaryValue"] = pd.Series(np.cbrt(customers['MonetaryValue'])).values
    
    scaler = StandardScaler()
    scaler.fit(customers_fix)
    customers_normalized = scaler.transform(customers_fix)
    
    return customers_normalized

@app.route("/demo/segmentionCustomer")
def segCustPage():
    return flask.render_template('customerSegmentation.html')


@app.route('/CustomerSegmentation', methods=['POST'])
def CustSegment_():
    print('in make_detection')
    if request.method=='POST':
        print('in request.method')
        # get uploaded image file if it exists
        file = request.files['data']
        if not file: return render_template('customerSegmentation.html', wrnMsg_label="No file")
        
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
#             flash('No selected file')
            return render_template('customerSegmentation.html', wrnMsg_label="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('execution',execution_path)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

#             saving uploaded file
            upload = execution_path + '/inputed'
            print(upload)
            uploadPath = os.path.join(upload, filename)
            file.save(uploadPath)
            print('file uploaded!')
            
            uploadPath_1 = uploadPath
            
            print('uploadPath_1',uploadPath_1)
            
            print("11111111", uploadPath.split('.')[-1])

            if uploadPath.split('.')[-1] == 'csv':
                print('in cvs')
                df = pd.read_csv(uploadPath_1,encoding = 'unicode_escape')
            else:
                df = pd.read_excel(uploadPath_1,encoding = 'unicode_escape')
            print(df)

            customers_normalized =predict_custSegment(df)
            print('detecting...')


           

            predicted_ = loaded_model__.predict(customers_normalized)
            print(predicted_)

            print(type(predicted_))


            final = predicted_.tolist() 
            print(final)
            
            name = {0:"Active_client", 1:"regular_client",2:"passive_client"}
            output = [*map(name.get, final)]
            print(output)

            
            return render_template('customerSegmentation.html', results=output)
        else:
            return render_template('customerSegmentation.html', wrnMsg_label="Error: Only 'csv', 'xls' and 'xlsx' files are allowed.")



#ocr nlp


@app.route("/demo/NlpOcr")
def NlpOcrPage():
    return flask.render_template('ocrNlp.html')

@app.route('/OcrNlp', methods=['POST'])
def OcrNlp_():
    print('in ocr nlp')
    if request.method=='POST':
        print('in request.method')
        # get uploaded image file if it exists
        file = request.files['image']
        if not file: return render_template('ocrNlp.html', wrnMsg_label="No file")
        
        print("file:", file)
     
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('execution',execution_path)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

#             saving uploaded file
            upload = execution_path + '/inputed'
            print(upload)
            uploadPath = os.path.join(upload, filename)
            file.save(uploadPath)
            print('file uploaded!')
            
            IMAGE_PATH1 = uploadPath
            
            print('IMAGE_PATH1',IMAGE_PATH1)
            
            print("11111111", uploadPath.split('.')[-1])

            ocr_reader = easyocr.Reader(['en'], gpu=False)
            
            ocr_result_ = ocr_reader.readtext(IMAGE_PATH1, detail =0)
            
            joined_string = ' '.join(ocr_result_)
            
            print(joined_string)
            #show uploaded image
            frame = cv2.imdecode(np.fromfile(IMAGE_PATH1, dtype=np.uint8), -1)
            #frame = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            image_content = cv2.imencode('.jpg', frame)[1].tostring()
            encoded_image = base64.encodestring(image_content)
            to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    
      
            #return render_template('ocrNlp.html', results=joined_string, image_to_show=encoded_img_data.decode('utf-8'))
            return render_template('ocrNlp.html', results=joined_string, image_to_show=to_send)

   
            #return render_template('ocrNlp.html', results=joined_string, image_to_show=to_send)
    
        else:
            return render_template('ocrNlp.html', wrnMsg_label="Error: Only 'PNG', and 'JPG' files are allowed.")
        


#car price


def carPrice_predict(df):
    inputs=df.drop(['Car_Name','Owner','Seller_Type'],axis='columns') 
    target=df.Selling_Price
    Numerics=LabelEncoder()
    inputs['Fuel_Type_n']=Numerics.fit_transform(inputs['Fuel_Type'])
    inputs['Transmission_n']=Numerics.fit_transform(inputs['Transmission'])
    inputs_n=inputs.drop(['Fuel_Type','Transmission','Selling_Price'],axis='columns')
    return inputs_n

@app.route("/demo/PriceCar")
def PriceCarPage():
    return flask.render_template('carPrice.html')

@app.route('/CarPrice', methods=['POST'])
def CarPrice_():
    print('in make_detection')
    if request.method=='POST':
        print('in request.method')
        # get uploaded image file if it exists
        file = request.files['data']
        if not file: return render_template('carPrice.html', wrnMsg_label="No file")
        
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
#             flash('No selected file')
            return render_template('carPrice.html', wrnMsg_label="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('execution',execution_path)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

#             saving uploaded file
            upload = execution_path + '/inputed'
            print(upload)
            uploadPath = os.path.join(upload, filename)
            file.save(uploadPath)
            print('file uploaded!')
            
            uploadPath_1 = uploadPath
            
            print('uploadPath_1',uploadPath_1)
            
            print("11111111", uploadPath.split('.')[-1])

            if uploadPath.split('.')[-1] == 'csv':
                print('in cvs')
                df = pd.read_csv(uploadPath_1,encoding = 'unicode_escape')
            else:
                df = pd.read_excel(uploadPath_1,encoding = 'unicode_escape')
            print(df)

            inputs_n =carPrice_predict(df)
            print('detecting...')
            
            
            y_pred = carPrice_model.predict(inputs_n) 
            final = y_pred.tolist()
            #print(final)
            final_2 = []
            for i in final:
                answer = (round(i, 2))
                final_2.append(answer)


            print(final_2)

            
            return render_template('carPrice.html', results=final_2)
        else:
            return render_template('carPrice.html', wrnMsg_label="Error: Only 'csv', 'xls' and 'xlsx' files are allowed.")


#anomlay detection


def Anomaly_Detection_Function(File_name):
    df = pd.read_csv(File_name, header=None, names =['col'])
    print("dfffffffff", df)
    model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
    model.fit(df)
    df['scores']=model.decision_function(df[['col']])
    df['anomaly']=model.predict(df[['col']])
    #anomaly=df.loc[df['anomaly']==-1] 
    #anomaly1=df.loc[df['anomaly']==1] 
    
    #print(anomaly)
    return df

from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
@app.route("/demo/DetectAnomaly")
def DetectAnomalyPage():
    return flask.render_template('anomaly.html')



@app.route('/AnomalyDetect', methods=['POST', 'GET'])
def AnomlayDetect_():
    print('in make_detection')
    
    if request.method=='POST':
        print('in request.method')
        # get uploaded image file if it exists
        file = request.files['data']
        print(file)
        if not file: 
            print('if no file condition')
            return render_template('anomaly.html', wrnMsg_label="No file")
        
        
        
        anomaly_=Anomaly_Detection_Function(file)
        print("2222222222dfffffff:hwshj")
        df_ =anomaly_.copy()
        df_.drop(['scores'], axis=1, inplace=True)
        df_['col2'] =range(len(df_['col']))
        df_['anomaly'] = df_['anomaly'].map(lambda x: 1 if x==-1 else 0)
        
        
        fig, ax = plt.subplots(figsize=(10,6))
        
        a = df_.loc[df_['anomaly'] == 1, ['col', 'col2']] #anomaly
        
        ax.plot(df_['col2'], df_['col'], color='blue', label='Normal')
        ax.scatter(a['col2'],a['col'], color='red', label='Anomaly')
        plt.xlabel('Serial Number')
        plt.ylabel('univariate')
        plt.legend()
        #plt.show();
        
        fig.savefig("hist_.jpg")
            

        # Full Script.
        im = Image.open("hist_.jpg")
        data = io.BytesIO()
        im.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())
        
        return render_template('anomaly.html', image_to_show=encoded_img_data.decode('utf-8'))
    else:
        return render_template('anomaly.html', wrnMsg_label="Error: please upload input image.. only allowed:[png or jpg....]")

    

#tesseract

@app.route("/demo/OcrPytesseract")
def pytesseractPage():
    return flask.render_template('pytesseract.html')



@app.route('/pytesseractOcr', methods=['POST', 'GET'])
def pytesseract_():
    print('in make_detection')
    
    if request.method=='POST':
        print('in request.method')
        # get uploaded image file if it exists
        file = request.files['data']
        print(file)
        if not file: 
            print('if no file condition')
            return render_template('pytesseract.html', wrnMsg_label="No file")
        
        frame = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        print("222222222222:", file)
        
        ocr_data = pytesseract.image_to_string(frame, lang='eng',config='--psm 6')
        # In memory
        image_content = cv2.imencode('.jpg', frame)[1].tostring()
        encoded_image = base64.encodestring(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')

        
        return render_template('pytesseract.html', results=ocr_data, image_to_show =to_send)
    else:
        return render_template('pytesseract.html', wrnMsg_label="Error: Only 'PNG', and 'JPG' files are allowed.")
    

#        
@app.route("/demo/EasyOcrPytesseract")
def EasyOcrPytesseractPage():
    return flask.render_template('pytesseract_easyOcr.html')


@app.route('/PytesseractEasyOcr', methods=['POST'])
def PytesseractEasyOcr_():
    if request.method == 'POST':
        radio_button_response = request.form.getlist('prediction')
        print("radio_button_response:", radio_button_response)
        file = request.files['data']
        if not file:
            return render_template('pytesseract_easyOcr.html', wrnMsg_label="No file")
        
        if 'tesseract' in radio_button_response:
            print("inside tesseract")
            frame = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            print("222222222222:", file)
            
            ocr_data_ = pytesseract.image_to_string(frame, lang='eng',config='--psm 6')
    
            print("ocr_data_:", ocr_data_)
            
 
            # In memory
            image_content = cv2.imencode('.jpg', frame)[1].tostring()
            encoded_image = base64.encodestring(image_content)
            to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')

        
            return render_template('pytesseract_easyOcr.html', results=ocr_data_, image_to_show =to_send)
  
        
        if 'easyocr' in radio_button_response:
            print("inside easyocr")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                print('execution',execution_path)
    #             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    #             saving uploaded file
                upload = execution_path + '/inputed'
                print(upload)
                uploadPath = os.path.join(upload, filename)
                file.save(uploadPath)
                print('file uploaded!')
                
                IMAGE_PATH1 = uploadPath
                
                print('IMAGE_PATH1',IMAGE_PATH1)
                
                print("11111111", uploadPath.split('.')[-1])
    
                ocr_reader = easyocr.Reader(['en'], gpu=False)
                
                ocr_result_ = ocr_reader.readtext(IMAGE_PATH1, detail =0)
                
                joined_string = ' '.join(ocr_result_)


                print(joined_string)
                
                #show uploaded image
                frame = cv2.imdecode(np.fromfile(IMAGE_PATH1, dtype=np.uint8), -1)
                #frame = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    
                image_content = cv2.imencode('.jpg', frame)[1].tostring()
                encoded_image = base64.encodestring(image_content)
                to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')

                return render_template('pytesseract_easyOcr.html', results=joined_string, image_to_show =to_send)

#

#House price
import joblib

hmodel = joblib.load('model/House_prices_model.pkl')

@app.route("/demo/PriceHouse")
def PriceHousePage():
    return flask.render_template('housePrice.html')

@app.route('/HousePrice', methods=['POST'])
def HousePrice_():
    print('in make_detection')
    if request.method=='POST':
        print('in request.method')
        # get uploaded image file if it exists
        file = request.files['data']
        if not file: return render_template('housePrice.html', wrnMsg_label="No file")
        
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
#             flash('No selected file')
            return render_template('housePrice.html', wrnMsg_label="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('execution',execution_path)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

#             saving uploaded file
            upload = execution_path + '/inputed'
            print(upload)
            uploadPath = os.path.join(upload, filename)
            file.save(uploadPath)
            print('file uploaded!')
            
            uploadPath_1 = uploadPath
            
            print('uploadPath_1',uploadPath_1)
            
            print("11111111", uploadPath.split('.')[-1])

            if uploadPath.split('.')[-1] == 'csv':
                print('in cvs')
                df = pd.read_csv(uploadPath_1,encoding = 'unicode_escape')
            else:
                df = pd.read_excel(uploadPath_1,encoding = 'unicode_escape')
            print(df)

                        
            df_ =df.copy()
            
            dummies = pd.get_dummies(df_.location)
            df__ = pd.concat([df_,dummies.drop('other',axis='columns')],axis='columns')
            df___= df__.drop('location',axis='columns')
            
            """# Build a model now"""
            
            X_ = df___.drop(['price'],axis='columns')
            y_ = df___.price
            
            print('detecting...')
            
            
            y_pred = hmodel.predict(X_) 
            final = y_pred.tolist()
            #print(final)
            final_2 = []
            for i in final:
                answer = (round(i, 2))
                final_2.append(answer)


            print(final_2)

            
            return render_template('housePrice.html', results=final_2)
        else:
            return render_template('housePrice.html', wrnMsg_label="Error: Only 'csv', 'xls' and 'xlsx' files are allowed.")

#energ consumptions

import joblib
import pandas as pd
import time
import numpy as np
import datetime
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
logging.debug('Loading libraries for feature selection and prediction')
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
import joblib

ecmodel = joblib.load('model/Energy_Consumption.pkl')

def get_ec_data(data_ec):
    df_loaded =data_ec.copy()
    df = df_loaded

    df.head()

    df['NSM'] = df.date.apply(lambda x: x.hour*3600 + x.minute*60 +x.second)
    df['day_of_week'] = df.date.apply(lambda x: x.dayofweek)
    df['week_status'] = df.day_of_week.apply(lambda x: 0 if (x == 5 or x == 6) else 1)

    shape_bool = df.date.nunique() == df.shape[0]


    all_columns = df.columns.tolist()

    df_describe = df.describe().T

    df_describe['Interquartile Range'] = 1.5*(df_describe['75%'] - df_describe['25%'])
    df_describe['Major Outlier'] = (df_describe['75%'] + df_describe['Interquartile Range'])
    df_describe['Minor Outlier'] = (df_describe['25%'] - df_describe['Interquartile Range'])


    def remove_outlier(df, variable):
        major_o = df_describe.loc[variable,'Major Outlier']
        minor_o = df_describe.loc[variable,'Minor Outlier']
        df = df.drop(df[(df[variable]>major_o) | (df[variable]<minor_o)].index)
        return df

    outlier_column_list = [x for x in all_columns 
                           if x not in ('date', 'Appliances', 'lights')]


    for column_name in outlier_column_list:
        df = remove_outlier(df, column_name)

    dropped = ((df_loaded.shape[0] - df.shape[0])/df_loaded.shape[0])*100


    week_status = pd.get_dummies(df['week_status'], prefix = 'week_status')
    day_of_week = pd.get_dummies(df['day_of_week'], prefix = 'day_of_week')

    df = pd.concat((df,week_status),axis=1)
    df = pd.concat((df,day_of_week),axis=1)

    df = df.drop(['week_status','day_of_week'],axis=1)

    df = df.rename(columns={'week_status_0': 'Weekend', 'week_status_1': 'Weekday',
                       'day_of_week_0': 'Monday', 'day_of_week_1': 'Tuesday', 'day_of_week_2': 'Wednesday',
                      'day_of_week_3': 'Thursday', 'day_of_week_4': 'Friday', 'day_of_week_5': 'Saturday',
                      'day_of_week_6': 'Sunday'})

    df['Appliances'] = df['Appliances'] + df['lights']
    df = df.drop(['lights'],axis=1)
    df = df.drop(['date'],axis=1)

    X = df.drop(['Appliances'],axis=1)
    y = df['Appliances']
    X.head()

    # Use the loaded model to make predictions 
    y_pred = ecmodel.predict(df) 
    print(y_pred)
    return y_pred




@app.route("/demo/ConsumptionEnergy")
def ConsumptionEnergyPage():
    return flask.render_template('energyConsumptions.html')

@app.route('/EnergyConsumption', methods=['POST'])
def EnergyConsumption_():
    print('in make_detection')
    if request.method=='POST':
        print('in request.method')
        # get uploaded image file if it exists
        file = request.files['data']
        if not file: return render_template('energyConsumptions.html', wrnMsg_label="No file")
        
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
#             flash('No selected file')
            return render_template('energyConsumptions.html', wrnMsg_label="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('execution',execution_path)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

#             saving uploaded file
            upload = execution_path + '/inputed'
            print(upload)
            uploadPath = os.path.join(upload, filename)
            file.save(uploadPath)
            print('file uploaded!')
            
            uploadPath_1 = uploadPath
            
            print('uploadPath_1',uploadPath_1)
            
            print("11111111", uploadPath.split('.')[-1])

            if uploadPath.split('.')[-1] == 'csv':
                print('in cvs')
                df = pd.read_csv(uploadPath_1,encoding = 'unicode_escape', parse_dates=['date'])
            else:
                df = pd.read_excel(uploadPath_1,encoding = 'unicode_escape', parse_dates=['date'])
            print(df)

                        
            df_result =get_ec_data(df)
            
            
            final = df_result.tolist()
            #print(final)
            final_2 = []
            for i in final:
                answer = (round(i, 2))
                final_2.append(answer)


            print(final_2)

            
            return render_template('energyConsumptions.html', results=final_2)
        else:
            return render_template('energyConsumptions.html', wrnMsg_label="Error: Only 'csv', 'xls' and 'xlsx' files are allowed.")


if __name__ == '__main__':
    # load ml model
    try:
        import os
        print('in main')
        execution_path = os.getcwd()
       
        print('out main')
        # start api
        #app.run(host='192.168.6.19', port=5000)#, debug=True)#host='0.0.0.0', port=8000, debug=True
        app.run(host='0.0.0.0', port=5000, debug=False)
        #app.run(host='192.168.4.194', port=5015)
    except:
        traceback.print_exc()

