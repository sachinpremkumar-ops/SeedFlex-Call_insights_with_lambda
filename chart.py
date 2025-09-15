import boto3
from botocore.exceptions import ClientError
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from openai import OpenAI
from pydantic import BaseModel
import io
import os
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import spacy
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords

sia=SentimentIntensityAnalyzer()

def get_secret():
    secret_name = "experiment"
    region_name = "ap-southeast-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = get_secret_value_response['SecretString']
    return json.loads(secret)

class User(BaseModel):
    customer:str
    abstract_summary: str
    key_points: str
    action_items: str
    sentiment: str
    size:int
    prompt:str
    word_scores:str
    # abstract_summary1:str
    # abstract_summary2:str
    # prompt1:str
    # prompt2:str


def connect_to_rds():
    secret = get_secret()
     # Extract the necessary details from the secret
    db_host = secret['host']
    db_user = secret['username']
    db_password = secret['password']
    db_name = secret['database']
    db_port = secret['port']

    # Connect to the RDS instance
    try:
        connection = psycopg2.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            dbname=db_name,
            port=int(db_port),
            cursor_factory=RealDictCursor
        )
        print("Connection successful!")
        return connection
    except Exception as e:
        print(f"Failed to connect to the database: {e}")
        return None
    

s3_client= boto3.client('s3')


def read(file):
    file_name=file
    with open(file_name,"rb") as f:
        data=f.read()
        return data

def server_delete(file_path):
    try:
        os.remove(file_path)
        print(f'{file_path} has been deleted.')
    except FileNotFoundError:
        print(f'{file_path} not found.')
    except Exception as e:
        print(f'Error: {e}')

def s3_upload(file):
        body=read(file)
        s3_client.put_object(
            ACL='private',
            Bucket='experiment2407',
            Body=body,
            Key=f'processed/{file}'
        )

def wordcloud(data):
    plt.style.use('ggplot')



    nlp = spacy.load('en_core_web_sm')
    nltk.download('stopwords')

    df=pd.DataFrame(data)

    abstract_summary=''.join(df['abstract_summary'])
 
    doc = nlp(abstract_summary)
    excluded_pos = {'PUNCT', 'DET', 'ADP', 'AUX', 'CCONJ', 'SCONJ'} 
    dont_include={'conversation','agent','seller','back','seems','back','loan','commerce','concerns','information','provide','provided','therefore','please','however','product','sales','introduce','customer','merchant','also','ended','sent','number','email','called','contain','text','person','summary','instead'}

    stop_words=set(stopwords.words('english'))
    filtered_words = [
    token.text.lower() for token in doc
    if token.text.lower() not in stop_words
    and token.pos_ not in excluded_pos
    and not token.ent_type_
    and token.text.lower() not in dont_include
    ]
    filtered_words = [word for word in filtered_words if len(word) > 3]
    
    words_counts=Counter(filtered_words)

    wordcloud_pos = WordCloud(collocations= True,
                          background_color="white",
                          colormap="Set1").generate_from_frequencies(words_counts)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud_pos, interpolation='bilinear')
    plt.axis('off')
    
    plt.savefig(f'{from_no}-{to_no}WordCloud.png')
    s3_upload(f'{from_no}-{to_no}WordCloud.png')
    server_delete(f'{from_no}-{to_no}WordCloud.png')
    print("Uploaded WordCloud Successfully")


def sentiment_table(data):
    df=pd.DataFrame(data)

    df_comb=''.join(df['abstract_summary'])
    print(df_comb)
    scores=sia.polarity_scores(df_comb)
    print(scores)

    result=pd.DataFrame([scores])
    print(result)


    result={
        'NEGATIVE':[scores['neg']], 
        'NEUTRAL':[scores['neu']],
        'POSITIVE':[scores['pos']],
        
    }
    total=scores['pos']-scores['neg']
    print(total)
    result=pd.DataFrame(result)
    fig,ax=plt.subplots(figsize=(10,8))
    colors=['red','blue','green']

    bars=result.plot(kind='bar',ax=ax,color=colors)

    
    plt.xlabel('Scores')
    plt.ylabel('Range')
    plt.title=('Sentiment Analysis')
    for bar in  ax.containers:
        ax.bar_label(bar,fmt='%2f',label_type='edge')
    plt.legend()
    plt.fill()
    plt.savefig(f'{from_no}-{to_no}bar_chart.png')
    s3_upload(f'{from_no}-{to_no}bar_chart.png')
    server_delete(f'{from_no}-{to_no}bar_chart.png')
    print("Uploaded Bar Chart Successfully")

from_no=1
to_no=100
def get_table(connection,from_no,to_no):
    query = """SELECT abstract_summary FROM call_insight WHERE no BETWEEN %s AND %s"""
    sentiment_data=[]

    try:
        with connection.cursor() as cursor:
            sentiment_data1={}
            cursor.execute(query,(from_no,to_no))
            rows = cursor.fetchall()

            for row in rows:
               x=row['abstract_summary']
               
               sentiment_data.append(x)
            sentiment_data1['abstract_summary']=sentiment_data
               
               
    except Exception as e:
        print(f"Error fetching data: {e}")

    return sentiment_data1





conn=connect_to_rds()
if conn:
    sentiment_data=get_table(conn,from_no,to_no)
    print(sentiment_data)
    sentiment_table(sentiment_data)
    wordcloud(sentiment_data)


    conn.close()

