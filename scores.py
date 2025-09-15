import boto3
from botocore.exceptions import ClientError
import psycopg2
from psycopg2.extras import RealDictCursor,RealDictRow
import json
from openai import OpenAI
from pydantic import BaseModel
import io
import os
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

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

    pos_score:float
    neg_score:float
    neu_score:float
    comp_score:float
    sentiment_score:float
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



# def create_table(connection,table_name):
#     with connection.cursor() as cursor:
#         try:
#             create_script = f"""
#             CREATE TABLE IF NOT EXISTS {table_name} (
#             No SERIAL NOT NULL PRIMARY KEY,
#             Customer VARCHAR NOT NULL,
#             Key_Points VARCHAR NOT NULL,
#             Action_Items VARCHAR NOT NULL,
#             Sentiment VARCHAR NOT NULL

#             );"""
#             cursor.execute(create_script)
#             connection.commit()
#             print("Table was created successfully!")
#         except Exception as e:
#             print(f"Error creating table: {e}")
#             connection.rollback()










def create_table4(connection,table_name):
    with connection.cursor() as cursor:
        try:
            create_script = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
            No SERIAL NOT NULL PRIMARY KEY,
            Customer VARCHAR NOT NULL,
            Positive_Score INTEGER NULL,
            Negative_Score INTEGER NULL,
            Neutral_Score INTEGER NULL,
            Created_at TIMESTAMP DEFAULT LOCALTIMESTAMP
            );"""
            cursor.execute(create_script)
            connection.commit()
            print("Table was created successfully!")
        except Exception as e:
            print(f"Error creating table: {e}")
            connection.rollback()





def create_data4(connection, post: User):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO SCORES(Customer,Positive_Score,Negative_Score,Neutral_Score,Compound_Score,Sentiment_score)
                VALUES (%s,%s,%s, %s,%s,%s) 
            """, (post.customer,post.pos_score,post.neg_score,post.neu_score,post.comp_score,post.sentiment_score))
            connection.commit()
            print("Data inserted successfully!")


            return cursor.fetchone()[0]
    except Exception as e:
        print(f"Error inserting data: {e}")
        connection.rollback()

def create_data5(connection, post: User):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO NEGATIVE_CUSTOMER(Customer,Positive_Score,Negative_Score,Neutral_Score,Compound_Score,Sentiment_score)
                VALUES (%s,%s,%s, %s,%s,%s) 
            """, (post.customer,post.pos_score,post.neg_score,post.neu_score,post.comp_score,post.sentiment_score))
            connection.commit()
            print("Data inserted successfully!")


            return cursor.fetchone()[0]
    except Exception as e:
        print(f"Error inserting data: {e}")
        connection.rollback()

def meeting_minutes(transcription,x):
    if isinstance(transcription, str):
        transcription = [{'abstract_summary': transcription}]
    
    
    print("Processed transcription data:", transcription)


    sentiment_score=sentiment_chart(transcription)
    if sentiment_score and isinstance(sentiment_score, list):
        sentiment_score = sentiment_score[0]  
    else:
        sentiment_score = {}
    return User(
        customer=x,
        pos_score=sentiment_score.get('pos_score',''),
        neg_score=sentiment_score.get('neg_score',''),
        neu_score=sentiment_score.get('neu_score',''),
        comp_score=sentiment_score.get('comp_score',''),
        sentiment_score=sentiment_score.get('total',''),
        
    ).dict()

abstract_summaries=[]

def print_data(connection,table_name):
    query = f"""SELECT * FROM {table_name}"""

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            for row in rows:
               print(row)
               

    except Exception as e:
        print(f"Error fetching data: {e}")

def print_data1(connection,table_name,from_id=275,to_id=1000):
    query = f"""SELECT * FROM {table_name} WHERE NO BETWEEN {from_id} AND {to_id}"""
    

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            for row in rows:
               print(row)
               abstract_summaries.append(row)

    except Exception as e:
        print(f"Error fetching data: {e}")



def sentiment_chart(data):

    if isinstance(data, RealDictRow):
        data = [dict(data)]

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("Data must be a list of dictionaries.")

    
    
    df=pd.DataFrame(data)
    df['score']=df['abstract_summary'].apply(lambda abstract_summary:sia.polarity_scores(abstract_summary))

    print(f"Sentiment Analysis Successful for ")
    sentiment_result=[]
    for score in df['score']:
        sentiment_result.append({
            'pos_score': score['pos'],
            'neg_score': score['neg'],
            'neu_score': score['neu'],
            'comp_score':score['compound'],
            'total':score['pos']-score['neg'],
        })
    print(sentiment_result)    

    return sentiment_result
    

def get():
     conn=connect_to_rds()
     if conn:
          print_data1(conn,'CALL_INSIGHT')
          print(abstract_summaries)

def rds(result):
        conn = connect_to_rds()
        if conn:


            create_table4(conn,'SCORES')

            for item in result:
                post = User(
                    customer=item['customer'],

                    pos_score=item['pos_score'],
                    neg_score=item['neg_score'],
                    neu_score=item['neu_score'],
                    comp_score=item['comp_score'],
                    sentiment_score=item['sentiment_score']
                    # abstract_summary1=item['abstract_summary1'],
                    # abstract_summary2=item['abstract_summary2'],
                    # prompt1=item['prompt1'],
                    # prompt2=item['prompt2']
                )
            create_data4(conn,post)
            if post.comp_score<0:
                create_data5(conn,post)

            
            


            

            print_data(conn,'SCORES')


            conn.close()


data=get()

for x in abstract_summaries:    

                    result=[]


                    data=meeting_minutes(x['abstract_summary'],x['customer'])


                    result.append(data)
                    print(result)
                    rds(result)



        

