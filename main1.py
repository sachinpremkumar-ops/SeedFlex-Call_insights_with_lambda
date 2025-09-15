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
    transcription:str
    translation:str
    word_scores:str
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


def create_table(connection,table_name):
    with connection.cursor() as cursor:
        try:
            create_script = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
            No SERIAL NOT NULL PRIMARY KEY, 
            Customer VARCHAR NOT NULL,
            Abstract_Summary VARCHAR NOT NULL,
            Key_Points VARCHAR NOT NULL,
            Action_Items VARCHAR NOT NULL,
            Sentiment VARCHAR NOT NULL,
            Prompt VARCHAR NOT NULL

            );"""
            cursor.execute(create_script)
            connection.commit()
            print("Table was created successfully!")
        except Exception as e:
            print(f"Error creating table: {e}")
            connection.rollback()



def create_table1(connection,table_name):
    with connection.cursor() as cursor:
        try:
            create_script = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
            No SERIAL NOT NULL PRIMARY KEY,
            Customer VARCHAR NOT NULL,
            Size INTEGER NULL,
            Created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );"""
            cursor.execute(create_script)
            connection.commit()
            print("Table was created successfully!")
        except Exception as e:
            print(f"Error creating table: {e}")
            connection.rollback()

def create_table3(conn,table_name):
    with conn.cursor() as cursor:
        try:
            create_scripts=f"""CREATE TABLE IF NOT EXISTS {table_name}(
            No SERIAL NOT NULL PRIMARY KEY,
            Customer VARCHAR NOT NULL ,
            Word_Scores VARCHAR Null
            );"""
            cursor.execute(create_scripts)
            conn.commit()
            print("table created")
        except Exception as e:
            print(e)

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


def create_table2(conn,table_name):
   with conn.cursor() as cursor:
     try:
         create_scripts=f"""CREATE TABLE IF NOT EXISTS {table_name}(
         No SERIAL NOT NULL PRIMARY KEY,
         Customer VARCHAR NOT NULL,
         Abstract_summary1 VARCHAR NULL,
         Abstract_summary2 VARCHAR NULL,
         Prompt1 VARCHAR NULL,
         Prompt2 VARCHAR NULL
         );"""
         cursor.execute(create_scripts)
         conn.commit()
         print('table created ')
     except Exception as e:
        print(e)


def create_data(connection, post: User):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO CALL_INSIGHT(Customer,Abstract_Summary, Key_Points, Action_Items, Sentiment,Prompt,Transcription,Translation)
                VALUES (%s,%s,%s, %s, %s,%s,%s,%s) RETURNING No
            """, (post.customer,post.abstract_summary, post.key_points, post.action_items, post.sentiment,post.prompt,post.transcription,post.translation))
            connection.commit()
            print("Data inserted successfully!")


            return cursor.fetchone()[0]
    except Exception as e:
        print(f"Error inserting data: {e}")
        connection.rollback()

def create_data4(connection, post: User):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO SCORES(Customer,Positive_Score,Negative_Score,Neutral_Score,Compound_Score,Sentiment_Score)
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

def create_data1(connection, post: User):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO META_DATA(Customer,Size)
                VALUES (%s,%s) RETURNING No
            """, (post.customer,post.size))
            connection.commit()
            print("Data inserted successfully!")


            return cursor.fetchone()[0]
    except Exception as e:
        print(f"Error inserting data: {e}")
        connection.rollback()

# def create_data2(conn, post: User):
#     with conn.cursor() as cursor:
#         try:
            
#             query = f"""
#                 INSERT INTO ABSTRACT_SUMMARIES (Customer,Abstract_summary1,Abstract_summary2,Prompt1,Prompt2)
#                 VALUES (%s,%s, %s,%s,%s)
#                 RETURNING no
#              """
            

#             cursor.execute(query, (
#                 post.customer,
#                 post.abstract_summary1,
#                 post.abstract_summary2,
#                 post.prompt1,
#                 post.prompt2
#             ))
            
            
#             result = cursor.fetchone()
#             conn.commit()
#             print("Data Inserted")
#             return result  
        
#         except Exception as e:
#             conn.rollback()  
#             print(f"Error: {e}")
#             return None


def create_data3(conn,post:User):
    try:
        with conn.cursor() as cursor:
            with conn.cursor() as cursor:
                cursor.execute("""
                INSERT INTO WORD_SCORES(Customer,Word_Scores)
                VALUES (%s,%s) RETURNING No
                """, (post.customer,post.word_scores))
                conn.commit()
                print("Data inserted successfully!")


            return cursor.fetchone()[0]
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()

        


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


client = OpenAI(
    organization='org-z5MJeMy5jvChDTPbiZexZDh8',
    project='proj_njFOpVgWCOhoidNpHimw4Ake',
)


class Translation:
       def __init__(self, text):
        self.text = text

class Transcription:
       def __init__(self, text):
        self.text = text

prompts=[]


# prompt_files=['prompt1.txt','prompt2.txt','prompt3.txt','prompt4.txt']
# for x in prompt_files:
#     if os.path.exists(x):
#      with open(x,"r") as f:
#         data=f.read()
#         prompts.append(data)
#     else:
#         pass

# print(prompts)

# def tandt(audio_file,file_name):
#     audio_file_content = (file_name, audio_file)
#     transcript = client.audio.transcriptions.create(
#         model="whisper-1",
#         file=audio_file_content
#         #,
#         #language="en"
#     )
    
#     # print(transcript)

#     translatedtranscript = client.audio.translations.create(
#         model="whisper-1",
#         file=audio_file_content
#     )
#     # print(translatedtranscript)
#     return translatedtranscript.text

def transcribe(audio_file,file_name):
    audio_file_content = (file_name, audio_file)
    transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file_content
        #,
        #language="en"
    )
    return transcript.text

def translate(audio_file,file_name):
    audio_file_content = (file_name, audio_file)
    translatedtranscript = client.audio.translations.create(
    model="whisper-1",
    file=audio_file_content
    )
    # print(translatedtranscript)
    return translatedtranscript.text
    

def meeting_minutes(transcription,translatedtranscript,x,y):

    abstract_summary = ''
    key_points = ''
    action_items = ''
    abstract_summary = abstract_summary_extraction(transcription)
    key_points = key_points_extraction(transcription)
    action_items = action_item_extraction(transcription)
    sentiment = sentiment_analysis(transcription)
    word_scores=word_scores_extraction(abstract_summary)
    sentiment_score=sentiment_chart(abstract_summary,f'{file}.png')
    if sentiment_score and isinstance(sentiment_score, list):
        sentiment_score = sentiment_score[0]  
    else:
        sentiment_score = {}
    return User(
        customer=x,
        key_points=key_points,
        action_items=action_items,
        sentiment=sentiment,
        size=y,
        # abstract_summary1=abstract_summary.get('abstract_summary1', ''),
        # abstract_summary2=abstract_summary.get('abstract_summary2', ''),
        # prompt1=abstract_summary.get('prompt1', ''),
        # prompt2=abstract_summary.get('prompt2', '')
        abstract_summary=abstract_summary.get('abstract_summary',''),
        prompt=abstract_summary.get('prompt', ''),
        transcription=translatedtranscript,
        translation=transcription,
        word_scores=word_scores.get('word_scores',''),
        pos_score=sentiment_score.get('pos_score',''),
        neg_score=sentiment_score.get('neg_score',''),
        neu_score=sentiment_score.get('neu_score',''),
        comp_score=sentiment_score.get('comp_score',''),
        sentiment_score=sentiment_score.get('total','')
        
    ).dict()

# def abstract_summary_extraction(transcription):
  
#   prompts_result={}
#   for x,i in enumerate(prompts):
#     response = client.chat.completions.create(
#         model="gpt-4",
#         temperature=0,  
#         messages=[
#             {
#                 "role": "system",
#                 "content": i
#             },
#             {
#                 "role": "user",
#                 "content": transcription
#             }
        
#         ]
#     )

#     summary = response.choices[0].message.content
#     prompts_result[f"abstract_summary{x+1}"] = summary
#     prompts_result[f"prompt{x+1}"]=i

#   return prompts_result

def abstract_summary_extraction(transcription):
  
    prompts_result={}
    prompt="You are a highly skilled AI trained in language comprehension and summarization. Please read the following conversation between a sales agent from SEEDFLEX and a customer who is an e-commerce SME merchant seller. Summarize the conversation into a concise abstract paragraph. Focus on capturing the main arguments, key details, and important conclusions. The summary should be clear and succinct, providing a well-rounded overview of the discussion’s content to help someone understand the main points without needing to read the entire text. Specifically, identify if the customer liked the loan product and, if not, what their concerns were. Avoid unnecessary details, tangential points, pause words, and fillers. Ensure that all major conclusions and significant details are clearly represented."
  
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,  
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": transcription
            }
        
        ]
    )

    summary = response.choices[0].message.content
    prompts_result[f"abstract_summary"] = summary
    prompts_result[f"prompt"]=prompt

    return prompts_result


def key_points_extraction(transcription):
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Please read the following conversation between a sales agent from SEEDFLEX and a customer who is an e-commerce SME merchant seller. Based on the text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand the conversation."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content


def action_item_extraction(transcription):
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI expert in analyzing conversations and extracting action items. Please read the following conversation between a sales agent from SEEDFLEX and a customer who is an e-commerce SME merchant seller. Identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. These could include tasks assigned to specific individuals or general actions the group has decided to take. List these action items clearly and concisely."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content


def sentiment_analysis(transcription):
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                #"content": "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the following text. Please consider the overall tone of the discussion, the emotio>
                "content": "As an AI with expertise in language and emotion analysis, please read the following conversation between a sales agent from SEEDFLEX and a customer who is an e-commerce SME merchant seller. Determine if the customer is genuinely interested in the SEEDFLEX product. Additionally, assess if the customer understands that this product does not require any documentation and that by consenting to SEEDFLEX pulling their sales data, they would be able to receive a free account with a credit limit."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )       
    return response.choices[0].message.content

nltk.download('punkt')
def word_scores_extraction(abstract_summary):
    data={}
    tokens=nltk.word_tokenize(abstract_summary['abstract_summary'].lower())
    scores={word:sia.lexicon.get(word,0) for word in tokens}
    
    data['word_scores'] =json.dumps(scores) 
    return data



s3_client= boto3.client('s3')

def is_supported_format(file_name):
    if file_name.startswith('processed/'):
        return False
    supported_formats = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
    return file_name.split('.')[-1].lower() in supported_formats



def s3_move(key):
    try:
        file_name = key.split('/')[-1]  
        processed_key = f'processed/{file_name}'
        if s3_compare(processed_key):
            print(f"{key} already exists in the 'processed/' folder.")
            return False


        audio_data = s3_get(key)
        s3_client.put_object(
            ACL='private',  
            Bucket='experiment2407',
            Body=audio_data,
            Key=processed_key
        )
        s3_client.delete_object(
            Bucket='experiment2407',
            Key=key,
        )
        print(f'{key} Moved to {processed_key}')
        
        return True
    except Exception as e:
        print(e)


def s3_get(key):
    try:
        audio=s3_client.get_object(
        Bucket='experiment2407',
        Key=key)['Body'].read()
        return io.BytesIO(audio)
    except ClientError as e:
        print(f"Error fetching object from S3: {e}")
        raise e

def s3_compare(key):
    try:
        s3_client.head_object(
            Bucket='experiment2407',
            Key=key
        )
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            print(f"Error checking object existence: {e}")
            raise e

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


def sentiment_chart(data,key):

    if isinstance(data, dict):
        data = [data]
    Content={
            'NEGATIVE':[],
            'POSITIVE':[],
            'NEUTRAL':[],
            'COMPOUND':[]
            }
    
    
    df=pd.DataFrame(data)
    df['score']=df['abstract_summary'].apply(lambda abstract_summary:sia.polarity_scores(abstract_summary))
    for i in df['score']:
    
        Content['NEGATIVE'].append(i['neg'])
        Content['POSITIVE'].append(i['pos'])
        Content['NEUTRAL'].append(i['neu'])
        Content['COMPOUND'].append(i['compound'])

    Content_df=pd.DataFrame(Content)



    bars=fig,ax=plt.subplots(figsize=(10,8))

    colors=['red','green','blue','orange']


    Content_df[['NEGATIVE','POSITIVE','NEUTRAL','COMPOUND']].plot(kind='bar',ax=ax,color=colors)
    plt.xlabel(f'Index(Compound Score is {Content["COMPOUND"]})')
    plt.ylabel('Score')
    plt.legend(['NEGATIVE', 'POSITIVE', 'NEUTRAL', 'COMPOUND'])
    for bar in  ax.containers:
        ax.bar_label(bar,fmt='%2f',label_type='edge')
    plt.fill()
    # plt.savefig(f'{key}')
    # s3_upload(f'{key}')
    # server_delete(f'{key}')
    print(f"Sentiment Analysis Successful for {key}")
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
    
    

def rds(result):
        conn = connect_to_rds()
        if conn:
            create_table(conn,'CALL_INSIGHT')
            create_table1(conn,'META_DATA')
            create_table2(conn,"ABSTRACT_SUMMARIES")
            create_table3(conn,'WORD_SCORES')
            create_table4(conn,"SCORES")

            for item in result:
                post = User(
                    customer=item['customer'],
                    abstract_summary=item['abstract_summary'],
                    key_points=item['key_points'],
                    action_items=item['action_items'],
                    sentiment=item['sentiment'],
                    size=item['size'],
                    prompt=item['prompt'],
                    transcription=item['transcription'],
                    translation=item['translation'],
                    word_scores=item['word_scores'],
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

            create_data(conn,post)
            create_data1(conn,post)
            # create_data2(conn,post)
            create_data3(conn,post)
            create_data4(conn,post)
            if post.comp_score<0:
                create_data5(conn,post)
            # print_data(conn,'CALL_INSIGHT')
            # print_data(conn,'META_DATA')
            # print_data(conn,'WORD_SCORES')
            # print_data(conn,'SCORES')
            # print_data(conn,"ABSTRACT_SUMMARIES")

            # ids=[]
            # for x in ids:
            #  delete_data(conn,x,table_name=)
            conn.close()

response2=s3_client.list_objects(
    Bucket='experiment2407'
)
if 'Contents' not in response2:
    print("No files found in the bucket.")


batch=[]

size=[]
for x in response2["Contents"]: 
    

    print(x['Key'])


    
    if  is_supported_format(x['Key']):
        print(f"Adding file to batch: {x['Key']}")
        batch_data=x['Key']
        size_data=x['Size']
        batch.append(batch_data)
        size.append(size_data)
        print(batch)
        
        
        
        if len(batch)>=1:
            for i,file in enumerate(batch):
                file_size=size[i]      
                if s3_move(file):
                    result=[]
                    processed_key = f'processed/{file.split("/")[-1]}'
                    audio_file=s3_get(processed_key)
                    transcription=transcribe(audio_file,file)
                    translatedtranscript=translate(audio_file,file)
                    data=meeting_minutes(translatedtranscript,transcription,file,file_size)
                    
                    
                    result.append(data)
                    # print(result)
                    rds(result)
        
            batch=[]
            size=[]
            break


if batch:
    for i,items in enumerate(batch):
        print(f"processing {items}")
        file_size=size[i]
        while s3_move(items) is True:
            result=[]
            processed_key = f'processed/{file.split("/")[-1]}'
            audio_file=s3_get(processed_key)
            transcription=transcribe(audio_file,file)
            translatedtranscript=translate(transcription)
            data=meeting_minutes(translatedtranscript,transcription,file,file_size)
            
            
            result.append(data)
            # print(result)
            rds(result)


        

