import requests
import os
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()


API_TOKEN = os.getenv("HUBSPOT_API_KEY")

if not API_TOKEN:
    raise ValueError("Please set HUBSPOT_API_KEY in your .env file")


CONTACT_ID = "435730262246" 


note_data = {
    "properties": {
        "hs_note_body": "This is a test note from AI insights 🚀",
        "hs_timestamp": int(datetime.now().timestamp() * 1000)
    }
}

try:
    note_response = requests.post(
        "https://api.hubapi.com/crm/v3/objects/notes",
        headers={
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        },
        json=note_data
    )
    note_response.raise_for_status()
    note_id = note_response.json()["id"]
    print(f"✅ Note created successfully with ID: {note_id}")
except requests.exceptions.HTTPError as err:
    print(f"❌ Failed to create note: {err.response.text}")
    raise

assoc_data = {
    "inputs": [
        {
            "from": {"id": note_id},
            "to": {"id": CONTACT_ID},
            "type": "note_to_contact"
        }
    ]
}

try:
    assoc_response = requests.post(
        "https://api.hubapi.com/crm/v3/associations/notes/contacts/batch/create",
        headers={
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        },
        json=assoc_data
    )
    assoc_response.raise_for_status()
    print(f"✅ Note associated with contact {CONTACT_ID} successfully!")
except requests.exceptions.HTTPError as err:
    print(f"❌ Failed to associate note: {err.response.text}")
    raise
