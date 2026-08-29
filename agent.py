from pydantic import BaseModel, Field
import openai
import json

class SatQuerySchema(BaseModel):
    location: str = Field(description="The geographical area, city, or coordinates mentioned.")
    target_features: list[str] = Field(description="List of objects or phenomena to identify.")
    time_period: str = Field(description="The requested date, year, or season.")

def parse_sat_query(user_prompt: str):
    client = openai.OpenAI(api_key="YOUR_OPENAI_OR_TOGETHER_KEY")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are SatQuery Geo-Parser. Extract key geospatial entities from user queries into structured JSON."},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# Usage
parsed_data = parse_sat_query("Find illegal aquaculture ponds near Chilika Lake from 2025 imagery")
print(parsed_data)