import os
import json
import requests  # Required for API requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API keys
OPENAI_API_KEY = os.getenv("GITHUB_TOKEN")
OPENWEATHER_API_KEY = os.getenv("WEATHER_KEY")

# Initialize OpenAI client
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=OPENAI_API_KEY,
)

def get_coordinates(city_name):
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={OPENWEATHER_API_KEY}"
    response = requests.get(geo_url)
    
    if response.status_code == 200 and response.json():
        data = response.json()[0]
        # print(data["local_names"]["sr"])
        return data["lat"], data["lon"]
    else:
        return None, None  
    
def get_weather(city_name):
    latitude, longitude = get_coordinates(city_name)
    
    if latitude is None or longitude is None:
        return {"error": f"City '{city_name}' not found. Please check the spelling."}
    
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    response = requests.get(weather_url)
    if response.status_code == 200:
        data = response.json()
        temperature_celsius = data["main"]["temp"]
        temperature_fahrenheit = round(temperature_celsius * 9/5 + 32, 1)
        description = data["weather"][0]["description"]

        return {
            "city": city_name.capitalize(),
            "temperature_celsius": temperature_celsius,
            "temperature_fahrenheit": temperature_fahrenheit,
            "description": description.capitalize()
        }
    else:
        return {"error": f"Failed to fetch weather data for '{city_name}'."}

# Step 2: Define the function schema for the model
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city_name": {"type": "string"}
            },
            "required": ["city_name"]
        }
    }
}]

# Step 3: Get user input for city name
city = input("Enter a city name: ")
question = input(f"Ask anything about {city}: ")

# Step 4: Send user request to OpenAI
messages = [{"role": "user", "content": f"What's the weather like in {city} today? {question}"}]

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

# Step 5: Extract function call request (if model calls it)
tool_calls = completion.choices[0].message.tool_calls

if tool_calls:
    tool_call = tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    # Execute function
    result = get_weather(args["city_name"])

    # Step 6: Send function result back to model
    messages.append(completion.choices[0].message)  # Append function call message
    messages.append({  # Append function result
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

    # Call model again with updated messages
    completion_2 = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    # Step 7: Print final response from the model
    print(completion_2.choices[0].message.content)

else:
    print("Model did not call the function.")
