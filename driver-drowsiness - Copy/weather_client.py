import requests
import os

# Placeholder key - User must replace this!
OPENWEATHER_API_KEY = "d48f42dcfd12e8fa7e6ce0076486e60d"

def get_weather(city_name):
    """
    Fetch weather for a city using OpenWeatherMap free API.
    Returns a spoken string summary.
    """
    # Try getting key from environment if set, else use global default
    api_key = OPENWEATHER_API_KEY
    
    if "YOUR_" in api_key:
         return "Please configure the Open Weather API key first."

    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"

    try:
        response = requests.get(complete_url)
        data = response.json()
        print(f"[DEBUG] API Response: {data}")

        if data["cod"] != "404":
            main = data["main"]
            weather = data["weather"][0]
            temp = main["temp"]
            desc = weather["description"]
            return f"The weather in {city_name} is {desc} with a temperature of {temp} degrees Celsius."
        else:
            return "City not found."
    except Exception as e:
        print(f"[ERROR] Weather API failed: {e}")
        return "I couldn't fetch the weather right now."

if __name__ == "__main__":
    import sys
    # Test with a default city if none provided
    test_city = "Mumbai"
    if len(sys.argv) > 1:
        test_city = sys.argv[1]
    
    print(f"Testing Weather for: {test_city}")
    result = get_weather(test_city)
    print(f"Result: {result}")
