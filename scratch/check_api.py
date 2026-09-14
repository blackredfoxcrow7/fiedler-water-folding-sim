import requests
import json

try:
    response = requests.post("http://localhost:5003/api/initialize", json={"sequence": "Ala-Ala-Ala-Ala-Ala"})
    print("Status Code:", response.status_code)
    data = response.json()
    
    print("\n--- ATOMS (first 2) ---")
    print(json.dumps(data["atoms"][:2], indent=2))
    
    print("\n--- BONDS (first 2) ---")
    print(json.dumps(data["bonds"][:2], indent=2))
    
    print("\n--- INITIAL COORDS (first 2) ---")
    print(json.dumps(data["initialCoords"][:2], indent=2))
    
    print("\n--- WATER COORDS (first 2) ---")
    print(json.dumps(data["waterCoords"][:2], indent=2))
except Exception as e:
    print("Error:", e)
