import urllib.request
import json
import numpy as np

req = urllib.request.Request(
    "http://localhost:5009/api/structure",
    data=json.dumps({"sequence": "GYDPETGTWG"}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read().decode("utf-8"))
        print("Status:", data["status"])
        print("hasFoldedTarget:", data["hasFoldedTarget"])
        
        unfolded_coords = data.get("unfoldedCoords", [])
        print("unfoldedCoords Length:", len(unfolded_coords))
        if len(unfolded_coords) > 0:
            print("First unfoldedCoord:", unfolded_coords[0])
            print("Contains NaN in unfoldedCoords:", np.isnan(unfolded_coords).any())
            
        unfolded_waters = data.get("unfoldedWaters", [])
        print("unfoldedWaters Length:", len(unfolded_waters))
        if len(unfolded_waters) > 0:
            print("First unfoldedWater:", unfolded_waters[0])
            
        target_sc = data.get("targetSidechainBonds", [])
        target_bb_sc = data.get("targetBBScBonds", [])
        target_hb = data.get("targetHBonds", [])
        print("targetHBonds:", len(target_hb))
        print("targetSidechainBonds:", len(target_sc))
        print("targetBBScBonds:", len(target_bb_sc))
except Exception as e:
    print("Error:", e)
