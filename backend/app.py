# ================================================================
#  Smart Crop Advisor - Flask Backend API v3.0
#  OpenWeatherMap Integration + ML Predictions
#  Run AFTER training: python app.py
#  API: http://localhost:5000
# ================================================================

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, json, os, numpy as np, pandas as pd
import urllib.request, urllib.parse, urllib.error, socket, time
from datetime import datetime
from collections import defaultdict

# ================================================================
# SECTION 1: FLASK SETUP
# ================================================================
app = Flask(__name__)
CORS(app)

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(BASE_DIR, 'crop_model.pkl')
CROP_INFO_PATH   = os.path.join(BASE_DIR, 'crop_info.json')
HISTORY_PATH     = os.path.join(BASE_DIR, 'prediction_history.json')
PRICE_MODEL_PATH = os.path.join(BASE_DIR, 'price_models.pkl')
CROP_ECON_PATH   = os.path.join(BASE_DIR, 'crop_economics.json')

# ================================================================
# SECTION 2: HINDI NAMES
# ================================================================
HINDI_NAMES = {
    "rice":        "धान",
    "maize":       "मक्का",
    "chickpea":    "चना",
    "kidneybeans": "राजमा",
    "pigeonpeas":  "अरहर",
    "mungbean":    "मूंग",
    "blackgram":   "उड़द",
    "lentil":      "मसूर",
    "mothbeans":   "मोठ",
    "pomegranate": "अनार",
    "banana":      "केला",
    "mango":       "आम",
    "grapes":      "अंगूर",
    "watermelon":  "तरबूज",
    "muskmelon":   "खरबूजा",
    "apple":       "सेब",
    "orange":      "संतरा",
    "papaya":      "पपीता",
    "coconut":     "नारियल",
    "cotton":      "कपास",
    "jute":        "जूट",
    "coffee":      "कॉफी"
}

# ================================================================
# SECTION 3: INPUT VALIDATION RANGES
# ================================================================
VALID_RANGES = {
    "N":           (0,   140,  "Nitrogen (kg/ha)"),
    "P":           (0,   145,  "Phosphorus (kg/ha)"),
    "K":           (0,   205,  "Potassium (kg/ha)"),
    "temperature": (0,   50,   "Temperature (degC)"),
    "humidity":    (0,   100,  "Humidity (%)"),
    "ph":          (3.0, 10.0, "Soil pH"),
    "rainfall":    (0,   500,  "Rainfall (mm)")
}

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ================================================================
# SECTION 4: LOAD MODEL & CROP INFO
# ================================================================
print("=" * 58)
print("  Smart Crop Advisor - Flask Backend v3.0 Starting...")
print("=" * 58)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "crop_model.pkl not found! Run 'python train_model.py' first."
    )

with open(MODEL_PATH, 'rb') as f:
    pkg = pickle.load(f)

model         = pkg["model"]
class_names   = pkg["class_names"]
model_name    = pkg["model_name"]
test_accuracy = pkg["test_accuracy"]
cv_accuracy   = pkg["cv_accuracy"]
cv_std        = pkg["cv_std"]
dataset_rows  = pkg["dataset_rows"]
n_classes     = pkg["n_classes"]
is_calibrated = pkg.get("is_calibrated", False)

with open(CROP_INFO_PATH, 'r', encoding='utf-8') as f:
    crop_info = json.load(f)

price_models = {}
if os.path.exists(PRICE_MODEL_PATH):
    with open(PRICE_MODEL_PATH, 'rb') as f:
        price_models = pickle.load(f)

crop_economics = {}
if os.path.exists(CROP_ECON_PATH):
    with open(CROP_ECON_PATH, 'r', encoding='utf-8') as f:
        crop_economics = json.load(f)

print(f"  [OK] Model        : {model_name}")
print(f"  [OK] Test Acc     : {test_accuracy*100:.2f}%")
print(f"  [OK] CV Acc       : {cv_accuracy*100:.2f}%")
print(f"  [OK] Calibrated   : {is_calibrated}")
print(f"  [OK] Crops        : {n_classes}")
print(f"  [OK] Price Models : {len(price_models)} crops loaded")
print(f"  [OK] Economics    : {len(crop_economics)} crops loaded")
print(f"  [OK] API ready    : http://localhost:5000")
print("=" * 58)


# ================================================================
# SECTION 5: OPENWEATHERMAP CONFIGURATION
# ================================================================
# STEP 1: Go to https://openweathermap.org/api
# STEP 2: Sign Up (free, no credit card needed)
# STEP 3: Go to your Profile → API Keys tab
# STEP 4: Copy your key and paste it below replacing YOUR_API_KEY_HERE
# NOTE: New keys activate within 2 hours of account creation
# ================================================================
OWM_API_KEY  = "4621bc3408a17d5f7247e06a5b5319c8"
OWM_CURRENT  = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORECAST = "https://api.openweathermap.org/data/2.5/forecast"

# Popular Indian city name corrections (user types → OWM name)
CITY_CORRECTIONS = {
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "madras": "Chennai",
    "poona": "Pune",
    "bangalore": "Bengaluru",
    "banglore": "Bengaluru",
    "ahmedabad": "Ahmedabad",
    "hyderabad": "Hyderabad",
    "delhi": "New Delhi",
}


def _fetch_json(url):
    """Fetches JSON from a URL using stdlib only (no requests library needed)."""
    with urllib.request.urlopen(url, timeout=8) as resp:
        return json.loads(resp.read().decode())


def _weather_emoji(owm_icon):
    """Maps OpenWeatherMap 2-char icon code to a weather emoji."""
    emoji_map = {
        "01": "\u2600\ufe0f",  # clear sky
        "02": "\U0001f324\ufe0f",  # few clouds
        "03": "\u26c5",  # scattered clouds
        "04": "\u2601\ufe0f",  # overcast clouds
        "09": "\U0001f327\ufe0f",  # shower rain
        "10": "\U0001f326\ufe0f",  # rain
        "11": "\u26c8\ufe0f",  # thunderstorm
        "13": "\U0001f328\ufe0f",  # snow
        "50": "\U0001f32b\ufe0f",  # mist
    }
    code = (owm_icon or "01")[:2]
    return emoji_map.get(code, "\U0001f321\ufe0f")


def _wind_direction(deg):
    """Converts wind degree to compass direction string."""
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[int((deg + 22.5) / 45) % 8]


def _farming_advisories(temp, humidity, rain_mm, wind_kph, description):
    """
    Generates 1-3 context-aware farming advisories from live weather data.
    Returns list of {icon, color, border, title, message} dicts.
    """
    advisories = []
    desc = description.lower()

    if rain_mm > 20:
        advisories.append({
            "icon": "\U0001f327\ufe0f", "color": "#1a4a7a", "border": "#4a90d9",
            "title": f"Heavy rainfall expected ({rain_mm:.0f}mm)",
            "message": "Avoid pesticide or fertilizer spraying. Check field drainage. Risk of waterlogging in low-lying fields."
        })
    elif rain_mm > 5 or "rain" in desc or "shower" in desc:
        advisories.append({
            "icon": "\U0001f326\ufe0f", "color": "#2d5a27", "border": "#4a7c3f",
            "title": "Rain expected — skip irrigation today",
            "message": "Reduce irrigation by 50-100%. Good opportunity for transplanting seedlings."
        })

    if humidity > 85:
        advisories.append({
            "icon": "\u26a0\ufe0f", "color": "#7a4a00", "border": "#e8a020",
            "title": f"High humidity alert ({humidity}%)",
            "message": "High risk of fungal diseases (blight, mildew, rust). Apply preventive fungicide spray."
        })
    elif humidity < 30:
        advisories.append({
            "icon": "\U0001f4a7", "color": "#1a4a7a", "border": "#4a90d9",
            "title": f"Low humidity ({humidity}%) — increase irrigation",
            "message": "Crops may wilt. Irrigate early morning. Use mulching to retain soil moisture."
        })

    if temp > 40:
        advisories.append({
            "icon": "\U0001f321\ufe0f", "color": "#c0392b", "border": "#e74c3c",
            "title": f"Extreme heat warning ({temp:.0f}\u00b0C)",
            "message": "Avoid afternoon field work. Irrigate at sunrise or sunset. Risk of heat stress and flower drop."
        })
    elif temp > 35:
        advisories.append({
            "icon": "\u2600\ufe0f", "color": "#8b5e00", "border": "#e8a020",
            "title": f"Hot conditions ({temp:.0f}\u00b0C)",
            "message": "Increase irrigation frequency. Best field activity window: early morning before 9 AM."
        })
    elif temp < 10:
        advisories.append({
            "icon": "\u2744\ufe0f", "color": "#1a4a7a", "border": "#4a90d9",
            "title": f"Cold weather ({temp:.0f}\u00b0C) — frost risk",
            "message": "Protect nursery beds. Delay transplanting sensitive crops. Cover young plants at night."
        })

    if wind_kph > 40:
        advisories.append({
            "icon": "\U0001f4a8", "color": "#7a4a00", "border": "#e8a020",
            "title": f"Strong winds ({wind_kph:.0f} km/h) — spraying not advised",
            "message": "Pesticide drift risk is very high. Stake tall crops. Secure net houses and poly tunnels."
        })

    if "thunderstorm" in desc:
        advisories.append({
            "icon": "\u26c8\ufe0f", "color": "#c0392b", "border": "#e74c3c",
            "title": "Thunderstorm conditions",
            "message": "Do not operate machinery outdoors. Stay indoors. Harvest mature crops before storm if possible."
        })

    if not advisories:
        if 18 <= temp <= 32 and 40 <= humidity <= 80:
            advisories.append({
                "icon": "\u2705", "color": "#2d5a27", "border": "#4a7c3f",
                "title": f"Excellent farming day — {temp:.0f}\u00b0C, {humidity}% humidity",
                "message": "Ideal conditions for field operations, spraying, sowing, and transplanting."
            })
        else:
            advisories.append({
                "icon": "\U0001f331", "color": "#4a7c3f", "border": "#6fa85c",
                "title": "Normal farming conditions",
                "message": "Proceed with planned field activities. Monitor crops for any stress symptoms."
            })

    return advisories[:3]


# ================================================================
# SECTION 6: OPENWEATHERMAP API ENDPOINT
# ================================================================
@app.route('/weather', methods=['GET'])
def get_weather():
    """
    GET /weather?city=Nashik
    
    Fetches live weather from OpenWeatherMap for any Indian city.
    Returns current conditions, 7-day forecast, and farming advisories.
    
    REQUIRES: Set OWM_API_KEY above with your free key from openweathermap.org
    """
    # Check if API key is configured
    if OWM_API_KEY == "YOUR_API_KEY_HERE":
        return jsonify({
            "error":              "OpenWeatherMap API key not configured.",
            "instruction":        "Open backend/app.py and replace 'YOUR_API_KEY_HERE' with your free API key.",
            "get_key_from":       "https://openweathermap.org/api",
            "steps":              [
                "1. Visit https://openweathermap.org/api",
                "2. Click 'Sign In' → Create free account",
                "3. Go to your Profile → 'API Keys' tab",
                "4. Copy the default key (or generate a new one)",
                "5. Paste it in backend/app.py replacing YOUR_API_KEY_HERE",
                "6. Restart the Flask server"
            ],
            "api_key_configured": False
        }), 503

    city = request.args.get("city", "").strip()
    if not city:
        return jsonify({"error": "City name required. Use: /weather?city=Mumbai"}), 400

    # Apply city name corrections
    city_corrected = CITY_CORRECTIONS.get(city.lower(), city)
    query = city_corrected + ",IN"

    try:
        # ── Current Weather ────────────────────────────────────────
        cur_url = (
            f"{OWM_CURRENT}?q={urllib.parse.quote(query)}"
            f"&appid={OWM_API_KEY}&units=metric"
        )
        cur = _fetch_json(cur_url)

        rain_now = 0.0
        if "rain" in cur:
            rain_now = cur["rain"].get("1h", 0.0)

        wind_ms  = cur["wind"]["speed"]
        wind_kph = round(wind_ms * 3.6, 1)
        wind_deg = cur["wind"].get("deg", 0)

        current = {
            "temp":        round(cur["main"]["temp"], 1),
            "feels_like":  round(cur["main"]["feels_like"], 1),
            "humidity":    cur["main"]["humidity"],
            "description": cur["weather"][0]["description"].title(),
            "emoji":       _weather_emoji(cur["weather"][0]["icon"]),
            "icon":        cur["weather"][0]["icon"],
            "wind_kph":    wind_kph,
            "wind_dir":    _wind_direction(wind_deg),
            "rain_mm":     rain_now,
            "pressure":    cur["main"].get("pressure", 1013),
            "visibility":  round(cur.get("visibility", 10000) / 1000, 1),
            "timestamp":   datetime.fromtimestamp(cur["dt"]).isoformat(timespec="seconds")
        }

        # ── 5-Day Forecast → Daily Summary ────────────────────────
        fc_url = (
            f"{OWM_FORECAST}?q={urllib.parse.quote(query)}"
            f"&appid={OWM_API_KEY}&units=metric&cnt=40"
        )
        fc = _fetch_json(fc_url)

        daily = defaultdict(lambda: {
            "temps": [], "humidities": [], "pops": [], "rain": 0.0,
            "icons": [], "descs": []
        })
        day_abbr  = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        mon_abbr  = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

        for item in fc["list"]:
            d = datetime.fromtimestamp(item["dt"])
            dk = d.strftime("%Y-%m-%d")
            daily[dk]["temps"].append(item["main"]["temp"])
            daily[dk]["humidities"].append(item["main"]["humidity"])
            daily[dk]["pops"].append(int(item.get("pop", 0) * 100))
            if "rain" in item:
                daily[dk]["rain"] += item["rain"].get("3h", 0.0)
            daily[dk]["icons"].append(item["weather"][0]["icon"])
            daily[dk]["descs"].append(item["weather"][0]["description"])

        forecast = []
        for dk in sorted(daily.keys())[:7]:
            d   = datetime.strptime(dk, "%Y-%m-%d")
            dd  = daily[dk]
            bi  = max(set(dd["icons"]), key=dd["icons"].count)
            forecast.append({
                "day":      day_abbr[d.weekday()],
                "date":     f"{mon_abbr[d.month-1]} {d.day}",
                "emoji":    _weather_emoji(bi),
                "icon":     bi,
                "temp_max": round(max(dd["temps"])),
                "temp_min": round(min(dd["temps"])),
                "rain_mm":  round(dd["rain"], 1),
                "pop":      max(dd["pops"])
            })

        # ── Farming Advisories ─────────────────────────────────────
        advisories = _farming_advisories(
            current["temp"], current["humidity"],
            current["rain_mm"], wind_kph,
            cur["weather"][0]["description"]
        )

        all_temps = []
        all_humids = []
        for dk in sorted(daily.keys())[:7]:
            all_temps.extend(daily[dk]["temps"])
            all_humids.extend(daily[dk]["humidities"])
            
        avg_7_day_temp = sum(all_temps) / len(all_temps) if all_temps else current["temp"]
        avg_7_day_humidity = sum(all_humids) / len(all_humids) if all_humids else current["humidity"]

        return jsonify({
            "city":               cur["name"],
            "country":            cur["sys"]["country"],
            "lat":                round(cur["coord"]["lat"], 4),
            "lon":                round(cur["coord"]["lon"], 4),
            "current":            current,
            "forecast":           forecast,
            "average_7_day": {
                "temp": round(avg_7_day_temp, 1),
                "humidity": round(avg_7_day_humidity, 1)
            },
            "advisories":         advisories,
            "api_key_configured": True,
            "fetched_at":         datetime.now().isoformat(timespec="seconds")
        })

    except urllib.error.HTTPError as e:
        if e.code == 401:
            return jsonify({
                "error": "Invalid API key.",
                "fix": "Check OWM_API_KEY in backend/app.py. New keys take up to 2 hours to activate."
            }), 401
        elif e.code == 404:
            return jsonify({
                "error": f"City '{city}' not found.",
                "suggestion": "Try adding state name e.g. 'Pune Maharashtra' or use English spelling."
            }), 404
        else:
            return jsonify({"error": f"OpenWeatherMap API error {e.code}"}), 502
    except Exception as e:
        return jsonify({"error": f"Weather fetch failed: {str(e)}"}), 500


# ================================================================
# SECTION 7: ML PREDICTION HELPERS
# ================================================================
def generate_smart_reason(crop_name, N, P, K, temperature, humidity, ph, rainfall):
    crop = crop_name.lower().strip()
    reasons = []
    if crop == 'rice':
        if rainfall >= 150: reasons.append(f"high rainfall ({rainfall:.0f}mm) ensures water for paddy fields")
        if humidity >= 75:  reasons.append(f"high humidity ({humidity:.0f}%) supports paddy cultivation")
        if 20 <= temperature <= 35: reasons.append(f"temperature ({temperature:.1f}\u00b0C) is optimal for rice")
        if 5.5 <= ph <= 7.0: reasons.append(f"soil pH ({ph:.1f}) is ideal for paddy")
        if N >= 60: reasons.append(f"adequate nitrogen ({N} kg/ha) supports grain formation")
    elif crop == 'maize':
        if 100 <= rainfall <= 500: reasons.append(f"moderate rainfall ({rainfall:.0f}mm) suits corn")
        if 20 <= temperature <= 30: reasons.append(f"warm temperature ({temperature:.1f}\u00b0C) accelerates corn growth")
        if N >= 80: reasons.append(f"high nitrogen ({N} kg/ha) supports vegetative growth")
    elif crop == 'jute':
        if rainfall >= 150: reasons.append(f"high rainfall ({rainfall:.0f}mm) supports jute fiber development")
        if humidity >= 70:  reasons.append(f"high humidity ({humidity:.0f}%) is ideal for jute")
        if temperature >= 24: reasons.append(f"warm temperature ({temperature:.1f}\u00b0C) promotes jute growth")
    elif crop == 'chickpea':
        if rainfall < 150: reasons.append(f"low rainfall ({rainfall:.0f}mm) suits chickpea's dry season requirements")
        if 15 <= temperature <= 25: reasons.append(f"cool temperature ({temperature:.1f}\u00b0C) is ideal for Rabi chickpea")
        if P >= 40: reasons.append(f"adequate phosphorus ({P} kg/ha) supports root development")
    elif crop == 'kidneybeans':
        if 60 <= rainfall <= 200: reasons.append(f"rainfall ({rainfall:.0f}mm) provides moisture for beans")
        if 15 <= temperature <= 28: reasons.append(f"moderate temperature ({temperature:.1f}\u00b0C) prevents pod drop")
    elif crop == 'pigeonpeas':
        if rainfall < 200: reasons.append(f"moderate rainfall ({rainfall:.0f}mm) suits this drought-tolerant crop")
        if 25 <= temperature <= 35: reasons.append(f"warm temperature ({temperature:.1f}\u00b0C) is ideal for arhar")
    elif crop == 'mungbean':
        if 60 <= rainfall <= 150: reasons.append(f"moderate rainfall ({rainfall:.0f}mm) is adequate for mungbean")
        if 25 <= temperature <= 35: reasons.append(f"warm temperature ({temperature:.1f}\u00b0C) promotes fast growth")
    elif crop == 'blackgram':
        if 60 <= rainfall <= 150: reasons.append(f"moderate rainfall ({rainfall:.0f}mm) suits blackgram")
        if 25 <= temperature <= 35: reasons.append(f"warm temperature ({temperature:.1f}\u00b0C) is optimal for urad")
    elif crop == 'lentil':
        if rainfall < 150: reasons.append(f"low rainfall ({rainfall:.0f}mm) suits lentil's minimal water needs")
        if 15 <= temperature <= 25: reasons.append(f"cool climate ({temperature:.1f}\u00b0C) is ideal for Rabi lentil")
    elif crop == 'mothbeans':
        if rainfall < 200: reasons.append(f"low rainfall ({rainfall:.0f}mm) matches moth bean's arid-zone adaptability")
        if temperature >= 25: reasons.append(f"warm temperature ({temperature:.1f}\u00b0C) suits this heat-tolerant crop")
    elif crop == 'pomegranate':
        if rainfall < 150: reasons.append(f"low rainfall ({rainfall:.0f}mm) suits pomegranate's drought tolerance")
        if 25 <= temperature <= 38: reasons.append(f"hot dry temperature ({temperature:.1f}\u00b0C) promotes fruit quality")
        if humidity < 70: reasons.append(f"low humidity ({humidity:.0f}%) prevents fruit cracking")
    elif crop == 'banana':
        if rainfall >= 100: reasons.append(f"high water availability ({rainfall:.0f}mm) supports banana")
        if temperature >= 25: reasons.append(f"tropical temperature ({temperature:.1f}\u00b0C) is optimal")
        if humidity >= 70:  reasons.append(f"high humidity ({humidity:.0f}%) promotes bunch development")
    elif crop == 'mango':
        if rainfall < 200: reasons.append(f"moderate dry conditions ({rainfall:.0f}mm) promote mango flowering")
        if temperature >= 24: reasons.append(f"warm temperature ({temperature:.1f}\u00b0C) is ideal for mango")
    elif crop == 'grapes':
        if rainfall < 100: reasons.append(f"low rainfall ({rainfall:.0f}mm) concentrates grape sugars")
        if humidity < 70: reasons.append(f"low humidity ({humidity:.0f}%) reduces fungal disease in vineyards")
        if K >= 40: reasons.append(f"adequate potassium ({K} kg/ha) improves berry quality")
    elif crop == 'watermelon':
        if rainfall <= 200: reasons.append(f"low rainfall ({rainfall:.0f}mm) suits dryland watermelon")
        if temperature >= 25: reasons.append(f"hot temperature ({temperature:.1f}\u00b0C) promotes sweetness")
    elif crop == 'muskmelon':
        if rainfall <= 150: reasons.append(f"low rainfall ({rainfall:.0f}mm) is adequate for muskmelon")
        if temperature >= 25: reasons.append(f"warm temperature ({temperature:.1f}\u00b0C) promotes aroma")
    elif crop == 'apple':
        if temperature <= 20: reasons.append(f"cool temperature ({temperature:.1f}\u00b0C) provides chilling hours for apple")
        if humidity < 75: reasons.append(f"moderate humidity ({humidity:.0f}%) reduces apple scab risk")
    elif crop == 'orange':
        if 75 <= rainfall <= 250: reasons.append(f"moderate rainfall ({rainfall:.0f}mm) suits Nagpur mandarin")
        if 15 <= temperature <= 35: reasons.append(f"subtropical temperature ({temperature:.1f}\u00b0C) is ideal")
    elif crop == 'papaya':
        if 100 <= rainfall <= 250: reasons.append(f"moderate rainfall ({rainfall:.0f}mm) is suited for papaya")
        if temperature >= 22: reasons.append(f"tropical temperature ({temperature:.1f}\u00b0C) promotes fruiting")
    elif crop == 'coconut':
        if rainfall >= 150: reasons.append(f"high rainfall ({rainfall:.0f}mm) meets coconut's water needs")
        if temperature >= 25: reasons.append(f"tropical temperature ({temperature:.1f}\u00b0C) is optimal")
        if humidity >= 70: reasons.append(f"high humidity ({humidity:.0f}%) is ideal for nariyal")
    elif crop == 'cotton':
        if 50 <= rainfall <= 250: reasons.append(f"moderate rainfall ({rainfall:.0f}mm) suits cotton")
        if temperature >= 25: reasons.append(f"warm temperature ({temperature:.1f}\u00b0C) promotes boll development")
        if K >= 40: reasons.append(f"adequate potassium ({K} kg/ha) improves fiber quality")
    elif crop == 'coffee':
        if rainfall >= 150: reasons.append(f"high rainfall ({rainfall:.0f}mm) is essential for shade-grown coffee")
        if 15 <= temperature <= 28: reasons.append(f"mild temperature ({temperature:.1f}\u00b0C) produces quality beans")
        if 5.5 <= ph <= 6.5: reasons.append(f"acidic soil pH ({ph:.1f}) is optimal for coffee")
    else:
        if N >= 60: reasons.append(f"adequate nitrogen ({N} kg/ha)")
        if 15 <= temperature <= 35: reasons.append(f"suitable temperature ({temperature:.1f}\u00b0C)")

    if not reasons:
        reasons = [
            f"compatible soil (N={N}, P={P}, K={K})",
            f"suitable climate ({temperature:.1f}\u00b0C, {humidity:.0f}% humidity, {rainfall:.0f}mm)"
        ]
    top = reasons[:3]
    if len(top) == 1: combined = top[0]
    elif len(top) == 2: combined = f"{top[0]} and {top[1]}"
    else: combined = f"{top[0]}, {top[1]}, and {top[2]}"
    return f"{crop_name.capitalize()} is recommended because {combined}."


def get_confidence_label(score):
    if score > 80:   return "Very High"
    elif score > 60: return "High"
    elif score > 40: return "Moderate"
    else:            return "Low"


def generate_ai_summary(N, P, K, temperature, humidity, ph, rainfall, recs):
    if not recs:
        return "Insufficient data. Check your input values."
    n_level   = "high" if N > 80 else "moderate" if N > 40 else "low"
    p_level   = "high" if P > 80 else "moderate" if P > 40 else "low"
    k_level   = "high" if K > 100 else "moderate" if K > 50 else "low"
    ph_desc   = "acidic" if ph < 6 else "alkaline" if ph > 7.5 else "neutral to slightly acidic"
    rain_desc = "very high" if rainfall > 300 else "high" if rainfall > 150 else "moderate" if rainfall > 80 else "low"
    hum_desc  = "very high" if humidity > 85 else "high" if humidity > 70 else "moderate" if humidity > 50 else "low"
    temp_desc = "hot" if temperature > 35 else "warm" if temperature > 25 else "moderate" if temperature > 15 else "cool"
    top      = recs[0]
    top_conf = get_confidence_label(top["score"])
    lines = [
        f"<strong>Soil &amp; Climate Profile:</strong> Your farm shows {n_level} nitrogen "
        f"({N} kg/ha), {p_level} phosphorus ({P} kg/ha), and {k_level} potassium ({K} kg/ha). "
        f"Soil pH is {ph:.1f} ({ph_desc}), temperature is {temperature}\u00b0C ({temp_desc}), "
        f"humidity is {humidity}% ({hum_desc}), and annual rainfall is {rainfall} mm ({rain_desc})."
    ]
    lines.append(
        f"<br><br><strong>Primary Recommendation &mdash; {top['name_en']} ({top['name_hi']}) "
        f"[{top_conf} Confidence: {top['score']}%]:</strong> {top['reason']}"
    )
    if len(recs) >= 2:
        sec = recs[1]
        sec_conf = get_confidence_label(sec["score"])
        lines.append(
            f"<br><br><strong>Alternative &mdash; {sec['name_en']} ({sec['name_hi']}) "
            f"[{sec_conf} Confidence: {sec['score']}%]:</strong> {sec['reason']}"
        )
    if rainfall > 150 and humidity > 70:
        lines.append("<br><br><em style='color:#4a7c3f;'>\u2022 Advisory: High moisture detected. Prefer water-intensive Kharif crops.</em>")
    elif rainfall < 100:
        lines.append("<br><br><em style='color:#c77c00;'>\u2022 Advisory: Low rainfall. Consider drought-tolerant crops or drip irrigation.</em>")
    if ph < 5.5:
        lines.append(f"<br><br><em style='color:#c0392b;'>\u2022 Advisory: Strongly acidic soil (pH {ph:.1f}). Apply lime.</em>")
    elif ph > 8.0:
        lines.append(f"<br><br><em style='color:#c0392b;'>\u2022 Advisory: Alkaline soil (pH {ph:.1f}). Apply gypsum or sulphur.</em>")

    # Add Market & Profit Advisory for Top Crop
    top_pf = top.get("price_forecast", {})
    top_prof = top.get("estimated_profit_per_hectare", {})
    if top_pf and top_pf.get("available"):
        trend_sym = top_pf.get("trend_symbol", "➡️ stable")
        fc_price = top_pf.get("forecast_price", 0)
        net_prof = top_prof.get("estimated_profit_inr_ha", 0)
        prof_str = f"₹{net_prof:,.0f}" if net_prof >= 0 else f"-₹{abs(net_prof):,.0f}"
        lines.append(
            f"<br><br><strong>📈 Market Price &amp; Profit Advisory ({top['name_en']}):</strong> "
            f"8-week mandi price outlook is <strong>{trend_sym}</strong> (avg ~₹{fc_price:,.0f}/quintal). "
            f"Estimated net profit is <strong>{prof_str} / ha</strong> based on an average yield of {top_prof.get('yield_q_ha', 0)} q/ha."
        )

    return "".join(lines)


CROP_OPTIMAL = {
    "rice":        {"rainfall": (150, 300), "humidity": (75, 90), "temp": (20, 35), "ph": (5.5, 7.0), "N": (60, 120)},
    "maize":       {"rainfall": (100, 500), "humidity": (50, 80), "temp": (20, 30), "ph": (5.5, 7.5), "N": (80, 140)},
    "jute":        {"rainfall": (150, 300), "humidity": (70, 90), "temp": (24, 38), "ph": (6.0, 8.0), "N": (60, 100)},
    "chickpea":    {"rainfall": (40, 150),  "humidity": (30, 65), "temp": (15, 25), "ph": (6.0, 8.0), "N": (0, 40)},
    "kidneybeans": {"rainfall": (60, 200),  "humidity": (50, 75), "temp": (15, 28), "ph": (6.0, 7.5), "N": (20, 60)},
    "pigeonpeas":  {"rainfall": (60, 200),  "humidity": (40, 75), "temp": (25, 35), "ph": (6.0, 7.5), "N": (0, 40)},
    "mungbean":    {"rainfall": (60, 150),  "humidity": (50, 70), "temp": (25, 35), "ph": (6.0, 7.5), "N": (10, 40)},
    "blackgram":   {"rainfall": (60, 150),  "humidity": (50, 75), "temp": (25, 35), "ph": (6.0, 7.5), "N": (10, 40)},
    "lentil":      {"rainfall": (30, 150),  "humidity": (30, 60), "temp": (15, 25), "ph": (6.0, 8.0), "N": (10, 40)},
    "mothbeans":   {"rainfall": (20, 100),  "humidity": (20, 50), "temp": (25, 45), "ph": (7.0, 9.0), "N": (0, 40)},
    "pomegranate": {"rainfall": (50, 150),  "humidity": (30, 65), "temp": (25, 38), "ph": (6.5, 8.5), "N": (20, 60)},
    "banana":      {"rainfall": (100, 300), "humidity": (70, 90), "temp": (25, 35), "ph": (6.0, 7.5), "N": (80, 140)},
    "mango":       {"rainfall": (75, 200),  "humidity": (30, 65), "temp": (24, 40), "ph": (5.5, 7.5), "N": (40, 100)},
    "grapes":      {"rainfall": (30, 100),  "humidity": (30, 65), "temp": (15, 35), "ph": (6.0, 7.5), "N": (20, 60)},
    "watermelon":  {"rainfall": (40, 200),  "humidity": (30, 65), "temp": (25, 40), "ph": (6.0, 7.5), "N": (20, 60)},
    "muskmelon":   {"rainfall": (40, 150),  "humidity": (30, 65), "temp": (25, 40), "ph": (6.0, 7.5), "N": (20, 60)},
    "apple":       {"rainfall": (100, 300), "humidity": (50, 75), "temp": (5,  22), "ph": (5.5, 7.0), "N": (40, 100)},
    "orange":      {"rainfall": (75, 250),  "humidity": (40, 70), "temp": (15, 35), "ph": (5.5, 7.5), "N": (40, 100)},
    "papaya":      {"rainfall": (100, 250), "humidity": (60, 85), "temp": (22, 35), "ph": (6.0, 7.0), "N": (60, 120)},
    "coconut":     {"rainfall": (150, 300), "humidity": (70, 90), "temp": (25, 35), "ph": (5.5, 8.0), "N": (40, 100)},
    "cotton":      {"rainfall": (50, 250),  "humidity": (30, 65), "temp": (25, 40), "ph": (6.0, 8.0), "N": (40, 100)},
    "coffee":      {"rainfall": (150, 300), "humidity": (60, 85), "temp": (15, 28), "ph": (5.5, 6.5), "N": (40, 100)},
}


def generate_why_not(crop_name, N, P, K, temperature, humidity, ph, rainfall):
    crop = crop_name.lower()
    opt  = CROP_OPTIMAL.get(crop, {})
    issues = []
    if "rainfall" in opt:
        rlo, rhi = opt["rainfall"]
        if rainfall < rlo: issues.append(f"Rainfall ({rainfall:.0f}mm) too low \u2014 needs {rlo}\u2013{rhi}mm")
        elif rainfall > rhi: issues.append(f"Rainfall ({rainfall:.0f}mm) too high \u2014 ideal is {rlo}\u2013{rhi}mm")
    if "humidity" in opt:
        hlo, hhi = opt["humidity"]
        if humidity < hlo: issues.append(f"Humidity ({humidity:.0f}%) too low \u2014 optimal is {hlo}\u2013{hhi}%")
        elif humidity > hhi: issues.append(f"Humidity ({humidity:.0f}%) too high \u2014 causes disease risk")
    if "temp" in opt:
        tlo, thi = opt["temp"]
        if temperature < tlo: issues.append(f"Temperature ({temperature:.1f}\u00b0C) too cool \u2014 needs {tlo}\u2013{thi}\u00b0C")
        elif temperature > thi: issues.append(f"Temperature ({temperature:.1f}\u00b0C) too hot \u2014 optimal {tlo}\u2013{thi}\u00b0C")
    if "ph" in opt:
        phlo, phhi = opt["ph"]
        if ph < phlo: issues.append(f"Soil pH ({ph:.1f}) too acidic \u2014 needs {phlo}\u2013{phhi}")
        elif ph > phhi: issues.append(f"Soil pH ({ph:.1f}) too alkaline \u2014 optimal {phlo}\u2013{phhi}")
    if "N" in opt:
        nlo, nhi = opt["N"]
        if N < nlo: issues.append(f"Nitrogen ({N} kg/ha) below minimum of {nlo} kg/ha")
        elif N > nhi: issues.append(f"Nitrogen ({N} kg/ha) too high \u2014 prefers {nlo}\u2013{nhi} kg/ha")
    if not issues:
        issues.append("Other crops better match the complete soil and climate profile")
    return issues[:3]


# ================================================================
# SECTION 7B: RULES ENGINE (Post-Prediction Context Adjustment)
# ================================================================
# Crop classification tables for rules
SEASON_CROPS = {
    "kharif": {"rice","maize","cotton","jute","blackgram","mungbean",
               "pigeonpeas","mothbeans","banana","papaya","coconut"},
    "rabi":   {"chickpea","lentil","kidneybeans","orange","pomegranate",
               "grapes","apple","mango"},
    "zaid":   {"watermelon","muskmelon","mungbean","maize"}
}

HIGH_WATER_CROPS   = {"rice","banana","jute","coconut","coffee","papaya","sugarcane"}
DROUGHT_CROPS      = {"mothbeans","chickpea","pigeonpeas","cotton","mungbean",
                       "lentil","pomegranate","grapes"}
LEGUMES            = {"chickpea","lentil","mungbean","blackgram","pigeonpeas",
                       "mothbeans","kidneybeans"}
HIGH_VALUE_CROPS   = {"coffee","apple","grapes","cotton","pomegranate","mango"}
FAST_GROWING_CROPS = {"mungbean","blackgram","maize"}
SLOW_GROWING_CROPS = {"apple","coffee","coconut","mango"}
EXPORT_CROPS       = {"coffee","cotton","grapes","mango","banana","pomegranate","rice"}
HARDY_CROPS        = {"mothbeans","pigeonpeas","cotton","maize"}
PEST_SENSITIVE     = {"grapes","apple","coffee","mango"}

ROTATION_BOOST = {
    "cotton":  (LEGUMES,                          "ideal rotation after Cotton — legumes restore soil nitrogen"),
    "rice":    ({"chickpea","lentil","mungbean"},  "good Rabi rotation after Kharif Rice"),
    "maize":   ({"chickpea","lentil","mungbean","blackgram"}, "legumes after Maize improve soil health"),
    "wheat":   ({"mungbean","blackgram","maize"},  "good rotation after Wheat"),
    "soybean": ({"rice","maize","cotton"},         "cereal/cash crop rotation after Soybean"),
    "onion":   ({"maize","cotton","rice"},         "suitable rotation after Onion"),
}

GOAL_CONFIG = {
    "Maximum Profit":  {"boost": HIGH_VALUE_CROPS,     "boost_x": 1.25, "boost_note": "high market value crop",
                        "pen":   {"mungbean","blackgram","lentil"}, "pen_x": 0.7,  "pen_note": "low market value"},
    "Fast Growing":    {"boost": FAST_GROWING_CROPS,   "boost_x": 1.3,  "boost_note": "short growing cycle (60-90 days)",
                        "pen":   SLOW_GROWING_CROPS,  "pen_x":   0.4,  "pen_note": "very long maturity (3-10 years)"},
    "Low Water Use":   {"boost": DROUGHT_CROPS,        "boost_x": 1.3,  "boost_note": "low water requirement",
                        "pen":   HIGH_WATER_CROPS,     "pen_x":   0.3,  "pen_note": "extremely high water requirement"},
    "Export Quality":  {"boost": EXPORT_CROPS,         "boost_x": 1.2,  "boost_note": "high export demand crop",
                        "pen":   set(),                "pen_x":   1.0,  "pen_note": ""},
    "Sustainable":     {"boost": LEGUMES,              "boost_x": 1.25, "boost_note": "improves soil health — sustainable choice",
                        "pen":   {"cotton"},            "pen_x":   0.65, "pen_note": "heavy pesticide requirement"},
    "Low Risk":        {"boost": {"rice","maize","chickpea","mungbean"}, "boost_x": 1.2, "boost_note": "stable yields and reliable market",
                        "pen":   {"grapes","apple","coffee"}, "pen_x": 0.6, "pen_note": "volatile market, management-intensive"},
}


def apply_rules_engine(recs, context):
    if not context:
        return recs, []

    season       = context.get("season", "").lower().split("(")[0].strip()
    water        = context.get("water_availability", "Good")
    irrigation   = context.get("irrigation", "")
    last_crop    = context.get("last_crop", "").lower().strip()
    pest_level   = context.get("pest_level", "None")
    farming_pref = context.get("farming_pref", "Chemical")
    goals        = [g.strip() for g in context.get("goals", [])]
    weather_alerts = context.get("weather_alerts", [])

    rules_log = []
    adjusted  = []

    for rec in recs:
        crop    = rec["name_en"].lower()
        mult    = 1.0
        notes   = []

        if season in SEASON_CROPS:
            if crop in SEASON_CROPS[season]:
                mult *= 1.1
                notes.append(f"✅ In-season crop for {season.capitalize()}")
            else:
                mult *= 0.15
                notes.append(f"❌ Off-season — not recommended for {season.capitalize()} season")

        if water == "Scarce":
            if crop in HIGH_WATER_CROPS:
                mult *= 0.2
                notes.append("❌ Penalized: high water demand vs your scarce supply")
            elif crop in DROUGHT_CROPS:
                mult *= 1.3
                notes.append("✅ Boosted: drought-tolerant — suits scarce water conditions")
        elif water == "Moderate":
            if crop in HIGH_WATER_CROPS:
                mult *= 0.7
                notes.append("⚠️ Slightly penalized: high water demand vs moderate supply")

        if irrigation == "Rainfed":
            if crop in DROUGHT_CROPS:
                mult *= 1.15
                notes.append("✅ Boosted: suitable for rainfed farming")
            elif crop in HIGH_WATER_CROPS:
                mult *= 0.6
                notes.append("❌ Penalized: rainfed insufficient for high water demand")

        if last_crop in ROTATION_BOOST:
            boost_set, rot_note = ROTATION_BOOST[last_crop]
            if crop in boost_set:
                mult *= 1.2
                notes.append(f"✅ Boosted: {rot_note}")

        if pest_level == "Severe":
            if crop in HARDY_CROPS:
                mult *= 1.15
                notes.append("✅ Boosted: naturally pest-resistant crop")
            elif crop in PEST_SENSITIVE:
                mult *= 0.5
                notes.append("❌ Penalized: highly sensitive to pest damage")
        elif pest_level == "Moderate":
            if crop in PEST_SENSITIVE:
                mult *= 0.75
                notes.append("⚠️ Slightly penalized: moderate pest sensitivity")

        if farming_pref == "Organic":
            if crop in LEGUMES:
                mult *= 1.2
                notes.append("✅ Boosted: thrives under organic management")
            elif crop == "cotton":
                mult *= 0.6
                notes.append("❌ Penalized: Cotton requires heavy pesticide use")
        elif farming_pref == "Mixed":
            if crop in LEGUMES:
                mult *= 1.1
                notes.append("✅ Slight boost: legumes reduce chemical fertilizer needs")

        is_heavy_rain = any("Heavy rainfall" in a or "Thunderstorm" in a for a in weather_alerts)
        is_heatwave   = any("Extreme heat" in a or "Hot conditions" in a for a in weather_alerts)
        
        if is_heavy_rain:
            if crop in HIGH_WATER_CROPS:
                mult *= 1.2
                notes.append("✅ Boosted: incoming rain is highly beneficial for this water-loving crop")
            elif crop in PEST_SENSITIVE or crop in {"papaya", "lentil", "cotton"}:
                mult *= 0.5
                notes.append("❌ Penalized: sensitive to waterlogging and disease from incoming heavy rain")
                
        if is_heatwave:
            if crop in DROUGHT_CROPS or crop in HARDY_CROPS:
                mult *= 1.15
                notes.append("✅ Boosted: naturally tolerant to incoming extreme heat")
            elif crop in HIGH_WATER_CROPS or crop in {"apple", "grapes"}:
                mult *= 0.6
                notes.append("❌ Penalized: incoming heatwave severely increases evaporation stress")

        for goal in goals:
            cfg = GOAL_CONFIG.get(goal)
            if not cfg: continue
            if crop in cfg["boost"]:
                mult *= cfg["boost_x"]
                notes.append(f"✅ [{goal}] {cfg['boost_note']}")
            if crop in cfg["pen"] and cfg["pen_x"] < 1.0:
                mult *= cfg["pen_x"]
                notes.append(f"❌ [{goal}] {cfg['pen_note']}")

        new_rec = dict(rec)
        new_rec["_raw_score"] = rec["score"]
        new_rec["_mult"]      = round(mult, 4)
        new_rec["rules_applied"] = notes if notes else ["No context rules applied"]
        new_rec["score"] = round(rec["score"] * mult, 2)
        adjusted.append(new_rec)

    total = sum(r["score"] for r in adjusted)
    if total > 0:
        factor = 100.0 / total
        for r in adjusted:
            r["score"] = min(round(r["score"] * factor, 2), 98.0)

    adjusted = sorted(adjusted, key=lambda x: -x["score"])
    adjusted = [r for r in adjusted if r["score"] >= 1.0][:5]

    for r in adjusted:
        r["confidence"] = get_confidence_label(r["score"])

    summary_parts = []
    if season in SEASON_CROPS: summary_parts.append(f"{season.capitalize()} season filter applied")
    if water != "Good": summary_parts.append(f"{water} water availability considered")
    if last_crop in ROTATION_BOOST: summary_parts.append(f"Crop rotation after {last_crop.capitalize()} considered")
    if goals: summary_parts.append(f"goals: {', '.join(goals)}")
    
    if "rules_summary" not in context:
        context["rules_summary"] = summary_parts
        
    return adjusted, summary_parts


# ================================================================
# SECTION 7C: MARKET PRICE FORECASTING & PROFIT ESTIMATION
# ================================================================
def get_price_forecast(crop_name, horizon_weeks=8):
    crop = crop_name.lower().strip()
    pm = price_models.get(crop)
    if not pm:
        return {
            "latest_price": None,
            "forecast_price": None,
            "trend": "stable",
            "trend_symbol": "➡️ stable",
            "change_pct": 0.0,
            "forecast": [],
            "confidence_low": [],
            "confidence_high": [],
            "available": False
        }

    fcst_series = pm.get("forecast", [])
    forecast_price = round(float(np.mean(fcst_series)), 2) if fcst_series else pm.get("last_price")

    return {
        "latest_price": pm.get("last_price"),
        "forecast_price": forecast_price,
        "trend": pm.get("trend", "stable"),
        "trend_symbol": pm.get("trend_symbol", "➡️ stable"),
        "change_pct": pm.get("change_pct", 0.0),
        "forecast": fcst_series,
        "confidence_low": pm.get("confidence_low", []),
        "confidence_high": pm.get("confidence_high", []),
        "available": True
    }


def estimate_profit_per_hectare(crop_name, forecast_price):
    crop = crop_name.lower().strip()
    econ = crop_economics.get(crop, {})
    if not econ or forecast_price is None:
        return {
            "estimated_profit_inr_ha": 0.0,
            "yield_q_ha": 0.0,
            "cost_inr_ha": 0.0,
            "revenue_inr_ha": 0.0,
            "source": "N/A"
        }

    yield_q = econ.get("avg_yield_per_hectare_quintal", 0.0)
    cost_inr = econ.get("avg_input_cost_per_hectare_inr", 0.0)

    # Mandi price is in INR / Quintal
    revenue_inr = round(yield_q * forecast_price, 2)
    profit_inr = round(revenue_inr - cost_inr, 2)

    return {
        "estimated_profit_inr_ha": profit_inr,
        "yield_q_ha": yield_q,
        "cost_inr_ha": cost_inr,
        "revenue_inr_ha": revenue_inr,
        "source": econ.get("source", "ICAR / CACP")
    }


def validate_inputs(data):
    errors = []
    for field, (lo, hi, label) in VALID_RANGES.items():
        if field not in data:
            errors.append(f"Missing field '{field}' ({label})")
            continue
        try:
            val = float(data[field])
        except (ValueError, TypeError):
            errors.append(f"'{field}' must be a number. Got: {data[field]}")
            continue
        if not (lo <= val <= hi):
            errors.append(f"'{field}' ({label}) = {val} is out of range [{lo}, {hi}].")
    return (len(errors) == 0), errors


def get_top_recommendations(N, P, K, temperature, humidity, ph, rainfall,
                             top_n=5, min_score_pct=1.0):
    input_df = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURE_COLS
    )
    proba = model.predict_proba(input_df)[0]
    crop_scores = sorted(zip(class_names, proba), key=lambda x: -x[1])
    recs = []
    for crop_name, prob in crop_scores:
        score = round(prob * 100, 2)
        if score < min_score_pct: continue
        info   = crop_info.get(crop_name.lower(), {})
        reason = generate_smart_reason(crop_name, N, P, K, temperature, humidity, ph, rainfall)
        pf     = get_price_forecast(crop_name)
        prof   = estimate_profit_per_hectare(crop_name, pf.get("forecast_price"))
        recs.append({
            "name_en":                      crop_name.capitalize(),
            "name_hi":                      HINDI_NAMES.get(crop_name.lower(), crop_name),
            "score":                        score,
            "confidence":                   get_confidence_label(score),
            "reason":                       reason,
            "season":                       info.get("season", "Season varies"),
            "water_requirement":            info.get("water_requirement", "Moderate"),
            "tips":                         info.get("tips", "Follow standard farming practices."),
            "emoji":                        info.get("emoji", "\U0001f331"),
            "price_forecast":               pf,
            "estimated_profit_per_hectare": prof
        })
        if len(recs) >= top_n: break
    return recs


def save_prediction(inputs, recs):
    history = load_prediction_history()
    entry = {
        "id":       len(history) + 1,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "inputs":   inputs,
        "top_crop": recs[0]["name_en"] if recs else "N/A",
        "results":  recs
    }
    history.append(entry)
    with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return entry["id"]


def load_prediction_history():
    if not os.path.exists(HISTORY_PATH): return []
    with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
        try:    return json.load(f)
        except: return []


# ================================================================
# SECTION 9: PREDICTION API ENDPOINTS
# ================================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":    "running",
        "message":   "Smart Crop Advisor API v3.0 is live!",
        "timestamp": datetime.now().isoformat(timespec='seconds')
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "best_model":                model_name,
        "test_accuracy":             round(test_accuracy * 100, 2),
        "cross_validation_accuracy": round(cv_accuracy * 100, 2),
        "cross_validation_std":      round(cv_std * 100, 2),
        "dataset_rows":              dataset_rows,
        "crop_classes":              n_classes,
        "is_calibrated":             is_calibrated,
        "features":                  FEATURE_COLS
    })


@app.route('/price-forecast', methods=['GET'])
def crop_price_forecast():
    """
    GET /price-forecast?crop=rice  or GET /price-forecast?crop=all
    Returns 8-week price forecast, trend, and profit estimation for a crop or all 22 crops.
    """
    crop = request.args.get("crop", "").strip().lower()
    if crop == 'all' or not crop:
        all_forecasts = []
        for c_name in class_names:
            pf = get_price_forecast(c_name)
            prof = estimate_profit_per_hectare(c_name, pf.get("forecast_price"))
            info = crop_info.get(c_name.lower(), {})
            all_forecasts.append({
                "crop": c_name.capitalize(),
                "crop_key": c_name.lower(),
                "name_hi": HINDI_NAMES.get(c_name.lower(), c_name),
                "emoji": info.get("emoji", "🌱"),
                "price_forecast": pf,
                "economics": prof
            })
        return jsonify({"total_crops": len(all_forecasts), "crops": all_forecasts})

    pf = get_price_forecast(crop)
    prof = estimate_profit_per_hectare(crop, pf.get("forecast_price"))
    return jsonify({
        "crop":           crop.capitalize(),
        "price_forecast": pf,
        "economics":      prof
    })


# ================================================================
# SECTION 8B: LIVE MANDI MARKET PRICES (data.gov.in + SARIMAX HYBRID)
# ================================================================
DATA_GOV_IN_API_KEY = os.environ.get(
    "DATA_GOV_IN_API_KEY",
    "579b464db66ec23bdd00000175692175101d4a0f5e0e859c59d4c495"
)
DATA_GOV_IN_MANDI_RESOURCE = "9ef84268-d588-465a-a308-a864a43d0070"
DATA_GOV_IN_BASE_URL = f"https://api.data.gov.in/resource/{DATA_GOV_IN_MANDI_RESOURCE}"

REQUEST_TIMEOUT_SECONDS = 6
MAX_LIMIT = 100

AGMARKNET_COMMODITY_MAP = {
    "rice":        ["Rice", "Paddy(Dhan)(Common)", "Paddy(Dhan)(Basmati)"],
    "maize":       ["Maize"],
    "chickpea":    ["Bengal Gram(Gram)(Whole)", "Gram Raw(Chholia)", "Chana"],
    "kidneybeans": ["Rajma", "Rajmah"],
    "pigeonpeas":  ["Arhar (Tur/Red Gram)(Whole)", "Tur"],
    "mungbean":    ["Green Gram (Moong)(Whole)", "Moong(Gram)"],
    "blackgram":   ["Black Gram (Urd Beans)(Whole)", "Urad"],
    "lentil":      ["Masur Dal", "Lentil (Masur)(Whole)"],
    "mothbeans":   ["Moth"],
    "pomegranate": ["Pomegranate", "Anar"],
    "banana":      ["Banana"],
    "mango":       ["Mango", "Mango (Raw-Ripe)"],
    "grapes":      ["Grapes"],
    "watermelon":  ["Water Melon"],
    "muskmelon":   ["Muskmelon", "Kharbuja"],
    "apple":       ["Apple"],
    "orange":      ["Orange", "Santra", "Mousambi"],
    "papaya":      ["Papaya", "Papaya (Raw)"],
    "coconut":     ["Coconut"],
    "cotton":      ["Cotton"],
    "jute":        ["Jute"],
    "coffee":      ["Coffee"]
}

_MANDI_CACHE = {}
_CACHE_TTL_SECONDS = 20 * 60  # 20 minutes in-memory cache


def _cache_get(key):
    entry = _MANDI_CACHE.get(key)
    if entry is None:
        return None
    value, expires_at = entry
    if time.time() > expires_at:
        _MANDI_CACHE.pop(key, None)
        return None
    return value


def _cache_set(key, value):
    _MANDI_CACHE[key] = (value, time.time() + _CACHE_TTL_SECONDS)


def _safe_float(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.upper() in ("NR", "N/A", "NA", "-"):
        return default
    text = text.replace(",", "")
    try:
        return float(text)
    except (ValueError, TypeError):
        return default


def fetch_live_mandi_prices(crop_key, state=None, district=None, limit=50):
    """
    Queries api.data.gov.in for live commodity prices, testing all aliases.
    Returns: (list_of_cleaned_records, is_live_boolean, error_message_or_None)
    """
    crop_key_norm = (crop_key or "").lower().strip()
    aliases = AGMARKNET_COMMODITY_MAP.get(crop_key_norm)
    if not aliases:
        aliases = [crop_key.capitalize()]

    limit = max(1, min(int(limit or 50), MAX_LIMIT))

    base_params = {
        "api-key": DATA_GOV_IN_API_KEY,
        "format": "json",
        "limit": limit,
    }
    if state and state.strip().lower() != "all":
        base_params["filters[state]"] = state.strip()
    if district and district.strip().lower() != "all":
        base_params["filters[district]"] = district.strip()

    last_error = None

    for alias in aliases:
        params = dict(base_params)
        params["filters[commodity]"] = alias

        url = f"{DATA_GOV_IN_BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                raw = resp.read()
                data = json.loads(raw)
        except urllib.error.HTTPError as e:
            last_error = f"HTTP error {e.code} for alias '{alias}'"
            continue
        except urllib.error.URLError as e:
            last_error = f"URL error for alias '{alias}': {e.reason}"
            continue
        except socket.timeout:
            last_error = f"Timeout for alias '{alias}'"
            continue
        except json.JSONDecodeError:
            last_error = f"Invalid JSON response for alias '{alias}'"
            continue
        except Exception as e:
            last_error = f"Unexpected error for alias '{alias}': {e}"
            continue

        records = data.get("records", []) if isinstance(data, dict) else []
        if not records:
            last_error = last_error or f"No records for alias '{alias}'"
            continue

        cleaned = []
        for rec in records:
            try:
                cleaned.append({
                    "state": (rec.get("state") or "").strip().title(),
                    "district": (rec.get("district") or "").strip().title(),
                    "market": (rec.get("market") or "").strip().title(),
                    "commodity": (rec.get("commodity") or "").strip(),
                    "variety": (rec.get("variety") or "").strip(),
                    "arrival_date": rec.get("arrival_date", ""),
                    "min_price": _safe_float(rec.get("min_price")),
                    "max_price": _safe_float(rec.get("max_price")),
                    "modal_price": _safe_float(rec.get("modal_price")),
                })
            except Exception as e:
                last_error = f"Skipped malformed record: {e}"
                continue

        if cleaned:
            return cleaned, True, None

    return [], False, last_error


def _summarize_records(records):
    positive_priced = [r for r in records if r["modal_price"] > 0]

    if not positive_priced:
        return {
            "total_mandis_reporting": len(records),
            "avg_modal_price": 0.0,
            "min_reported_price": 0.0,
            "max_reported_price": 0.0,
            "price_spread": 0.0,
            "highest_market": None,
            "lowest_market": None,
        }

    modal_prices = [r["modal_price"] for r in positive_priced]
    avg_modal = sum(modal_prices) / len(modal_prices)

    highest = max(positive_priced, key=lambda r: r["modal_price"])
    lowest = min(positive_priced, key=lambda r: r["modal_price"])

    def _market_card(rec):
        return {
            "market": rec["market"],
            "district": rec["district"],
            "state": rec["state"],
            "modal_price": rec["modal_price"],
            "arrival_date": rec["arrival_date"],
        }

    return {
        "total_mandis_reporting": len(records),
        "avg_modal_price": round(avg_modal, 2),
        "min_reported_price": min(modal_prices),
        "max_reported_price": max(modal_prices),
        "price_spread": round(max(modal_prices) - min(modal_prices), 2),
        "highest_market": _market_card(highest),
        "lowest_market": _market_card(lowest),
    }


def _build_crop_payload(crop_key, state, district, limit):
    crop_key = (crop_key or "rice").lower().strip()
    cache_key = (crop_key, state or "", district or "", limit)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    records, is_live, error = fetch_live_mandi_prices(crop_key, state, district, limit)
    pf = get_price_forecast(crop_key)
    summary = _summarize_records(records)

    if is_live and summary["avg_modal_price"] > 0:
        base_price = summary["avg_modal_price"]
        source_label = "data.gov.in (live)"
    else:
        base_price = pf.get("forecast_price") or pf.get("latest_price")
        source_label = "SARIMAX Model (Offline Fallback)"

    economics = estimate_profit_per_hectare(crop_key, base_price)
    economics = dict(economics)
    economics.setdefault("source", "ICAR / CACP Benchmark")

    info = crop_info.get(crop_key, {})

    payload = {
        "status": "success" if is_live else "fallback",
        "crop": crop_key.capitalize(),
        "crop_key": crop_key,
        "name_hi": HINDI_NAMES.get(crop_key, info.get("name_hi", crop_key.capitalize())),
        "emoji": info.get("emoji", "🌱"),
        "source": source_label,
        "is_live": is_live,
        "filters_applied": {
            "crop": crop_key,
            "state": state or "All",
            "district": district or "All",
            "limit": limit,
        },
        "summary": summary,
        "live_mandis": records,
        "forecast": {
            "horizon_weeks": 8,
            "trend": pf.get("trend", "stable"),
            "trend_symbol": pf.get("trend_symbol", "➡️ stable"),
            "change_pct": pf.get("change_pct", 0),
            "forecast_price_avg": pf.get("forecast_price", 0),
            "weekly_forecast": pf.get("forecast", []),
            "confidence_low": pf.get("confidence_low", []),
            "confidence_high": pf.get("confidence_high", []),
            "available": pf.get("available", False)
        },
        "economics": economics,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }

    if not is_live:
        payload["notice"] = (
            "Live mandi rates currently unavailable for this selection. "
            "Showing trained model projections."
        )
        if error:
            payload["_debug_error"] = error

    _cache_set(cache_key, payload)
    return payload


@app.route('/market-prices', methods=['GET'])
def get_market_prices():
    """
    GET /market-prices?crop=rice&state=Maharashtra&district=Nashik&limit=50
    GET /market-prices?crop=all&state=Maharashtra
    """
    crop = request.args.get('crop', 'rice').strip().lower()
    state = request.args.get('state')
    district = request.args.get('district')
    limit = request.args.get('limit', 50, type=int)
    limit = max(1, min(limit, MAX_LIMIT))

    # Single crop detail
    if crop != 'all':
        payload = _build_crop_payload(crop, state, district, limit)
        return jsonify(payload)

    # Multi-crop overview
    cache_key = ("all", state or "", district or "", limit)
    cached = _cache_get(cache_key)
    if cached is not None:
        return jsonify(cached)

    results = []
    for crop_key in class_names:
        try:
            results.append(_build_crop_payload(crop_key, state, district, limit))
        except Exception as e:
            results.append({
                "status": "error",
                "crop": crop_key.capitalize(),
                "crop_key": crop_key,
                "is_live": False,
                "error": str(e),
            })

    response = {
        "status": "success",
        "filters_applied": {
            "crop": "all",
            "state": state or "All",
            "district": district or "All",
            "limit": limit,
        },
        "crops": results,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }

    _cache_set(cache_key, response)
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received."}), 400
    ok, errors = validate_inputs(data)
    if not ok:
        return jsonify({
            "error": "Input validation failed.", "details": errors,
            "valid_ranges": {k: f"{lo} to {hi}" for k, (lo, hi, _) in VALID_RANGES.items()}
        }), 422
    N, P, K     = float(data["N"]), float(data["P"]), float(data["K"])
    temperature = float(data["temperature"])
    humidity    = float(data["humidity"])
    ph          = float(data["ph"])
    rainfall    = float(data["rainfall"])
    # ── Extract optional farm context for Rules Engine ────────
    context = data.get("context", {})

    try:
        recs = get_top_recommendations(N, P, K, temperature, humidity, ph, rainfall)
    except Exception as e:
        return jsonify({"error": "Prediction failed.", "details": str(e)}), 500

    # ── Apply Rules Engine (post-ML adjustment) ────────────────
    rules_summary = []
    rules_engine_active = bool(context)
    if rules_engine_active:
        recs, rules_summary = apply_rules_engine(recs, context)

    inputs_log = {"N": N, "P": P, "K": K, "temperature": temperature,
                  "humidity": humidity, "ph": ph, "rainfall": rainfall,
                  "context": context}
    pred_id    = save_prediction(inputs_log, recs)
    ai_summary = generate_ai_summary(N, P, K, temperature, humidity, ph, rainfall, recs)

    # Append rules engine summary paragraph to AI summary
    if rules_summary:
        rules_text = "; ".join(rules_summary)
        ai_summary += (
            f"<br><br><strong>\U0001f527 Context Rules Applied:</strong> "
            f"Based on your farm context ({rules_text}), the ML scores have been "
            f"adjusted to better match your specific conditions. "
            f"Crops that are off-season, water-intensive (given scarce supply), "
            f"or misaligned with your goals have been deprioritized."
        )

    input_df   = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURE_COLS)
    all_proba  = model.predict_proba(input_df)[0]
    all_sorted = sorted(zip(class_names, all_proba), key=lambda x: -x[1])
    rejected   = [(name, p) for name, p in all_sorted if p * 100 < 1.0][:3]
    why_not    = []
    for crop_name, prob in rejected:
        info = crop_info.get(crop_name.lower(), {})
        why_not.append({
            "name_en": crop_name.capitalize(),
            "name_hi": HINDI_NAMES.get(crop_name.lower(), crop_name),
            "emoji":   info.get("emoji", "\U0001f331"),
            "score":   round(prob * 100, 2),
            "reasons": generate_why_not(crop_name, N, P, K, temperature, humidity, ph, rainfall)
        })
    return jsonify({
        "recommendations":       recs,
        "ai_summary":            ai_summary,
        "why_not_recommended":   why_not,
        "rules_engine_active":   rules_engine_active,
        "rules_summary":         rules_summary,
        "context_used":          context,
        "generated_at":          datetime.now().isoformat(timespec='seconds'),
        "model_used":            model_name,
        "prediction_id":         pred_id,
        "total_crops_analyzed":  n_classes,
        "crops_shown":           len(recs),
        "min_score_threshold":   "1.0%"
    })


@app.route('/history', methods=['GET'])
def get_history():
    history = load_prediction_history()
    return jsonify({"total_predictions": len(history), "history": history[-10:]})


@app.route('/history/clear', methods=['DELETE'])
def clear_history():
    with open(HISTORY_PATH, 'w') as f:
        json.dump([], f)
    return jsonify({"message": "History cleared."})


# ================================================================
# SECTION 10: RUN SERVER
# ================================================================
if __name__ == '__main__':
    print("\n  Starting server on http://localhost:5000")
    print("  Press Ctrl+C to stop.\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
