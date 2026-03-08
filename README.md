# 🧠 NeuroScan AI — Brain Tumor Detection & Hospital Recommendation

A full-stack web application that uses YOLOv8 to detect brain tumors in MRI scans and recommends nearby hospitals in Tamil Nadu.

---

## 📁 Project Structure

```
brain_tumor_app/
├── backend/
│   ├── app.py              # FastAPI main application
│   ├── model_loader.py     # YOLOv8 model inference
│   └── hospital_service.py # Hospital recommendation (Haversine)
├── frontend/
│   ├── index.html          # Main UI
│   ├── style.css           # Dark medical theme
│   └── script.js           # Frontend logic
├── data/
│   ├── hospitals.csv       # Hospital database
│   └── TamilNadu.geojson   # Tamil Nadu district map
├── static/
│   ├── uploads/            # Temp uploaded images
│   └── results/            # Annotated output images
├── best.pt                 # ← Place your YOLOv8 model here
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.9 or higher
- pip package manager

### 2. Clone / Download the project
```bash
cd brain_tumor_app
```

### 3. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Place your model file
Copy your trained `best.pt` YOLOv8 model to the project root:
```
brain_tumor_app/
└── best.pt   ← HERE
```

> **Note:** If `best.pt` is not present, the app runs in **DEMO MODE** with mock predictions so you can still test the UI.

---

## 🚀 Run the Application

### Start the backend server
```bash
cd backend
python app.py
```

Or with uvicorn directly:
```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: **http://localhost:8000**

### Open the frontend
Open `frontend/index.html` in your browser, OR navigate to:
```
http://localhost:8000
```
(The FastAPI server serves the frontend automatically.)

---

## 🎯 How to Use

1. **Upload MRI** — Drag and drop or browse to select an MRI scan image (JPEG/PNG)
2. **Analyze** — Click "Analyze MRI Scan" to run YOLOv8 detection
3. **View Result** — See detected class, confidence score, and annotated image
4. **Enter City** — Type your Tamil Nadu city name (with autocomplete)
5. **Find Hospitals** — Get top 5 nearest neurosurgery hospitals
6. **View Map** — See hospitals on interactive Tamil Nadu map

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend HTML |
| `POST` | `/predict` | Upload MRI & run tumor detection |
| `POST` | `/hospitals` | Get hospital recommendations by city |
| `GET` | `/geojson` | Serve Tamil Nadu GeoJSON |
| `GET` | `/cities` | List all available cities |
| `GET` | `/health` | Health check |

---

## 🧬 Model Details

- **Architecture**: YOLOv8 (ultralytics)
- **Classes**: `glioma`, `meningioma`, `pituitary`, `notumor`
- **Input**: MRI scan images (JPEG/PNG)
- **Output**: Class label + confidence + annotated bounding box image

---

## 🏥 Hospital Logic

- **Exact match**: Filters hospitals by the entered city name
- **Partial match**: Case-insensitive partial string search
- **Haversine fallback**: Calculates distance between user's city coordinates and all hospitals, returns 5 nearest

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `best.pt not found` | App runs in demo mode; place model in project root |
| `CORS error` | Ensure backend is running on port 8000 |
| `ultralytics not found` | Run `pip install ultralytics` |
| `Map not loading` | Check internet connection (Leaflet CDN) |
| Port already in use | Change port: `uvicorn backend.app:app --port 8001` |

---

## 📋 Requirements

```
fastapi
uvicorn
python-multipart
numpy
pandas
opencv-python-headless
Pillow
ultralytics
```
"# webapp" 
