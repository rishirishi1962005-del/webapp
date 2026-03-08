"""
hospital_service.py
Loads hospital CSV and recommends nearest hospitals using Haversine formula.
"""
import math
import logging
import pandas as pd
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth (km).
    Uses the Haversine formula.
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# Approximate city coordinates for Tamil Nadu cities (fallback lookup)
CITY_COORDS = {
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Trichy": (10.7905, 78.7047),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Salem": (11.6643, 78.1460),
    "Tirunelveli": (8.7139, 77.7567),
    "Erode": (11.3410, 77.7172),
    "Tiruppur": (11.1085, 77.3411),
    "Vellore": (12.9165, 79.1325),
    "Thoothukudi": (8.7642, 78.1348),
    "Dindigul": (10.3624, 77.9695),
    "Thanjavur": (10.7870, 79.1378),
    "Ranipet": (12.9283, 79.3331),
    "Sivakasi": (9.4533, 77.7959),
    "Karur": (10.9601, 78.0766),
    "Namakkal": (11.2189, 78.1676),
    "Kanchipuram": (12.8185, 79.6947),
    "Kumbakonam": (10.9617, 79.3788),
    "Cuddalore": (11.7480, 79.7714),
    "Villupuram": (11.9401, 79.4861),
    "Nagapattinam": (10.7672, 79.8449),
    "Krishnagiri": (12.5266, 78.2138),
    "Dharmapuri": (12.1357, 78.1602),
    "Perambalur": (11.2347, 78.8803),
    "Ariyalur": (11.1407, 79.0762),
    "Sivaganga": (9.8479, 78.4800),
    "Pudukkottai": (10.3797, 78.8231),
    "Ramanathapuram": (9.3762, 78.8308),
    "Virudhunagar": (9.5851, 77.9624),
    "Tenkasi": (8.9593, 77.3152),
    "Kanyakumari": (8.0883, 77.5385),
    "Kallakurichi": (11.7379, 78.9595),
    "Tirupattur": (12.4958, 78.5741),
    "Chengalpattu": (12.6921, 79.9762),
    "Tiruvarur": (10.7728, 79.6339),
    "Mayiladuthurai": (11.1035, 79.6520),
}


class HospitalService:
    """Manages hospital data and provides city-based recommendations."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.DataFrame()
        self._load_data()

    def _load_data(self):
        """Load hospital data. Handles .csv and .xls files (including XLS saved as CSV)."""
        try:
            # Try reading as plain CSV first (works for .xls files saved as CSV text)
            try:
                self.df = pd.read_csv(self.csv_path, header=0)
            except Exception:
                # Fallback: try with xlrd engine for true XLS binary files
                self.df = pd.read_excel(self.csv_path, engine="xlrd", header=0)

            logger.info(f"Raw columns before normalize: {self.df.columns.tolist()}")

            # Normalize column names: lowercase + strip whitespace
            self.df.columns = [str(c).lower().strip() for c in self.df.columns]

            # Verify required columns exist
            required = {"hospital_name", "city", "latitude", "longitude", "contact"}
            missing = required - set(self.df.columns)
            if missing:
                logger.error(f"Missing columns: {missing}. Available: {self.df.columns.tolist()}")
                # Attempt positional fallback if columns look numeric
                if all(str(c).isdigit() or c == 0 for c in self.df.columns):
                    self.df.columns = ["hospital_name", "city", "specialization",
                                       "latitude", "longitude", "contact"][:len(self.df.columns)]
                    logger.info("Applied positional column names as fallback")

            # Normalize city names
            if "city" in self.df.columns:
                self.df["city"] = self.df["city"].astype(str).str.strip()

            # Ensure specialization column exists
            if "specialization" not in self.df.columns:
                self.df["specialization"] = "Neurosurgery & Brain Tumor Care"

            # Convert coords to numeric, drop invalid rows
            self.df["latitude"] = pd.to_numeric(self.df["latitude"], errors="coerce")
            self.df["longitude"] = pd.to_numeric(self.df["longitude"], errors="coerce")
            self.df = self.df.dropna(subset=["latitude", "longitude"])

            logger.info(f"✅ Loaded {len(self.df)} hospitals. Columns: {self.df.columns.tolist()}")

        except Exception as e:
            logger.error(f"❌ Failed to load hospital data: {e}")
            self.df = pd.DataFrame(columns=["hospital_name", "city", "specialization",
                                             "latitude", "longitude", "contact"])

    def get_hospitals(self, city: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Find top N hospitals for a given city.
        - Exact city match first
        - Fuzzy match (case-insensitive, partial) second
        - Haversine nearest fallback third
        """
        if self.df.empty:
            return {
                "matched_city": city,
                "exact_match": False,
                "hospitals": [],
                "message": "Hospital database unavailable.",
                "user_coords": None
            }

        city_lower = city.lower().strip()

        # 1. Exact match (case-insensitive)
        mask_exact = self.df["city"].str.lower() == city_lower
        matched = self.df[mask_exact]

        if not matched.empty:
            return self._format_result(matched.head(top_n), city, city, True,
                                       "Showing hospitals in your city.", None)

        # 2. Partial match
        mask_partial = self.df["city"].str.lower().str.contains(city_lower, na=False)
        matched = self.df[mask_partial]
        if not matched.empty:
            matched_city = matched["city"].iloc[0]
            return self._format_result(matched.head(top_n), city, matched_city, False,
                                       f"Exact city not found. Showing results for '{matched_city}'.", None)

        # 3. Haversine nearest hospitals
        user_lat, user_lon = self._lookup_city_coords(city)
        if user_lat is None:
            # Use centroid of Tamil Nadu as fallback
            user_lat, user_lon = 10.7905, 78.7047

        df_copy = self.df.copy()
        df_copy["distance_km"] = df_copy.apply(
            lambda r: haversine(user_lat, user_lon, r["latitude"], r["longitude"]), axis=1
        )
        nearest = df_copy.nsmallest(top_n, "distance_km")

        return self._format_result(
            nearest, city,
            f"Nearest to {city}",
            False,
            f"City '{city}' not found in database. Showing {top_n} nearest hospitals.",
            (user_lat, user_lon)
        )

    def _lookup_city_coords(self, city: str):
        """Look up approximate city coordinates."""
        city_title = city.strip().title()
        if city_title in CITY_COORDS:
            return CITY_COORDS[city_title]
        # Try partial match in CITY_COORDS
        for k, v in CITY_COORDS.items():
            if city.lower() in k.lower() or k.lower() in city.lower():
                return v
        return None, None

    def _format_result(self, df_slice, input_city: str, matched_city: str,
                       exact: bool, message: str,
                       user_coords: Optional[tuple]) -> Dict[str, Any]:
        """Convert DataFrame slice to API-friendly hospital list."""
        hospitals = []
        for _, row in df_slice.iterrows():
            hospitals.append({
                "name": str(row.get("hospital_name", "Unknown Hospital")),
                "city": str(row.get("city", "")),
                "specialization": str(row.get("specialization", "")),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "contact": str(row.get("contact", "")),
                "distance_km": round(float(row["distance_km"]), 1) if "distance_km" in row else None
            })

        return {
            "matched_city": matched_city,
            "exact_match": exact,
            "hospitals": hospitals,
            "message": message,
            "user_coords": list(user_coords) if user_coords else None
        }
