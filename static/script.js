/**
 * NeuroScan AI — Brain Tumor Detection Frontend
 * Handles: upload, prediction, hospital search, map rendering
 */

const API = "http://localhost:8000";

// ── State ──────────────────────────────────────────────────
let selectedFile = null;
let predictionResult = null;
let mapInstance = null;

// ── DOM Refs ───────────────────────────────────────────────
const dropZone      = document.getElementById("dropZone");
const fileInput     = document.getElementById("fileInput");
const previewArea   = document.getElementById("previewArea");
const previewImg    = document.getElementById("previewImg");
const previewInfo   = document.getElementById("previewInfo");
const removeBtn     = document.getElementById("removeBtn");
const analyzeBtn    = document.getElementById("analyzeBtn");
const cityInput     = document.getElementById("cityInput");
const cityList      = document.getElementById("cityList");
const quickTags     = document.getElementById("quickTags");
const findBtn       = document.getElementById("findHospitalsBtn");

// ── Init ───────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  setupDragDrop();
  setupFileInput();
  setupRemoveBtn();
  setupAnalyzeBtn();
  setupCityInput();
  loadCities();
});

// ── Drag & Drop ────────────────────────────────────────────
function setupDragDrop() {
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file);
  });
  dropZone.addEventListener("click", (e) => {
    if (e.target.tagName !== "LABEL") fileInput.click();
  });
}

function setupFileInput() {
  fileInput.addEventListener("change", () => {
    if (fileInput.files?.[0]) handleFile(fileInput.files[0]);
  });
}

function handleFile(file) {
  const allowedTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
  if (!allowedTypes.includes(file.type)) {
    showToast("Please upload a JPEG or PNG image.", "error");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast("File too large. Max 10MB.", "error");
    return;
  }

  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewArea.style.display = "flex";
    dropZone.style.display = "none";
    previewInfo.innerHTML = `
      <strong style="color:var(--text)">${file.name}</strong><br/>
      Size: ${(file.size / 1024).toFixed(1)} KB<br/>
      Type: ${file.type}
    `;
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function setupRemoveBtn() {
  removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedFile = null;
    previewImg.src = "";
    previewArea.style.display = "none";
    dropZone.style.display = "block";
    analyzeBtn.disabled = true;
    fileInput.value = "";
  });
}

// ── Analyze ────────────────────────────────────────────────
function setupAnalyzeBtn() {
  analyzeBtn.addEventListener("click", runAnalysis);
}

async function runAnalysis() {
  if (!selectedFile) return;

  setLoadingState(analyzeBtn, true);

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(`${API}/predict`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Analysis failed");
    }

    const data = await response.json();
    predictionResult = data;
    showPredictionResult(data);

  } catch (err) {
    console.error(err);
    showToast(`Error: ${err.message}`, "error");
  } finally {
    setLoadingState(analyzeBtn, false);
  }
}

function showPredictionResult(data) {
  // Update step
  setStep(2);

  // Show result section
  const section = document.getElementById("section-result");
  section.classList.remove("hidden");
  section.scrollIntoView({ behavior: "smooth", block: "start" });

  // Result image
  const resultImg = document.getElementById("resultImg");
  if (data.image) {
    resultImg.src = `data:image/jpeg;base64,${data.image}`;
  }

  // Class name
  const resultClass = document.getElementById("resultClass");
  resultClass.textContent = formatClassName(data.class);

  // Style class card by type
  const classCard = document.getElementById("resultClassCard");
  const colorMap = {
    glioma: "var(--red)",
    meningioma: "var(--orange)",
    pituitary: "var(--yellow)",
    notumor: "var(--green)"
  };
  const borderColor = colorMap[data.class] || "var(--cyan)";
  classCard.style.borderColor = borderColor;
  resultClass.style.color = borderColor;

  // Confidence bar
  const pct = data.confidence;
  setTimeout(() => {
    document.getElementById("confidenceBar").style.width = `${pct}%`;
    document.getElementById("confidencePct").textContent = `${pct}%`;
  }, 200);

  // Info box
  const infoMap = {
    glioma: { desc: "Glioma originates from glial cells in the brain or spine. Immediate specialist consultation is critical.", severity: "High", cls: "severity-high" },
    meningioma: { desc: "Meningioma grows from the protective membranes surrounding the brain and spinal cord. Often slow-growing.", severity: "Medium", cls: "severity-medium" },
    pituitary: { desc: "Pituitary tumor forms in the pituitary gland at the base of the brain. Treatment options are generally effective.", severity: "Medium", cls: "severity-medium" },
    notumor: { desc: "No tumor structures detected in this MRI scan. Continue regular health monitoring as advised by your physician.", severity: "None", cls: "severity-none" }
  };
  const info = infoMap[data.class] || infoMap.notumor;
  document.getElementById("resultInfoBox").innerHTML = `
    <span class="severity-badge ${info.cls}">Severity: ${info.severity}</span>
    <p>${info.desc}</p>
  `;

  // Demo note
  const demoNote = document.getElementById("demoNote");
  if (data.message && data.message.includes("DEMO")) {
    demoNote.textContent = data.message;
    demoNote.style.display = "block";
  } else if (data.message) {
    demoNote.textContent = data.message;
    demoNote.style.display = "block";
    demoNote.style.color = "var(--text3)";
    demoNote.style.background = "rgba(255,255,255,0.03)";
    demoNote.style.borderColor = "var(--border)";
  }

  // Show city input section
  setTimeout(() => {
    const citySection = document.getElementById("section-city");
    citySection.classList.remove("hidden");
    setStep(3);
  }, 600);
}

function formatClassName(cls) {
  const nameMap = {
    glioma: "Glioma",
    meningioma: "Meningioma",
    pituitary: "Pituitary Tumor",
    notumor: "No Tumor"
  };
  return nameMap[cls] || cls.charAt(0).toUpperCase() + cls.slice(1);
}

// ── Cities ─────────────────────────────────────────────────
async function loadCities() {
  try {
    const res = await fetch(`${API}/cities`);
    const data = await res.json();
    const cities = data.cities || [];

    // Populate datalist
    cities.forEach(c => {
      const opt = document.createElement("option");
      opt.value = c;
      cityList.appendChild(opt);
    });

    // Quick tags (first 8 cities)
    const popular = ["Chennai", "Coimbatore", "Madurai", "Trichy", "Salem", "Vellore", "Erode", "Thanjavur"];
    const tagsToShow = popular.filter(c => cities.includes(c)).slice(0, 8);
    tagsToShow.forEach(city => {
      const tag = document.createElement("button");
      tag.className = "quick-tag";
      tag.textContent = city;
      tag.addEventListener("click", () => {
        cityInput.value = city;
        cityInput.focus();
      });
      quickTags.appendChild(tag);
    });
  } catch (err) {
    console.warn("Could not load cities:", err);
  }
}

// ── Hospital Search ────────────────────────────────────────
function setupCityInput() {
  findBtn.addEventListener("click", findHospitals);
  cityInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") findHospitals();
  });
}

async function findHospitals() {
  const city = cityInput.value.trim();
  if (!city) {
    showToast("Please enter a city name.", "error");
    cityInput.focus();
    return;
  }

  setLoadingState(findBtn, true, "Searching...");

  try {
    const formData = new FormData();
    formData.append("city", city);

    const response = await fetch(`${API}/hospitals`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Search failed");
    }

    const data = await response.json();
    showHospitals(data);
    showMap(data);

  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
  } finally {
    setLoadingState(findBtn, false);
  }
}

function showHospitals(data) {
  setStep(4);

  const section = document.getElementById("section-hospitals");
  section.classList.remove("hidden");

  // Subtitle
  document.getElementById("hospitalSubtitle").textContent =
    `${data.hospitals.length} neurosurgery centers found`;

  // Match banner
  const banner = document.getElementById("matchBanner");
  if (!data.exact_match) {
    banner.innerHTML = `⚠️ ${data.message}`;
    banner.style.display = "flex";
  } else {
    banner.style.display = "none";
  }

  // Hospital cards
  const list = document.getElementById("hospitalList");
  list.innerHTML = "";

  if (!data.hospitals.length) {
    list.innerHTML = `<p style="color:var(--text3);text-align:center;padding:32px">No hospitals found for "${data.city}".</p>`;
    return;
  }

  data.hospitals.forEach((h, i) => {
    const card = document.createElement("div");
    card.className = "hospital-card";
    card.innerHTML = `
      <div class="hospital-rank">#${i + 1}</div>
      <div class="hospital-body">
        <div class="hospital-name">${h.name}</div>
        <div class="hospital-meta">
          <span>📍 ${h.city}</span>
          <span>📞 <span class="hospital-contact">${h.contact}</span></span>
        </div>
        <div class="hospital-spec">${h.specialization}</div>
      </div>
      ${h.distance_km != null ? `
      <div class="hospital-dist">
        <div class="dist-value">${h.distance_km} km</div>
        <div class="dist-label">distance</div>
      </div>` : ""}
    `;
    list.appendChild(card);
  });

  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Map ────────────────────────────────────────────────────
async function showMap(data) {
  const section = document.getElementById("section-map");
  section.classList.remove("hidden");

  // Short delay then scroll to map
  setTimeout(() => section.scrollIntoView({ behavior: "smooth" }), 400);

  // Init map only once
  if (!mapInstance) {
    mapInstance = L.map("map", {
      center: [10.8505, 76.2711],
      zoom: 7,
      zoomControl: true
    });

    // OpenStreetMap tiles (dark-ish)
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap contributors",
      maxZoom: 18
    }).addTo(mapInstance);

    // Load TamilNadu GeoJSON overlay
    try {
      const res = await fetch(`${API}/geojson`);
      if (res.ok) {
        const geojson = await res.json();
        L.geoJSON(geojson, {
          style: {
            color: "#00D4FF",
            weight: 1.5,
            fillColor: "#0A1A2E",
            fillOpacity: 0.2,
            opacity: 0.5
          }
        }).addTo(mapInstance);
      }
    } catch (e) {
      console.warn("Could not load GeoJSON:", e);
    }
  } else {
    // Clear previous markers
    mapInstance.eachLayer((layer) => {
      if (layer instanceof L.Marker) mapInstance.removeLayer(layer);
    });
  }

  const bounds = [];

  // User location marker
  if (data.user_coords) {
    const [lat, lon] = data.user_coords;
    const userIcon = L.divIcon({
      className: "",
      html: `<div class="custom-marker-user" title="${data.city}">📍</div>`,
      iconSize: [36, 36],
      iconAnchor: [18, 18]
    });
    L.marker([lat, lon], { icon: userIcon })
      .addTo(mapInstance)
      .bindPopup(`<b style="color:#00D4FF">Your Location</b><br/>${data.city}`)
      .openPopup();
    bounds.push([lat, lon]);
  }

  // Hospital markers
  data.hospitals.forEach((h, i) => {
    const hospIcon = L.divIcon({
      className: "",
      html: `<div class="custom-marker-hospital" title="${h.name}">✚</div>`,
      iconSize: [36, 36],
      iconAnchor: [18, 18]
    });
    L.marker([h.latitude, h.longitude], { icon: hospIcon })
      .addTo(mapInstance)
      .bindPopup(`
        <div style="font-family:'Space Grotesk',sans-serif;min-width:180px;">
          <b style="color:#FF9800">#${i + 1} ${h.name}</b><br/>
          <span style="color:#8AB4CC;font-size:0.82rem">📍 ${h.city}</span><br/>
          <span style="color:#8AB4CC;font-size:0.82rem">📞 ${h.contact}</span>
          ${h.distance_km != null ? `<br/><span style="color:#00E676;font-size:0.82rem">📏 ${h.distance_km} km away</span>` : ""}
        </div>
      `);
    bounds.push([h.latitude, h.longitude]);
  });

  // Fit bounds
  if (bounds.length > 0) {
    mapInstance.fitBounds(bounds, { padding: [60, 60] });
  }
}

// ── Helpers ────────────────────────────────────────────────
function setLoadingState(btn, loading, loadingText = "Processing...") {
  const textEl = btn.querySelector(".btn-text") || btn.querySelector("span");
  if (loading) {
    btn.disabled = true;
    if (textEl) textEl.textContent = loadingText;
    // Add spinner if not present
    let loader = btn.querySelector(".btn-loader");
    if (!loader) {
      btn.innerHTML = `
        <svg class="spin" viewBox="0 0 24 24" width="18" height="18">
          <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" fill="none" stroke-dasharray="60" stroke-dashoffset="20"/>
        </svg>
        ${loadingText}
      `;
    }
  } else {
    btn.disabled = false;
    if (btn.id === "analyzeBtn") {
      btn.innerHTML = `<span class="btn-text">Analyze MRI Scan</span>`;
    } else if (btn.id === "findHospitalsBtn") {
      btn.innerHTML = `<span>Find Hospitals</span>`;
    }
  }
}

function setStep(n) {
  document.querySelectorAll(".step").forEach((el, i) => {
    el.classList.remove("active", "done");
    if (i + 1 < n) el.classList.add("done");
    if (i + 1 === n) el.classList.add("active");
  });
}

let toastTimer;
function showToast(msg, type = "info") {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.className = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove("show");
  }, 3500);
}
