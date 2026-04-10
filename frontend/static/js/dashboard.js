/* ═══════════════════════════════════════════════════════════════
   dashboard.js  ·  Nagpur AQI Dashboard (XGBoost Edition)
   ═══════════════════════════════════════════════════════════════ */

const API = "http://localhost:5000/api";

const STATION_INFO = {
  Ambazari:   { label:"Ambazari",    lat:21.1293, lng:79.0562 },
  Mahal:      { label:"Mahal",       lat:21.1502, lng:79.0849 },
  Civil_Lines:{ label:"Civil Lines", lat:21.1533, lng:79.0849 },
  Ram_Nagar:  { label:"Ram Nagar",   lat:21.1458, lng:79.1012 },
};

const AQI_BANDS = [
  [0,  50,  "Good",        "#00e400"],
  [51, 100, "Satisfactory","#d4d700"],
  [101,200, "Moderate",    "#ff7e00"],
  [201,300, "Poor",        "#ff0000"],
  [301,400, "Very Poor",   "#99004c"],
  [401,999, "Severe",      "#7e0023"],
];

function aqiInfo(aqi) {
  for (const [lo,hi,cat,color] of AQI_BANDS)
    if (aqi >= lo && aqi <= hi)
      return { category:cat, color, advisory:advText(aqi) };
  return { category:"Unknown", color:"#888", advisory:"Data unavailable" };
}

function advText(aqi) {
  if (aqi <= 50)  return "Air quality is satisfactory. Safe for all.";
  if (aqi <= 100) return "Minor discomfort for sensitive individuals.";
  if (aqi <= 200) return "Reduce prolonged outdoor exertion.";
  if (aqi <= 300) return "Health effects for all. Limit outdoor activity.";
  return "Health emergency. Stay indoors.";
}

// ── STATE ────────────────────────────────────────────────────────
let activeStation = "Ambazari";
let map, mapMarkers = {};
let histChart, fcChart, trendChart, fimpChart;
let currentData = {};

// ── INIT ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  startClock();
  buildTabs();
  initMap();
  refreshAll();
  setInterval(refreshAll, 5 * 60 * 1000);
});

function startClock() {
  const el = document.getElementById("live-clock");
  const tick = () => {
    el.textContent = new Date().toLocaleString("en-IN", {
      timeZone:"Asia/Kolkata", hour12:false,
      year:"numeric", month:"short", day:"2-digit",
      hour:"2-digit", minute:"2-digit", second:"2-digit"
    }) + " IST";
  };
  tick(); setInterval(tick, 1000);
}

function buildTabs() {
  const nav = document.getElementById("station-tabs");
  Object.entries(STATION_INFO).forEach(([key, s]) => {
    const btn = document.createElement("button");
    btn.className = "tab-btn" + (key === activeStation ? " active" : "");
    btn.textContent = s.label;
    btn.id = `tab-${key}`;
    btn.onclick = () => switchStation(key);
    nav.appendChild(btn);
  });
}

function switchStation(station) {
  activeStation = station;
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(`tab-${station}`).classList.add("active");
  updateGauge(station);
  loadPollutants(station);
  loadHistory(station);
  loadForecast(station);
  loadFeatImportance(station);
  loadHealthAdvisory();
}

// ── MAP ───────────────────────────────────────────────────────────
function initMap() {
  map = L.map("nagpur-map", {
    center:[21.145, 79.088], zoom:12,
    zoomControl:true, attributionControl:false
  });
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom:18 }).addTo(map);

  Object.entries(STATION_INFO).forEach(([key, s]) => {
    const m = L.circleMarker([s.lat, s.lng], {
      radius:13, fillColor:"#2ca02c", color:"#fff",
      weight:2, opacity:1, fillOpacity:0.85
    }).addTo(map)
      .bindPopup(`<b>${s.label}</b><br>AQI: loading…`);
    m.on("click", () => switchStation(key));
    mapMarkers[key] = m;
  });
}

function updateMapMarkers(current) {
  Object.entries(current).forEach(([station, d]) => {
    const m = mapMarkers[station];
    if (!m) return;
    m.setStyle({ fillColor: d.color });
    m.bindPopup(
      `<b>${STATION_INFO[station]?.label || station}</b><br>
       AQI: <b>${d.aqi}</b> — ${d.category}<br>
       PM2.5: ${d["PM2.5"]} µg/m³`
    );
  });
}

// ── CURRENT AQI ───────────────────────────────────────────────────
async function loadCurrent() {
  try {
    const res  = await fetch(`${API}/current`);
    const data = await res.json();
    currentData = data;

    // City average banner
    const vals = Object.values(data).map(d => d.aqi).filter(Boolean);
    if (vals.length) {
      const avg = vals.reduce((a,b)=>a+b,0)/vals.length;
      const info = aqiInfo(avg);
      document.getElementById("city-avg-aqi").textContent = avg.toFixed(0);
      document.getElementById("city-avg-aqi").style.color = info.color;
      const catEl = document.getElementById("city-avg-cat");
      catEl.textContent = info.category;
      catEl.style.background = info.color + "33";
      catEl.style.color = info.color;
      document.getElementById("city-advisory").textContent = info.advisory;
      document.getElementById("city-banner").style.borderLeftColor = info.color;
    }

    updateMapMarkers(data);
    updateGauge(activeStation);
    loadPollutants(activeStation);
  } catch(e) { console.error("current", e); }
}

// ── GAUGE ─────────────────────────────────────────────────────────
function updateGauge(station) {
  const d = currentData[station];
  if (!d) return;
  document.getElementById("gauge-val").textContent  = d.aqi;
  document.getElementById("gauge-val").style.color  = d.color;
  document.getElementById("gauge-cat").textContent  = d.category;
  document.getElementById("gauge-station").textContent = STATION_INFO[station]?.label || station;
  document.getElementById("gauge-time").textContent =
    new Date(d.datetime).toLocaleString("en-IN",{timeZone:"Asia/Kolkata"});
  drawGauge(d.aqi, d.color);
}

function drawGauge(aqi, color) {
  const canvas = document.getElementById("gaugeCanvas");
  const ctx = canvas.getContext("2d");
  const cx = 100, cy = 100, r = 78;
  ctx.clearRect(0, 0, 200, 200);

  // Background arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI*0.75, Math.PI*2.25);
  ctx.lineWidth = 14; ctx.strokeStyle = "#21262d";
  ctx.lineCap = "round"; ctx.stroke();

  // Coloured arc
  const pct = Math.min(aqi/500, 1);
  const end  = Math.PI*0.75 + pct*Math.PI*1.5;
  const grad = ctx.createLinearGradient(0,0,200,0);
  grad.addColorStop(0,   "#00e400");
  grad.addColorStop(0.3, "#d4d700");
  grad.addColorStop(0.5, "#ff7e00");
  grad.addColorStop(0.7, "#ff0000");
  grad.addColorStop(1,   "#7e0023");
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI*0.75, end);
  ctx.strokeStyle = grad; ctx.stroke();

  // Scale labels
  ["0","100","200","300","400","500"].forEach((lbl,i) => {
    const a = Math.PI*0.75 + (i/5)*Math.PI*1.5;
    ctx.beginPath();
    ctx.moveTo(cx+(r+8)*Math.cos(a),  cy+(r+8)*Math.sin(a));
    ctx.lineTo(cx+(r+14)*Math.cos(a), cy+(r+14)*Math.sin(a));
    ctx.lineWidth=1.5; ctx.strokeStyle="#555"; ctx.stroke();
    ctx.fillStyle="#888"; ctx.font="9px sans-serif";
    ctx.textAlign="center"; ctx.textBaseline="middle";
    ctx.fillText(lbl, cx+(r+24)*Math.cos(a), cy+(r+24)*Math.sin(a));
  });
}

// ── POLLUTANTS ─────────────────────────────────────────────────────
async function loadPollutants(station) {
  try {
    const res  = await fetch(`${API}/pollutants/${station}`);
    const data = await res.json();
    if (!data.pollutants) return;

    const el = document.getElementById("pollutant-bars");
    el.innerHTML = "";
    Object.entries(data.pollutants).forEach(([name, p]) => {
      const pct = Math.min(p.value/p.limit*100, 120);
      const cls = pct > 100 ? "fill-danger" : pct > 70 ? "fill-warn" : "fill-ok";
      el.innerHTML += `
        <div class="poll-item">
          <div class="poll-row">
            <span>${name}</span>
            <span class="poll-right">${p.value} ${p.unit}
              <span style="color:${pct>100?"#ff5555":"#8b949e"}">
                / ${p.limit}
              </span>
            </span>
          </div>
          <div class="poll-bg">
            <div class="poll-fill ${cls}" style="width:${Math.min(pct,100)}%"></div>
          </div>
        </div>`;
    });
  } catch(e) { console.error("pollutants", e); }
}

// ── HISTORY CHART ──────────────────────────────────────────────────
async function loadHistory(station) {
  try {
    const res  = await fetch(`${API}/history/${station}`);
    const data = await res.json();
    const rows = data.history || [];

    document.getElementById("hist-station").textContent =
      "— " + (STATION_INFO[station]?.label || station);

    const labels = rows.map(r => {
      const d = new Date(r.datetime);
      return `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,"0")}h`;
    });
    const vals   = rows.map(r => r.aqi);
    const colors = rows.map(r => r.color || "#58a6ff");

    const ctx = document.getElementById("historyChart").getContext("2d");
    if (histChart) histChart.destroy();
    histChart = new Chart(ctx, {
      type:"line",
      data:{
        labels,
        datasets:[{
          label:"AQI", data:vals,
          borderColor:"#58a6ff",
          backgroundColor:"rgba(88,166,255,.07)",
          pointBackgroundColor:colors,
          pointRadius:1, tension:0.3, fill:true, borderWidth:1.5
        }]
      },
      options: chartOpts("AQI — 7-day history")
    });
  } catch(e) { console.error("history", e); }
}

// ── FORECAST CHART ─────────────────────────────────────────────────
async function loadForecast(station) {
  try {
    const res  = await fetch(`${API}/forecast/${station}`);
    const data = await res.json();
    if (data.error) return;

    // Peak badge
    const peakEl = document.getElementById("fc-peak");
    peakEl.textContent = `Peak t+${data.peak_hour}h: AQI ${data.peak_aqi}`;
    peakEl.style.color = data.peak_info?.color || "#888";

    const rows   = data.forecast;
    const labels = rows.map(r => r.datetime.slice(11,16));
    const vals   = rows.map(r => r.aqi);
    const bgs    = rows.map(r => (r.color || "#2ca02c") + "cc");

    const ctx = document.getElementById("forecastChart").getContext("2d");
    if (fcChart) fcChart.destroy();
    fcChart = new Chart(ctx, {
      type:"bar",
      data:{
        labels,
        datasets:[{
          label:"XGBoost Forecast", data:vals,
          backgroundColor:bgs, borderColor:bgs,
          borderRadius:3, borderWidth:0
        }]
      },
      options: chartOpts("Predicted AQI — next 24 h")
    });

    // Hourly chips
    const chips = document.getElementById("fc-chips");
    chips.innerHTML = "";
    rows.forEach(r => {
      const chip = document.createElement("div");
      chip.className = "fc-chip";
      chip.style.background = (r.color || "#2ca02c") + "22";
      chip.style.borderColor = r.color || "#2ca02c";
      chip.innerHTML = `
        <span class="fc-chip-aqi" style="color:${r.color}">${r.aqi}</span>
        <span class="fc-chip-hr">${r.datetime.slice(11,16)}</span>`;
      chips.appendChild(chip);
    });
  } catch(e) { console.error("forecast", e); }
}

// ── 30-DAY TREND ───────────────────────────────────────────────────
async function loadTrend() {
  try {
    const res  = await fetch(`${API}/aqi_trend`);
    const data = await res.json();
    const labels = data.map(d => d.date);
    const vals   = data.map(d => d.aqi);
    const colors = vals.map(v => (aqiInfo(v).color || "#2ca02c") + "cc");

    const ctx = document.getElementById("trendChart").getContext("2d");
    if (trendChart) trendChart.destroy();
    trendChart = new Chart(ctx, {
      type:"bar",
      data:{
        labels,
        datasets:[{
          label:"Avg AQI", data:vals,
          backgroundColor:colors, borderRadius:3, borderWidth:0
        }]
      },
      options: chartOpts("City Average AQI — 30 days")
    });
  } catch(e) { console.error("trend", e); }
}

// ── HEALTH ADVISORY ────────────────────────────────────────────────
async function loadHealthAdvisory() {
  try {
    const res  = await fetch(`${API}/health_advisory`);
    const data = await res.json();
    const el   = document.getElementById("health-content");
    el.innerHTML = "";
    Object.entries(data).forEach(([station, d]) => {
      el.innerHTML += `
        <div class="health-st">
          <div class="health-top">
            <span class="health-name">${STATION_INFO[station]?.label || station}</span>
            <span class="health-badge" style="background:${d.color}33;color:${d.color}">
              ${d.category} · ${d.aqi}
            </span>
          </div>
          <div class="health-groups">
            ${Object.entries(d.groups).map(([g,t]) =>
              `<div class="health-grp">
                <span class="health-grp-l">${g}:</span>
                <span>${t}</span>
              </div>`
            ).join("")}
          </div>
        </div>`;
    });
  } catch(e) { console.error("health", e); }
}

// ── MODEL METRICS ──────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const res  = await fetch(`${API}/model_metrics`);
    const data = await res.json();
    const tbody = document.getElementById("metrics-tbody");
    tbody.innerHTML = "";
    Object.entries(data).forEach(([station, d]) => {
      const te = d.test || d; // support both formats
      const r2 = te.R2 || te.r2 || 0;
      const cls = r2 >= 0.9 ? "r2-great" : r2 >= 0.7 ? "r2-ok" : "r2-poor";
      tbody.innerHTML += `
        <tr>
          <td>${STATION_INFO[station]?.label || station}</td>
          <td>${te.MAE}</td>
          <td>${te.RMSE}</td>
          <td class="${cls}">${r2.toFixed(4)}</td>
          <td>${te.CatAcc ?? "–"}%</td>
        </tr>`;
    });
  } catch(e) {
    document.getElementById("metrics-tbody").innerHTML =
      `<tr><td colspan="5" style="color:var(--muted)">Run train_xgboost.py first</td></tr>`;
  }
}

// ── FEATURE IMPORTANCE ─────────────────────────────────────────────
async function loadFeatImportance(station) {
  try {
    const res  = await fetch(`${API}/feature_importance/${station}`);
    const data = await res.json();
    if (data.error || !data.importance) return;

    document.getElementById("fi-station").textContent =
      "— " + (STATION_INFO[station]?.label || station);

    const entries = Object.entries(data.importance).sort((a,b)=>a[1]-b[1]);
    const labels  = entries.map(([k]) => k);
    const vals    = entries.map(([,v]) => +v.toFixed(4));

    const ctx = document.getElementById("fimpChart").getContext("2d");
    if (fimpChart) fimpChart.destroy();
    fimpChart = new Chart(ctx, {
      type:"bar",
      data:{
        labels,
        datasets:[{
          label:"Importance (gain)",
          data:vals,
          backgroundColor: vals.map((v,i) =>
            i >= vals.length-3 ? "#2ca02c" : "#1f6feb88"),
          borderRadius:3, borderWidth:0
        }]
      },
      options:{
        indexAxis:"y",
        responsive:true,
        plugins:{ legend:{display:false}, tooltip:{
          backgroundColor:"#1c2230", titleColor:"#58a6ff",
          bodyColor:"#e6edf3", borderColor:"#30363d", borderWidth:1
        }},
        scales:{
          x:{ ticks:{color:"#8b949e"}, grid:{color:"#21262d"},
              title:{display:true,text:"Importance",color:"#8b949e",font:{size:11}} },
          y:{ ticks:{color:"#8b949e",font:{size:11}}, grid:{color:"#21262d"} }
        }
      }
    });
  } catch(e) { console.error("feature importance", e); }
}

// ── CUSTOM PREDICTOR ──────────────────────────────────────────────
async function runPredict() {
  const body = {
    station:  document.getElementById("p-station").value,
    "PM2.5":  +document.getElementById("p-pm25").value,
    "PM10":   +document.getElementById("p-pm10").value,
    "NO2":    +document.getElementById("p-no2").value,
    "SO2":    +document.getElementById("p-so2").value,
    "NH3":    +document.getElementById("p-nh3").value,
  };

  try {
    const res = await fetch(`${API}/predict`, {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify(body)
    });
    const d = await res.json();

    const el = document.getElementById("predict-result");
    el.classList.remove("hidden");
    el.style.borderColor = d.color || "#2ca02c";

    document.getElementById("pr-aqi").textContent = d.predicted_aqi;
    document.getElementById("pr-aqi").style.color = d.color || "#fff";
    document.getElementById("pr-cat").textContent = d.category;
    document.getElementById("pr-cat").style.color = d.color || "#fff";
    document.getElementById("pr-adv").textContent = d.advisory;

    // 24-h mini chips
    const chips = document.getElementById("pr-chips");
    chips.innerHTML = "";
    (d.forecast_24h || []).forEach(r => {
      const chip = document.createElement("div");
      chip.className = "fc-chip";
      chip.style.background = (r.color || "#2ca02c") + "22";
      chip.style.borderColor = r.color || "#2ca02c";
      chip.innerHTML = `
        <span class="fc-chip-aqi" style="color:${r.color}">${r.aqi}</span>
        <span class="fc-chip-hr">${r.datetime}</span>`;
      chips.appendChild(chip);
    });
  } catch(e) {
    alert("Prediction failed. Ensure the Flask backend is running on port 5000.");
  }
}

// ── MASTER REFRESH ────────────────────────────────────────────────
async function refreshAll() {
  await loadCurrent();
  loadHistory(activeStation);
  loadForecast(activeStation);
  loadTrend();
  loadHealthAdvisory();
  loadMetrics();
  loadFeatImportance(activeStation);
}

// ── CHART DEFAULTS ────────────────────────────────────────────────
function chartOpts(title) {
  return {
    responsive:true,
    plugins:{
      legend:{ display:false },
      tooltip:{
        backgroundColor:"#1c2230", titleColor:"#58a6ff",
        bodyColor:"#e6edf3", borderColor:"#30363d", borderWidth:1
      }
    },
    scales:{
      x:{
        ticks:{color:"#8b949e", maxRotation:45, font:{size:10}},
        grid:{color:"#21262d"}
      },
      y:{
        ticks:{color:"#8b949e"},
        grid:{color:"#21262d"},
        title:{display:true, text:"AQI", color:"#8b949e", font:{size:11}}
      }
    }
  };
}
