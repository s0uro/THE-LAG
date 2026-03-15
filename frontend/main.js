const API_BASE = "http://localhost:8000";

async function loadMetrics() {
  const pre = document.getElementById("metrics-json");
  pre.textContent = "Loading /metrics...";
  try {
    const resp = await fetch(API_BASE + "/metrics");
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error("HTTP " + resp.status + ": " + text);
    }
    const data = await resp.json();
    pre.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    pre.textContent = "Σφάλμα: " + err.message;
  }
}

function loadShapImage() {
  const img = document.getElementById("shap-img");
  img.src = API_BASE + "/shap-summary";
}

async function uploadAndRun() {
  const input = document.getElementById("file-input");
  const statusDiv = document.getElementById("upload-status");
  const btn = document.getElementById("upload-btn");

  if (!input.files || input.files.length === 0) {
    statusDiv.textContent = "Διάλεξε ένα αρχείο (.xlsx ή .csv) πρώτα.";
    return;
  }

  const file = input.files[0];
  const formData = new FormData();
  formData.append("file", file);

  btn.disabled = true;
  statusDiv.textContent = "Ανέβασμα αρχείου και εκτέλεση pipeline...";

  try {
    const resp = await fetch(API_BASE + "/upload-and-run", {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error("HTTP " + resp.status + ": " + text);
    }
    const data = await resp.json();
    statusDiv.textContent = data.message || "Ολοκληρώθηκε.";

    // Ανανεώνουμε metrics και SHAP εικόνα
    await loadMetrics();
    loadShapImage();
  } catch (err) {
    statusDiv.textContent = "Σφάλμα: " + err.message;
  } finally {
    btn.disabled = false;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("load-metrics-btn").addEventListener("click", loadMetrics);
  document.getElementById("upload-btn").addEventListener("click", uploadAndRun);
  loadShapImage();
});

