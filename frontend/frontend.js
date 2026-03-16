const API_BASE = (typeof window.API_BASE_OVERRIDE !== "undefined") ? window.API_BASE_OVERRIDE : "http://localhost:8000";
const USER_STORAGE_KEY = "thelag_user";

function getStoredUser() {
  try {
    const raw = localStorage.getItem(USER_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function setStoredUser(user) {
  localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(user));
}

function clearStoredUser() {
  localStorage.removeItem(USER_STORAGE_KEY);
}

function updateHeaderForUser() {
  const authButtons = document.getElementById("auth-buttons");
  const headerLogoutWrap = document.getElementById("header-logout-wrap");
  const headerWelcomeName = document.getElementById("header-welcome-name");
  const user = getStoredUser();

  // Desktop
  if (user && user.username) {
    if (authButtons) authButtons.style.display = "none";
    if (headerLogoutWrap) headerLogoutWrap.style.display = "flex";
    if (headerWelcomeName) headerWelcomeName.textContent = user.firstName || user.username;
  } else {
    if (authButtons) authButtons.style.display = "flex";
    if (headerLogoutWrap) headerLogoutWrap.style.display = "none";
    if (headerWelcomeName) headerWelcomeName.textContent = "";
  }

  // Mobile drawer
  const mobileAuthRow   = document.getElementById("mobile-auth-row");
  const mobileLogoutRow = document.getElementById("mobile-logout-row");
  const mobileWelcome   = document.getElementById("mobile-welcome-name");
  if (user && user.username) {
    if (mobileAuthRow)   mobileAuthRow.style.display   = "none";
    if (mobileLogoutRow) mobileLogoutRow.style.display  = "flex";
    if (mobileWelcome)   mobileWelcome.textContent       = user.firstName || user.username;
  } else {
    if (mobileAuthRow)   mobileAuthRow.style.display   = "flex";
    if (mobileLogoutRow) mobileLogoutRow.style.display  = "none";
    if (mobileWelcome)   mobileWelcome.textContent       = "";
  }
}

async function loadMetrics() {
  const btn = document.getElementById("load-metrics-btn");
  if (btn) btn.disabled = true;
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val ?? "—"; };
  const fmt = v => v != null ? (typeof v === "number" ? (v * 100).toFixed(1) + "%" : String(v)) : "—";
  try {
    const resp = await fetch(API_BASE + "/metrics");
    if (!resp.ok) throw new Error("HTTP " + resp.status + " — run the pipeline first");
    const data = await resp.json();
    // API returns "xgboost" key (not "xgb")
    set("xgb-acc", fmt(data.xgboost?.accuracy));
    set("xgb-f1",  fmt(data.xgboost?.f1));
    set("xgb-r2",  fmt(data.xgboost?.report?.["weighted avg"]?.precision));
    set("mlp-acc", fmt(data.mlp?.accuracy));
    set("mlp-f1",  fmt(data.mlp?.f1));
    set("mlp-r2",  fmt(data.mlp?.report?.["weighted avg"]?.precision));
    const statusDiv = document.getElementById("upload-status");
    if (statusDiv && !statusDiv.textContent.startsWith("✓")) {
      statusDiv.textContent = "✓ Metrics loaded";
      statusDiv.style.color = "var(--gold)";
    }
  } catch (err) {
    ["xgb-acc","xgb-f1","xgb-r2","mlp-acc","mlp-f1","mlp-r2"].forEach(id => set(id, "—"));
    const statusDiv = document.getElementById("upload-status");
    if (statusDiv) {
      statusDiv.textContent = "❌ " + err.message;
      statusDiv.style.color = "#e07060";
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

function clearResults() {
  // Metrics
  ["xgb-acc", "xgb-f1", "xgb-r2", "mlp-acc", "mlp-f1", "mlp-r2"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = "—";
  });

  // Upload status + pipeline steps UI
  const statusDiv = document.getElementById("upload-status");
  if (statusDiv) {
    statusDiv.textContent = "";
    statusDiv.style.color = "";
  }
  const pwrap = document.getElementById("pipeline-steps-status");
  if (pwrap) pwrap.style.display = "none";

  // SHAP image + placeholder
  const img = document.getElementById("shap-img");
  if (img) {
    img.removeAttribute("src");
    img.style.display = "none";
    const frame = img.closest(".shap-frame");
    frame?.querySelector(".shap-placeholder")?.remove();
    if (frame) {
      const p = document.createElement("p");
      p.className = "shap-placeholder";
      p.style.cssText =
        "text-align:center;padding:2rem;font-family:'DM Mono',monospace;font-size:.7rem;color:var(--muted)";
      p.textContent = "Run the pipeline to generate the SHAP plot";
      frame.appendChild(p);
    }
  }

  // Clear selected file UI (dashboard)
  const mainFileInput = document.getElementById("file-input");
  if (mainFileInput) mainFileInput.value = "";
  const selEl = document.getElementById("selected-filename");
  if (selEl) selEl.textContent = "";
  const uploadZoneWrap = document.getElementById("upload-zone-wrap");
  const uploadedFileDisplay = document.getElementById("uploaded-file-display");
  const uploadedFileName = document.getElementById("uploaded-file-name");
  uploadZoneWrap?.classList.remove("hidden");
  uploadedFileDisplay?.classList.remove("visible");
  if (uploadedFileName) uploadedFileName.textContent = "";
}

function loadShapImage() {
  const img = document.getElementById("shap-img");
  if (!img) return;
  // Cache-bust so the new image always loads after pipeline
  img.src = API_BASE + "/shap-summary?t=" + Date.now();
  img.onerror = () => {
    img.style.display = "none";
    const frame = img.closest(".shap-frame");
    if (frame && !frame.querySelector(".shap-placeholder")) {
      const p = document.createElement("p");
      p.className = "shap-placeholder";
      p.style.cssText = "text-align:center;padding:2rem;font-family:'DM Mono',monospace;font-size:.7rem;color:var(--muted)";
      p.textContent = "Run the pipeline to generate the SHAP plot";
      frame.appendChild(p);
    }
  };
  img.onload = () => {
    img.style.display = "block";
    const placeholder = img.closest(".shap-frame")?.querySelector(".shap-placeholder");
    if (placeholder) placeholder.remove();
  };
}

async function uploadAndRun(optionalFileInput) {
  const input = optionalFileInput || document.getElementById("file-input");
  const statusDiv = document.getElementById("upload-status");
  const btn = optionalFileInput ? document.getElementById("hero-generate-btn") : document.getElementById("upload-btn");

  if (!input || !input.files || input.files.length === 0) {
    if (statusDiv) statusDiv.textContent = "Διάλεξε ένα αρχείο (.xlsx ή .csv) πρώτα.";
    return;
  }

  const file = input.files[0];
  const formData = new FormData();
  formData.append("file", file);

  if (btn) btn.disabled = true;
  if (statusDiv) {
    statusDiv.textContent = "⏳ Uploading " + file.name + "…";
    statusDiv.style.color = "var(--gold)";
  }


  // Animate pipeline steps
  const stepsWrap = document.getElementById("pipeline-steps-status");
  const stepEls = [
    document.getElementById("pstep-1"),
    document.getElementById("pstep-2"),
    document.getElementById("pstep-3"),
    document.getElementById("pstep-4"),
  ];
  const stepLabels = ["Preprocessing", "Training", "Evaluation", "SHAP Analysis"];
  let stepTimer = null;
  function startPipelineSteps() {
    if (!stepsWrap) return;
    stepsWrap.style.display = "block";
    stepEls.forEach((el, i) => {
      if (!el) return;
      el.textContent = (i === 0 ? "⏳ " : "○ ") + stepLabels[i];
      el.style.opacity = i === 0 ? "1" : "0.35";
    });
    let step = 0;
    // Approximate timing: preprocessing ~5s, training ~30s, evaluation ~5s, shap ~10s
    const delays = [0, 6000, 40000, 46000];
    delays.forEach((delay, i) => {
      setTimeout(() => {
        if (stepEls[i]) {
          stepEls[i].textContent = "⏳ " + stepLabels[i] + "...";
          stepEls[i].style.opacity = "1";
        }
        if (i > 0 && stepEls[i-1]) {
          stepEls[i-1].textContent = "✓ " + stepLabels[i-1];
          stepEls[i-1].style.opacity = "0.6";
        }
      }, delay);
    });
  }
  function endPipelineSteps(success) {
    if (!stepsWrap) return;
    if (success) {
      stepEls.forEach((el, i) => {
        if (el) { el.textContent = "✓ " + stepLabels[i]; el.style.opacity = "0.6"; }
      });
      setTimeout(() => { if (stepsWrap) stepsWrap.style.display = "none"; }, 2000);
    } else {
      if (stepsWrap) stepsWrap.style.display = "none";
    }
  }
  startPipelineSteps();

  // Save scroll position so page doesn't jump during the async pipeline
  const savedScrollY = window.scrollY;

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
    if (statusDiv) {
      statusDiv.textContent = "✓ Pipeline complete — " + file.name;
      statusDiv.style.color = "var(--gold)";
    }
    endPipelineSteps(true);
    await loadMetrics();
    loadShapImage();

    // Always scroll to dashboard section to show results
    const dashboardSection = document.getElementById("dashboard");
    if (dashboardSection) {
      dashboardSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (err) {
    endPipelineSteps(false);
    if (statusDiv) {
      statusDiv.textContent = "❌ " + err.message;
      statusDiv.style.color = "#e07060";
    }
  } finally {
    if (btn) btn.disabled = false;
    if (optionalFileInput && input) {
      input.value = "";
      const heroNameEl = document.getElementById("hero-selected-filename");
      if (heroNameEl) heroNameEl.textContent = "";
      const heroGenBtn = document.getElementById("hero-generate-btn");
      if (heroGenBtn) heroGenBtn.disabled = true;
    }
  }
}

function wordFlick() {
  var words = [
    "Predicting reading cognitive load by analyzing eye-tracking data through a comparative study of XGBoost and Multi-Layer Perceptron."
  ];
  var part, i = 0, offset = 0;
  var speed = 35;
  var el = document.querySelector(".word");
  if (!el) return;

  el.textContent = "";
  var id = setInterval(function () {
    if (offset > words[i].length) {
      clearInterval(id); // one text only (no loop / no delete)
      return;
    }
    part = words[i].substr(0, offset);
    el.textContent = part;
    offset++;
  }, speed);
}

function openSignupModal() {
  const modal = document.getElementById("signup-modal");
  if (modal) {
    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  }
}

function closeSignupModal() {
  const modal = document.getElementById("signup-modal");
  if (modal) {
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  }
}

function openLoginModal() {
  const modal = document.getElementById("login-modal");
  if (modal) {
    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  }
}

function closeLoginModal() {
  const modal = document.getElementById("login-modal");
  if (modal) {
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  }
}

window.addEventListener("DOMContentLoaded", () => {
  updateHeaderForUser();

  document.getElementById("load-metrics-btn")?.addEventListener("click", loadMetrics);
  document.getElementById("clear-results-btn")?.addEventListener("click", (e) => {
    e.preventDefault();
    clearResults();
  });
  document.getElementById("upload-btn")?.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();
    uploadAndRun();
  });
  const heroFileInput = document.getElementById("hero-file-input");
  const heroGenerateBtn = document.getElementById("hero-generate-btn");
  if (heroFileInput) {
    heroFileInput.addEventListener("change", function () {
      const heroNameEl = document.getElementById("hero-selected-filename");
      if (this.files && this.files[0] && heroNameEl) {
        heroNameEl.textContent = this.files[0].name;
        if (heroGenerateBtn) heroGenerateBtn.disabled = false;
      } else {
        if (heroNameEl) heroNameEl.textContent = "";
        if (heroGenerateBtn) heroGenerateBtn.disabled = true;
      }
    });
  }
  if (heroGenerateBtn) {
    heroGenerateBtn.addEventListener("click", function () {
      if (heroFileInput && heroFileInput.files && heroFileInput.files.length > 0) {
        uploadAndRun(heroFileInput);
      }
    });
  }
  const mainFileInput = document.getElementById("file-input");
  const uploadZoneWrap = document.getElementById("upload-zone-wrap");
  const uploadedFileDisplay = document.getElementById("uploaded-file-display");
  const uploadedFileName = document.getElementById("uploaded-file-name");
  const uploadedFileDeleteBtn = document.getElementById("uploaded-file-delete");
  if (mainFileInput) {
    mainFileInput.addEventListener("change", function () {
      const selEl = document.getElementById("selected-filename");
      const hasFile = this.files && this.files[0];
      if (selEl) selEl.textContent = hasFile ? this.files[0].name : "";
      if (uploadZoneWrap && uploadedFileDisplay && uploadedFileName) {
        if (hasFile) {
          uploadedFileName.textContent = this.files[0].name;
          uploadZoneWrap.classList.add("hidden");
          uploadedFileDisplay.classList.add("visible");
        } else {
          uploadZoneWrap.classList.remove("hidden");
          uploadedFileDisplay.classList.remove("visible");
          uploadedFileName.textContent = "";
        }
      }
    });
  }
  if (uploadedFileDeleteBtn && mainFileInput && uploadZoneWrap && uploadedFileDisplay && uploadedFileName) {
    uploadedFileDeleteBtn.addEventListener("click", function () {
      mainFileInput.value = "";
      const selEl = document.getElementById("selected-filename");
      if (selEl) selEl.textContent = "";
      uploadZoneWrap.classList.remove("hidden");
      uploadedFileDisplay.classList.remove("visible");
      uploadedFileName.textContent = "";
    });
  }
  loadShapImage();
  wordFlick();

  document.querySelectorAll(".auth-logout-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      clearStoredUser();
      updateHeaderForUser();
    });
  });

  document.getElementById("btn-login")?.addEventListener("click", (e) => {
    openLoginModal();
  });
  document.getElementById("btn-signup")?.addEventListener("click", (e) => {
    openSignupModal();
  });

  document.getElementById("signup-modal-close")?.addEventListener("click", closeSignupModal);
  document.getElementById("signup-modal-overlay")?.addEventListener("click", closeSignupModal);
  document.getElementById("signup-to-signin")?.addEventListener("click", (e) => {
    e.preventDefault();
    closeSignupModal();
    openLoginModal();
  });

  document.getElementById("login-modal-close")?.addEventListener("click", closeLoginModal);
  document.getElementById("login-modal-overlay")?.addEventListener("click", closeLoginModal);
  document.getElementById("login-to-signup")?.addEventListener("click", (e) => {
    e.preventDefault();
    closeLoginModal();
    openSignupModal();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (document.getElementById("signup-modal")?.classList.contains("open")) closeSignupModal();
      if (document.getElementById("login-modal")?.classList.contains("open")) closeLoginModal();
    }
  });

  document.getElementById("signup-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const statusEl = document.getElementById("signup-status");
    const btn = document.getElementById("signup-submit-btn");
    const agreementCheckbox = form.querySelector('input[name="agreement"]');
    const payload = {
      firstName: form.firstName?.value?.trim() || "",
      lastName: form.lastName?.value?.trim() || "",
      username: form.username?.value?.trim() || "",
      email: form.email?.value?.trim() || "",
      password: form.password?.value || "",
      agreement: !!agreementCheckbox?.checked,
    };
    if (statusEl) statusEl.textContent = "";
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Creating...";
    }
    try {
      const resp = await fetch(API_BASE + "/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        const msg = typeof data.detail === "string" ? data.detail : Array.isArray(data.detail) ? data.detail.map((x) => x.msg || x).join(" ") : "Registration failed.";
        if (statusEl) statusEl.textContent = msg;
        return;
      }
      if (statusEl) statusEl.textContent = data.message || "Account created.";
      setStoredUser({ username: payload.username });
      updateHeaderForUser();
      setTimeout(() => {
        closeSignupModal();
        form.reset();
        if (statusEl) statusEl.textContent = "";
      }, 1200);
    } catch (err) {
      if (statusEl) statusEl.textContent = "Error: " + (err.message || "Network error");
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = "Submit";
      }
    }
  });
  document.getElementById("login-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const statusEl = document.getElementById("login-status");
    const btn = document.getElementById("login-submit-btn");
    const username = form.username?.value?.trim() || "";
    const password = form.password?.value || "";
    if (statusEl) {
      statusEl.textContent = "";
      statusEl.classList.remove("error");
    }
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Logging in...";
    }
    try {
      const resp = await fetch(API_BASE + "/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await resp.json().catch(() => ({}));
      let msg = typeof data.detail === "string" ? data.detail : data.detail?.msg || "Invalid username or password.";
      if (resp.status === 404) {
        msg = "Login not available. Restart the backend: uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload";
      }
      if (!resp.ok) {
        if (statusEl) {
          statusEl.textContent = msg;
          statusEl.classList.add("error");
        }
        return;
      }
      if (data.user) {
        setStoredUser({ username: data.user.username });
        updateHeaderForUser();
      }
      closeLoginModal();
      form.reset();
      if (statusEl) statusEl.textContent = "";
    } catch (err) {
      if (statusEl) {
        statusEl.textContent = "Error: " + (err.message || "Network error");
        statusEl.classList.add("error");
      }
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = "Log in";
      }
    }
  });

  // Pop out sections when scrolling down from hero
  const revealEls = document.querySelectorAll(".reveal");
  if (revealEls.length > 0 && "IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries, observer) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        }
      },
      { threshold: 0.15, rootMargin: "0px 0px -10% 0px" }
    );
    revealEls.forEach((el) => io.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("is-visible"));
  }

  // Fade out title and auth buttons on scroll; when title is fully gone, pop up the sentence and show unlock notice (once per user)
  const UNLOCK_NOTICE_DISMISSED_KEY = "thelag-unlock-notice-dismissed";
  const hero = document.getElementById("project");
  const authHeader = document.querySelector(".auth-header");
  const unlockNotice = document.getElementById("unlock-notice");
  const scrollFade = 40;
  const titleGone = 120;
  const sentenceFade = 200;  /* first scroll: sentence starts fading (like title at 40) */
  const sentenceGone = 280;  /* second scroll: sentence gone (like title at 120) */

  function isUnlockNoticeDismissed() {
    return localStorage.getItem(UNLOCK_NOTICE_DISMISSED_KEY) === "true";
  }

  window.addEventListener("scroll", () => {
    const y = window.scrollY || window.pageYOffset;
    if (hero) {
      if (y > scrollFade) {
        hero.classList.add("scrolled");
      } else {
        hero.classList.remove("scrolled");
      }
      if (y > titleGone) {
        hero.classList.add("title-gone");
      } else {
        hero.classList.remove("title-gone");
      }
      if (y > sentenceFade) {
        hero.classList.add("sentence-scrolled");
      } else {
        hero.classList.remove("sentence-scrolled");
      }
      if (y > sentenceGone) {
        hero.classList.add("sentence-gone");
      } else {
        hero.classList.remove("sentence-gone");
      }
    }
    if (authHeader) {
      if (y > titleGone) {
        authHeader.classList.add("scrolled");
      } else {
        authHeader.classList.remove("scrolled");
      }
    }
    if (unlockNotice) {
      if (y > titleGone && !isUnlockNoticeDismissed()) {
        unlockNotice.classList.add("visible");
      } else {
        unlockNotice.classList.remove("visible");
      }
    }
  });

  document.getElementById("unlock-notice-dismiss")?.addEventListener("click", () => {
    localStorage.setItem(UNLOCK_NOTICE_DISMISSED_KEY, "true");
    unlockNotice?.classList.remove("visible");
  });
});


