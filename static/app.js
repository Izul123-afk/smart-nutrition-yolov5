const NUTRITION_DB = {
    "ayam bakar": { energi: 190, protein: 24, lemak: 8, karbo: 3 },
    "ayam goreng": { energi: 260, protein: 21.9, lemak: 14.6, karbo: 10.8 },
    "bakso": { energi: 190, protein: 10.3, lemak: 6.3, karbo: 23.1 },
    "capcay kuah": { energi: 80, protein: 4, lemak: 3.5, karbo: 7 },
    "coto makassar": { energi: 96, protein: 6.8, lemak: 5.9, karbo: 3.8 },
    "gule": { energi: 126, protein: 4.2, lemak: 9.4, karbo: 6.2 },
    "ikan bakar": { energi: 140, protein: 20, lemak: 5, karbo: 2 },
    "ikan goreng": { energi: 167, protein: 13.3, lemak: 10, karbo: 6.7 },
    "kangkung": { energi: 98, protein: 2.6, lemak: 8.7, karbo: 4 },
    "kare ayam": { energi: 135, protein: 8, lemak: 8, karbo: 6.6 },
    "laksa": { energi: 120, protein: 5, lemak: 7, karbo: 10.7 },
    "lodeh": { energi: 70, protein: 2.5, lemak: 5, karbo: 8 },
    "lontong sayur": { energi: 90, protein: 2.5, lemak: 5, karbo: 9 },
    "mie ayam": { energi: 175, protein: 7, lemak: 7.8, karbo: 19.3 },
    "mie kocok": { energi: 110, protein: 6, lemak: 3.5, karbo: 14 },
    "nasi goreng": { energi: 173, protein: 7.3, lemak: 3.7, karbo: 27.6 },
    "nasi putih": { energi: 180, protein: 3, lemak: 0.3, karbo: 39.8 },
    "nasi soto": { energi: 150, protein: 6, lemak: 4, karbo: 22 },
    "nasi kuning": { energi: 95, protein: 1.9, lemak: 0.2, karbo: 20.9 },
    "nasi pecel": { energi: 230, protein: 6.6, lemak: 9.3, karbo: 31.7 },
    "ramen": { energi: 140, protein: 6, lemak: 5, karbo: 17 },
    "rawon": { energi: 60, protein: 5.4, lemak: 2.5, karbo: 4 },
    "rendang": { energi: 193, protein: 22.6, lemak: 7.9, karbo: 7.8 },
    "sate ayam": { energi: 225, protein: 19.5, lemak: 14.8, karbo: 4.9 },
    "sayur asem": { energi: 64, protein: 2.1, lemak: 0.6, karbo: 12.5 },
    "seblak": { energi: 131, protein: 4.1, lemak: 6.7, karbo: 15.6 },
    "sop ayam": { energi: 50, protein: 4, lemak: 2, karbo: 4 },
    "sop buntut": { energi: 71, protein: 7.5, lemak: 3.6, karbo: 2.2 },
    "sop iga": { energi: 90, protein: 8, lemak: 5, karbo: 2.5 },
    "sop kambing": { energi: 34, protein: 5.5, lemak: 0.8, karbo: 1.1 },
    "soto betawi": { energi: 110, protein: 2.9, lemak: 9.5, karbo: 3.2 },
    "tahu goreng": { energi: 115, protein: 9.7, lemak: 8.5, karbo: 2.5 },
    "telur balado": { energi: 202, protein: 10, lemak: 16, karbo: 3.5 },
    "telur goreng": { energi: 166, protein: 14.2, lemak: 11.2, karbo: 0.9 },
    "telur rebus": { energi: 155, protein: 12.6, lemak: 10.6, karbo: 1.2 },
    "tempe goreng": { energi: 225, protein: 13.3, lemak: 15.2, karbo: 11.9 },
    "terong balado": { energi: 97, protein: 1.2, lemak: 8, karbo: 7.1 },
    "tomyum": { energi: 45, protein: 3.5, lemak: 1.5, karbo: 3 },
    "tongseng": { energi: 115, protein: 6, lemak: 6.5, karbo: 7 },
};

function normalizeLabel(label) {
    return String(label || "").trim().toLowerCase().replace(/\s+/g, " ");
}

function lookupNutrition(label) {
    const key = normalizeLabel(label);
    if (NUTRITION_DB[key]) return NUTRITION_DB[key];
    for (const [k, v] of Object.entries(NUTRITION_DB)) {
        if (key.includes(k) || k.includes(key)) return v;
    }
    return null;
}

const HISTORY_RANGES = ["6h", "24h", "7d", "30d", "all"];
const RANGE_LABELS = {
    "6h": "6 jam terakhir",
    "24h": "24 jam terakhir",
    "7d": "7 hari terakhir",
    "30d": "30 hari terakhir",
    "all": "semua data",
};

function loadSavedRange() {
    try {
        const saved = window.localStorage.getItem("historyRange");
        return HISTORY_RANGES.includes(saved) ? saved : "all";
    } catch (error) {
        return "all";
    }
}

const state = {
    history: [],
    historyRange: loadSavedRange(),
    historyCount: 0,
    scanning: false,
    pipelineTimer: null,
    pipelineIndex: 0,
};

const elements = {
    systemClock: document.getElementById("systemClock"),
    systemDate: document.getElementById("systemDate"),
    systemBadge: document.getElementById("systemBadge"),
    cameraStatus: document.getElementById("cameraStatus"),
    cameraDetail: document.getElementById("cameraDetail"),
    sensorStatus: document.getElementById("sensorStatus"),
    sensorDetail: document.getElementById("sensorDetail"),
    modelStatus: document.getElementById("modelStatus"),
    modelDetail: document.getElementById("modelDetail"),
    liveWeight: document.getElementById("liveWeight"),
    scanButton: document.getElementById("scanButton"),
    tareButton: document.getElementById("tareButton"),
    clearHistoryButton: document.getElementById("clearHistoryButton"),
    resultState: document.getElementById("resultState"),
    resultImage: document.getElementById("resultImage"),
    resultPlaceholder: document.getElementById("resultPlaceholder"),
    detectedLabel: document.getElementById("detectedLabel"),
    detectedConfidence: document.getElementById("detectedConfidence"),
    detectedWeight: document.getElementById("detectedWeight"),
    detectedCount: document.getElementById("detectedCount"),
    nutriEnergi: document.getElementById("nutriEnergi"),
    nutriProtein: document.getElementById("nutriProtein"),
    nutriLemak: document.getElementById("nutriLemak"),
    nutriKarbo: document.getElementById("nutriKarbo"),
    nutriTotalEnergi: document.getElementById("nutriTotalEnergi"),
    nutriTotalProtein: document.getElementById("nutriTotalProtein"),
    nutriTotalLemak: document.getElementById("nutriTotalLemak"),
    nutriTotalKarbo: document.getElementById("nutriTotalKarbo"),
    nutritionNote: document.getElementById("nutritionNote"),
    nutritionBasis: document.getElementById("nutritionBasis"),
    totalInference: document.getElementById("totalInference"),
    captureTime: document.getElementById("captureTime"),
    preprocessTime: document.getElementById("preprocessTime"),
    inferenceTime: document.getElementById("inferenceTime"),
    postprocessTime: document.getElementById("postprocessTime"),
    weightTime: document.getElementById("weightTime"),
    historyBody: document.getElementById("historyBody"),
    historyCount: document.getElementById("historyCount"),
    rangeFilter: document.getElementById("rangeFilter"),
    exportPdf: document.getElementById("exportPdf"),
    imageModal: document.getElementById("imageModal"),
    imageModalTitle: document.getElementById("imageModalTitle"),
    imageModalMeta: document.getElementById("imageModalMeta"),
    imageModalAnnotated: document.getElementById("imageModalAnnotated"),
    imageModalAnnotatedName: document.getElementById("imageModalAnnotatedName"),
    imageModalOriginalFigure: document.getElementById("imageModalOriginalFigure"),
    imageModalOriginal: document.getElementById("imageModalOriginal"),
    imageModalOriginalName: document.getElementById("imageModalOriginalName"),
    weightChart: document.getElementById("weightChart"),
    performanceChart: document.getElementById("performanceChart"),
    weightChartEmpty: document.getElementById("weightChartEmpty"),
    performanceChartEmpty: document.getElementById("performanceChartEmpty"),
    footerModel: document.getElementById("footerModel"),
    cameraOverlayText: document.getElementById("cameraOverlayText"),
    toast: document.getElementById("toast"),
};

function updateClock() {
    const now = new Date();
    elements.systemClock.textContent = now.toLocaleTimeString("id-ID", {
        hour12: false,
    });
    elements.systemDate.textContent = now.toLocaleDateString("id-ID", {
        day: "2-digit",
        month: "short",
        year: "numeric",
    });
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function responseSucceeded(data) {
    return data?.success === true || data?.ok === true;
}

function confidencePercent(value) {
    const numeric = Number(value || 0);
    return numeric <= 1 ? numeric * 100 : numeric;
}

function scanWeight(data) {
    return Number(data?.weight_grams ?? data?.weight ?? data?.berat ?? 0);
}

function setSystemBadge(ok, text) {
    elements.systemBadge.innerHTML = `
        <span class="pulse-dot" style="${ok ? "" : "background: var(--danger)"}"></span>
        <span>${escapeHtml(text)}</span>
    `;
}

function statusText(connected, successText, errorText) {
    return connected ? successText : errorText;
}

async function fetchStatus() {
    try {
        const response = await fetch("/api/status", { cache: "no-store" });
        const data = await response.json();
        if (!response.ok || !responseSucceeded(data)) {
            throw new Error(data.error || "Status sistem gagal dibaca.");
        }

        const camera = data.camera || {};
        const sensor = data.sensor || data.scale || {};
        const model = data.model || {};
        const modelSize = Number(
            model.image_size || model.input_width || model.input_height || 0,
        );

        elements.cameraStatus.textContent = statusText(
            camera.connected,
            "Terhubung",
            "Terputus",
        );
        elements.cameraDetail.textContent = camera.connected
            ? `${camera.width || "?"}×${camera.height || "?"} · ${camera.fps || 0} FPS`
            : camera.error || "Tidak ada frame";

        elements.sensorStatus.textContent = statusText(
            sensor.connected,
            "Aktif",
            "Tidak aktif",
        );
        elements.sensorDetail.textContent = sensor.connected
            ? `Faktor ${Number(sensor.calibration_factor || 0).toFixed(2)}`
            : sensor.error || "Periksa GPIO";

        elements.modelStatus.textContent = model.loaded ? "Siap" : "Belum siap";
        elements.modelDetail.textContent = model.loaded
            ? `${model.model || "best2.pt"} · ${modelSize || "?"}×${modelSize || "?"}`
            : model.error || "Model tidak ditemukan";
        elements.footerModel.textContent = model.loaded
            ? `Model: ${model.model || "best2.pt"} · ${model.backend || "YOLOv8n"}`
            : "Model: belum tersedia";

        const allReady = Boolean(
            camera.connected && sensor.connected && model.loaded,
        );
        setSystemBadge(
            allReady,
            allReady ? "Sistem siap" : "Sebagian sistem belum siap",
        );
    } catch (error) {
        console.error(error);
        setSystemBadge(false, "Dashboard tidak terhubung");
    }
}

async function fetchWeight() {
    try {
        const response = await fetch("/api/weight", { cache: "no-store" });
        const data = await response.json();
        if (!response.ok || !responseSucceeded(data)) {
            throw new Error(data.error || "Berat gagal dibaca.");
        }
        elements.liveWeight.textContent = Number(
            data.weight_grams ?? data.weight ?? 0,
        ).toFixed(2);
    } catch (error) {
        elements.liveWeight.textContent = "—";
    }
}

function resetPipeline() {
    document.querySelectorAll(".pipeline-step").forEach((step) => {
        step.classList.remove("active", "done");
    });
    state.pipelineIndex = 0;
}

function startPipelineAnimation() {
    resetPipeline();
    const steps = Array.from(document.querySelectorAll(".pipeline-step"));

    const activate = () => {
        steps.forEach((step, index) => {
            step.classList.toggle("active", index === state.pipelineIndex);
            step.classList.toggle("done", index < state.pipelineIndex);
        });

        if (state.pipelineIndex < steps.length - 1) {
            state.pipelineIndex += 1;
        }
    };

    activate();
    state.pipelineTimer = window.setInterval(activate, 850);
}

function normalizedTimings(data) {
    const timings = data?.timings || {};
    const inference = Number(
        timings.inference_ms ?? data?.inference_ms ?? data?.inference_time_ms ?? 0,
    );
    const complete = Number(
        timings.complete_ms ?? data?.total_ms ?? inference,
    );

    return {
        capture_ms: Number(timings.capture_ms || 0),
        preprocess_ms: Number(timings.preprocess_ms || 0),
        inference_ms: inference,
        postprocess_ms: Number(timings.postprocess_ms || 0),
        weight_ms: Number(timings.weight_ms || 0),
        complete_ms: complete,
    };
}

function finishPipeline(timings) {
    if (state.pipelineTimer) {
        window.clearInterval(state.pipelineTimer);
        state.pipelineTimer = null;
    }

    document.querySelectorAll(".pipeline-step").forEach((step) => {
        step.classList.remove("active");
        step.classList.add("done");
    });

    elements.captureTime.textContent = `${timings.capture_ms.toFixed(1)} ms`;
    elements.preprocessTime.textContent = `${timings.preprocess_ms.toFixed(1)} ms`;
    elements.inferenceTime.textContent = `${timings.inference_ms.toFixed(1)} ms`;
    elements.postprocessTime.textContent = `${timings.postprocess_ms.toFixed(1)} ms`;
    elements.weightTime.textContent = `${timings.weight_ms.toFixed(1)} ms`;
    elements.totalInference.textContent = `${timings.complete_ms.toFixed(1)} ms`;
}

function renderNutrition(label, weightGrams, nutritionSnapshot = null) {
    const stored = nutritionSnapshot?.available
        ? nutritionSnapshot.per_100g
        : null;
    const data = stored
        ? {
            energi: Number(stored.energy_kcal || 0),
            protein: Number(stored.protein_g || 0),
            lemak: Number(stored.fat_g || 0),
            karbo: Number(stored.carbohydrate_g || 0),
        }
        : lookupNutrition(label);
    const w = Number(weightGrams || 0);
    if (!data) {
        elements.nutriEnergi.textContent = "—";
        elements.nutriProtein.textContent = "—";
        elements.nutriLemak.textContent = "—";
        elements.nutriKarbo.textContent = "—";
        elements.nutriTotalEnergi.textContent = "—";
        elements.nutriTotalProtein.textContent = "—";
        elements.nutriTotalLemak.textContent = "—";
        elements.nutriTotalKarbo.textContent = "—";
        elements.nutritionNote.textContent = `Data gizi untuk "${label}" belum tersedia di csv.csv`;
        elements.nutritionBasis.textContent = "per 100g";
        return;
    }
    elements.nutriEnergi.textContent = `${data.energi}`;
    elements.nutriProtein.textContent = `${data.protein}`;
    elements.nutriLemak.textContent = `${data.lemak}`;
    elements.nutriKarbo.textContent = `${data.karbo}`;
    if (w > 0) {
        const f = w / 100;
        elements.nutriTotalEnergi.textContent = `${(data.energi * f).toFixed(1)} kkal`;
        elements.nutriTotalProtein.textContent = `${(data.protein * f).toFixed(1)} g`;
        elements.nutriTotalLemak.textContent = `${(data.lemak * f).toFixed(1)} g`;
        elements.nutriTotalKarbo.textContent = `${(data.karbo * f).toFixed(1)} g`;
        elements.nutritionNote.textContent = `Estimasi untuk ${w.toFixed(1)} g (${label}) — nilai per 100g dari csv.csv`;
        elements.nutritionBasis.textContent = `${w.toFixed(0)}g terukur`;
    } else {
        elements.nutriTotalEnergi.textContent = `${data.energi} kkal`;
        elements.nutriTotalProtein.textContent = `${data.protein} g`;
        elements.nutriTotalLemak.textContent = `${data.lemak} g`;
        elements.nutriTotalKarbo.textContent = `${data.karbo} g`;
        elements.nutritionNote.textContent = "Timbang makanan untuk hitung total gizi sesuai berat.";
        elements.nutritionBasis.textContent = "per 100g";
    }
}

function setResultState(type, text) {
    elements.resultState.className = `result-state ${type}`;
    elements.resultState.textContent = text;
}

async function runScan() {
    if (state.scanning) {
        return;
    }

    state.scanning = true;
    elements.scanButton.disabled = true;
    elements.tareButton.disabled = true;
    elements.cameraOverlayText.textContent = "CAPTURING · AI PROCESSING";
    setResultState("processing", "Memproses");
    startPipelineAnimation();

    try {
        const response = await fetch("/api/scan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
        });
        const data = await response.json();

        if (!response.ok || !responseSucceeded(data)) {
            throw new Error(data.error || "Scan gagal.");
        }

        finishPipeline(normalizedTimings(data));

        const image = data.image_url || data.result_image || data.image;
        if (image) {
            elements.resultImage.src = image;
            elements.resultImage.style.display = "block";
            elements.resultPlaceholder.style.display = "none";
        }

        const label = data.food_name || data.label || data.makanan || "Tidak terdeteksi";
        const confidence = confidencePercent(
            data.confidence_percent ?? data.confidence_ratio ?? data.confidence,
        );
        const weight = scanWeight(data);
        const objectCount = Number(data.object_count ?? data.jumlah_objek ?? 0);

        elements.detectedLabel.textContent = label;
        elements.detectedConfidence.textContent = `${confidence.toFixed(2)}%`;
        elements.detectedWeight.textContent = `${weight.toFixed(2)} g`;
        elements.detectedCount.textContent = String(objectCount);
        renderNutrition(label, weight, data.nutrition);

        setResultState(
            "success",
            objectCount > 0 ? "Terdeteksi" : "Tidak ditemukan",
        );
        showToast(
            objectCount > 0
                ? `${label} terdeteksi dengan confidence ${confidence.toFixed(1)}%.`
                : "Tidak ada objek yang melewati batas confidence.",
        );

        await loadHistory();
        await fetchWeight();
    } catch (error) {
        console.error(error);
        stopPipelineWithError();
        setResultState("error", "Gagal");
        showToast(error.message || "Scan gagal.", true);
    } finally {
        state.scanning = false;
        elements.scanButton.disabled = false;
        elements.tareButton.disabled = false;
        elements.cameraOverlayText.textContent = "RAW PREVIEW · INFERENCE OFF";
    }
}

function stopPipelineWithError() {
    if (state.pipelineTimer) {
        window.clearInterval(state.pipelineTimer);
        state.pipelineTimer = null;
    }
    document.querySelectorAll(".pipeline-step").forEach((step) => {
        step.classList.remove("active");
    });
}

async function tareScale() {
    elements.tareButton.disabled = true;
    showToast("Proses tare dimulai. Pastikan timbangan kosong.");

    try {
        const response = await fetch("/api/tare", { method: "POST" });
        const data = await response.json();

        if (!response.ok || !responseSucceeded(data)) {
            throw new Error(data.error || "Tare gagal.");
        }

        const tareValue = Number(data.tare_value ?? data.tare_raw ?? 0);
        showToast(`Tare berhasil. Nilai nol: ${tareValue.toFixed(2)}`);
        await fetchWeight();
    } catch (error) {
        showToast(error.message || "Tare gagal.", true);
    } finally {
        elements.tareButton.disabled = false;
    }
}

function historyQuery() {
    return `range=${encodeURIComponent(state.historyRange)}`;
}

function syncRangeUi() {
    elements.rangeFilter.querySelectorAll(".range-button").forEach((button) => {
        button.classList.toggle(
            "active",
            button.dataset.range === state.historyRange,
        );
    });
    // Export hanya PDF mengikuti filter yang sedang aktif.
    if (elements.exportPdf) {
        elements.exportPdf.href = `/api/export/pdf?${historyQuery()}`;
    }
}

function renderHistoryCount() {
    const shown = state.history.length;
    const total = state.historyCount;
    const label = RANGE_LABELS[state.historyRange] || state.historyRange;
    elements.historyCount.textContent = shown === total
        ? `${total} data · ${label}`
        : `${shown} dari ${total} data · ${label}`;
}

function setHistoryRange(range) {
    if (!HISTORY_RANGES.includes(range) || range === state.historyRange) {
        return;
    }
    state.historyRange = range;
    try {
        window.localStorage.setItem("historyRange", range);
    } catch (error) {
        // localStorage bisa tidak tersedia; filter tetap jalan tanpa disimpan.
    }
    syncRangeUi();
    loadHistory();
}

async function loadHistory() {
    try {
        const response = await fetch(
            `/api/history?${historyQuery()}&limit=500`,
            { cache: "no-store" },
        );
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Riwayat gagal dimuat.");
        }

        state.history = Array.isArray(data)
            ? data
            : Array.isArray(data.records)
                ? data.records
                : [];
        state.historyCount = Number(data.count ?? state.history.length);
        renderHistory();
        renderHistoryCount();
        drawCharts();
    } catch (error) {
        console.error(error);
    }
}

function historyConfidence(row) {
    return confidencePercent(
        row.confidence_percent ?? row.confidence_ratio ?? row.confidence,
    );
}

function historyWeight(row) {
    return Number(row.weight_grams ?? row.weight ?? row.berat ?? 0);
}

function historyImageUrl(row) {
    return row.image_url || row.result_image || row.image || row.original_image_url || "";
}

function historyFileName(row) {
    // Path relatif dari server, mis. data/photos/2026/09/13/<id>/annotated.jpg
    return row.annotated_image_path || row.original_image_path || "";
}

function renderHistory() {
    if (!state.history.length) {
        elements.historyBody.innerHTML = `
            <tr>
                <td colspan="9" class="empty-cell">
                    Tidak ada data untuk ${escapeHtml(RANGE_LABELS[state.historyRange] || "rentang ini")}.
                </td>
            </tr>
        `;
        return;
    }

    elements.historyBody.innerHTML = state.history
        .map((row) => {
            const confidence = historyConfidence(row);
            const weight = historyWeight(row);
            const inference = Number(row.inference_ms || 0);
            const total = Number(row.total_ms ?? inference);
            const label = row.label || row.food_name || row.makanan || "—";
            const imageUrl = historyImageUrl(row);
            const fileName = historyFileName(row);
            const scanId = escapeHtml(row.scan_id || "");

            const thumb = imageUrl
                ? `<button type="button" class="history-thumb" data-scan-id="${scanId}" title="Lihat foto">
                        <img src="${escapeHtml(imageUrl)}" alt="Foto ${escapeHtml(label)}" loading="lazy">
                   </button>`
                : `<span class="history-thumb empty">—</span>`;
            const fileCell = fileName
                ? `<button type="button" class="file-link" data-scan-id="${scanId}" title="${escapeHtml(fileName)}">
                        <code>${escapeHtml(fileName)}</code>
                   </button>`
                : "—";

            return `
                <tr>
                    <td>${thumb}</td>
                    <td>#${scanId || "-"}</td>
                    <td>${escapeHtml(formatTimestamp(row.timestamp))}</td>
                    <td>${escapeHtml(label)}</td>
                    <td>${confidence.toFixed(2)}%</td>
                    <td>${weight.toFixed(2)} g</td>
                    <td class="file-cell">${fileCell}</td>
                    <td>${inference.toFixed(1)} ms</td>
                    <td>${total.toFixed(1)} ms</td>
                </tr>
            `;
        })
        .join("");
}

function openImageModal(scanId) {
    const row = state.history.find((item) => item.scan_id === scanId);
    if (!row) return;

    const label = row.label || row.food_name || "—";
    const annotatedUrl = row.image_url || row.result_image || row.image || "";
    const originalUrl = row.original_image_url || "";

    elements.imageModalTitle.textContent = `${label} · #${row.scan_id}`;
    elements.imageModalMeta.textContent = [
        formatTimestamp(row.timestamp),
        `${historyWeight(row).toFixed(2)} g`,
        `${historyConfidence(row).toFixed(2)}%`,
        row.source ? `sumber: ${row.source}` : null,
    ].filter(Boolean).join(" · ");

    elements.imageModalAnnotated.src = annotatedUrl || originalUrl;
    elements.imageModalAnnotatedName.textContent = row.annotated_image_path || "—";

    if (originalUrl) {
        elements.imageModalOriginal.src = originalUrl;
        elements.imageModalOriginalName.textContent = row.original_image_path || "—";
        elements.imageModalOriginalFigure.hidden = false;
    } else {
        elements.imageModalOriginal.removeAttribute("src");
        elements.imageModalOriginalFigure.hidden = true;
    }

    elements.imageModal.hidden = false;
    document.body.classList.add("modal-open");
}

function closeImageModal() {
    elements.imageModal.hidden = true;
    document.body.classList.remove("modal-open");
    elements.imageModalAnnotated.removeAttribute("src");
    elements.imageModalOriginal.removeAttribute("src");
}

function formatTimestamp(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return value || "—";
    }
    return date.toLocaleString("id-ID", {
        day: "2-digit",
        month: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });
}

function prepareCanvas(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = 220;

    canvas.width = Math.max(1, Math.round(width * ratio));
    canvas.height = Math.round(height * ratio);

    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    return { context, width, height };
}

function drawGrid(context, width, height, padding) {
    context.strokeStyle = "rgba(164, 202, 219, 0.10)";
    context.lineWidth = 1;

    for (let index = 0; index <= 4; index += 1) {
        const y = padding.top
            + ((height - padding.top - padding.bottom) / 4) * index;
        context.beginPath();
        context.moveTo(padding.left, y);
        context.lineTo(width - padding.right, y);
        context.stroke();
    }
}

function drawLineChart(canvas, datasets, options = {}) {
    const { context, width, height } = prepareCanvas(canvas);
    const padding = { top: 18, right: 18, bottom: 30, left: 44 };
    drawGrid(context, width, height, padding);

    const values = datasets.flatMap((dataset) => dataset.values);
    const maxValue = Math.max(options.minimumMax || 1, ...values);
    const minValue = options.minimumValue || 0;
    const usableWidth = width - padding.left - padding.right;
    const usableHeight = height - padding.top - padding.bottom;
    const count = Math.max(1, datasets[0]?.values.length || 1);

    datasets.forEach((dataset, datasetIndex) => {
        context.strokeStyle = dataset.color;
        context.fillStyle = dataset.color;
        context.lineWidth = 2;
        context.beginPath();

        dataset.values.forEach((value, index) => {
            const x = padding.left
                + (count === 1
                    ? usableWidth / 2
                    : (usableWidth * index) / (count - 1));
            const normalized = (value - minValue) / (maxValue - minValue || 1);
            const y = padding.top + usableHeight * (1 - normalized);
            if (index === 0) context.moveTo(x, y);
            else context.lineTo(x, y);
        });
        context.stroke();

        dataset.values.forEach((value, index) => {
            const x = padding.left
                + (count === 1
                    ? usableWidth / 2
                    : (usableWidth * index) / (count - 1));
            const normalized = (value - minValue) / (maxValue - minValue || 1);
            const y = padding.top + usableHeight * (1 - normalized);
            context.beginPath();
            context.arc(x, y, 3.3, 0, Math.PI * 2);
            context.fill();
        });

        context.fillStyle = dataset.color;
        context.font = "11px system-ui";
        context.fillText(
            dataset.label,
            padding.left + datasetIndex * 118,
            height - 8,
        );
    });

    context.fillStyle = "rgba(134, 164, 180, 0.9)";
    context.font = "10px system-ui";
    context.fillText(maxValue.toFixed(0), 7, padding.top + 4);
    context.fillText(minValue.toFixed(0), 18, height - padding.bottom + 3);
}

function drawCharts() {
    const history = [...state.history].reverse().slice(-12);
    const hasData = history.length > 0;

    elements.weightChartEmpty.style.display = hasData ? "none" : "block";
    elements.performanceChartEmpty.style.display = hasData ? "none" : "block";

    if (!hasData) {
        prepareCanvas(elements.weightChart);
        prepareCanvas(elements.performanceChart);
        return;
    }

    const styles = getComputedStyle(document.documentElement);
    const accent = styles.getPropertyValue("--accent").trim() || "#26d69a";
    const accentTwo = styles.getPropertyValue("--accent-2").trim() || "#55a8ff";
    const warning = styles.getPropertyValue("--warning").trim() || "#f2be59";

    drawLineChart(
        elements.weightChart,
        [{
            label: "Berat (g)",
            values: history.map(historyWeight),
            color: accent,
        }],
        { minimumMax: 10, minimumValue: 0 },
    );

    drawLineChart(
        elements.performanceChart,
        [
            {
                label: "Confidence (%)",
                values: history.map(historyConfidence),
                color: accentTwo,
            },
            {
                label: "Inferensi (ms)",
                values: history.map((item) => Number(item.inference_ms || 0)),
                color: warning,
            },
        ],
        { minimumMax: 100, minimumValue: 0 },
    );
}

async function clearHistory() {
    if (!window.confirm("Hapus seluruh riwayat penelitian pada dashboard?")) {
        return;
    }

    try {
        const response = await fetch("/api/history/clear", { method: "POST" });
        const data = await response.json();
        if (!response.ok || !responseSucceeded(data)) {
            throw new Error(data.error || "Riwayat gagal dihapus.");
        }

        state.history = [];
        renderHistory();
        drawCharts();
        showToast("Riwayat berhasil dihapus.");
    } catch (error) {
        showToast(error.message || "Riwayat gagal dihapus.", true);
    }
}

let toastTimer = null;

function showToast(message, isError = false) {
    elements.toast.textContent = message;
    elements.toast.classList.toggle("error", isError);
    elements.toast.classList.add("show");

    if (toastTimer) window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
        elements.toast.classList.remove("show");
    }, 3800);
}

elements.scanButton.addEventListener("click", runScan);
elements.tareButton.addEventListener("click", tareScale);
elements.clearHistoryButton.addEventListener("click", clearHistory);

elements.rangeFilter.addEventListener("click", (event) => {
    const button = event.target.closest(".range-button");
    if (button) setHistoryRange(button.dataset.range);
});

elements.historyBody.addEventListener("click", (event) => {
    const trigger = event.target.closest("[data-scan-id]");
    if (trigger) openImageModal(trigger.dataset.scanId);
});

elements.imageModal.addEventListener("click", (event) => {
    if (event.target.closest("[data-close-modal]")) closeImageModal();
});

window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !elements.imageModal.hidden) closeImageModal();
});

window.addEventListener("resize", () => {
    window.clearTimeout(window.__chartResizeTimer);
    window.__chartResizeTimer = window.setTimeout(drawCharts, 150);
});

updateClock();
window.setInterval(updateClock, 1000);
syncRangeUi();
fetchStatus();
fetchWeight();
loadHistory();
window.setInterval(fetchStatus, 2500);
window.setInterval(fetchWeight, 800);
