// Automated Inspection — three-stage run (stator → chignon → file)
// Wires the Run Inspection button, progress overlay, results panel, and
// the threshold configuration card in Settings.

(function () {
    const THRESHOLD_KEYS = [
        "chignon.left_area_mm",
        "chignon.right_area_mm",
        "stator.magnet.diag-asc",
        "stator.magnet.diag-desc",
        "stator.mechanical_part.diag-asc",
        "stator.mechanical_part.diag-desc",
    ];

    let pollTimer = null;
    let liveMeasTimer = null;

    function $(id) { return document.getElementById(id); }

    // ── Run Inspection ────────────────────────────────────────────────────
    async function startInspection() {
        const panel = $("inspectionPanel");
        const resultBox = $("inspectionResult");
        if (panel) panel.style.display = "block";
        if (resultBox) { resultBox.style.display = "none"; resultBox.innerHTML = ""; }
        setStatusText("Starting…");
        setBar(0);
        setActiveStage(null);

        try {
            const res = await fetch("/api/inspection/start", { method: "POST" });
            const data = await res.json();
            if (!data.ok) {
                setStatusText("Error: " + (data.error || "could not start"));
                return;
            }
            startPolling();
        } catch (e) {
            setStatusText("Error: " + e.message);
        }
    }

    // ── Video preview helpers ────────────────────────────────────────────
    function showVideoPreview() {
        const wrapper = $("videoPreviewWrapper");
        const stream  = $("videoTestStream");
        const placeholder = $("mvPlaceholder");
        if (wrapper)  wrapper.style.display = "block";
        if (placeholder) placeholder.style.display = "none";
        if (stream)   stream.src = "/api/mindvision/stream?" + Date.now();
        // Prevent mindvision.js from hiding the preview during its poll cycle
        window._videoInspectionActive = true;
    }

    function hideVideoPreview() {
        const wrapper = $("videoPreviewWrapper");
        const stream  = $("videoTestStream");
        if (stream)  stream.src = "";
        if (wrapper) wrapper.style.display = "none";
        window._videoInspectionActive = false;
        // Restore placeholder if camera is not connected
        const mvFeed = $("mvFeedWrapper");
        if (mvFeed && mvFeed.style.display === "none") {
            const placeholder = $("mvPlaceholder");
            if (placeholder) placeholder.style.display = "flex";
        }
    }

    function setVideoPreviewStage(label) {
        const el = $("videoPreviewStage");
        if (el) el.textContent = label || "";
        // Show skip button only while in the stator_file stage so user can jump to chignon
        const skipBtn = $("btnSkipToChignon");
        if (skipBtn) skipBtn.style.display = (label === "Stator + File") ? "inline-block" : "none";
    }

    async function skipToChignon() {
        try {
            await fetch("/api/inspection/skip-stage", { method: "POST" });
        } catch (_) {}
    }

    let _isPaused = false;

    async function togglePause() {
        if (_isPaused) {
            try { await fetch("/api/inspection/resume", { method: "POST" }); } catch (_) {}
        } else {
            try { await fetch("/api/inspection/pause",  { method: "POST" }); } catch (_) {}
        }
    }

    function setPausedUI(paused) {
        _isPaused = paused;
        const btn = $("btnPauseInspection");
        if (!btn) return;
        btn.textContent = paused ? "Resume" : "Pause";
        btn.classList.toggle("btn-paused", paused);
    }

    async function setStage(stageName) {
        try {
            await fetch("/api/inspection/set-stage", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ stage: stageName }),
            });
        } catch (_) {}
    }

    // ── Upload video inspection ──────────────────────────────────────────
    async function uploadVideoInspection(file) {
        const panel = $("inspectionPanel");
        const resultBox = $("inspectionResult");
        if (panel) panel.style.display = "block";
        if (resultBox) { resultBox.style.display = "none"; resultBox.innerHTML = ""; }
        setStatusText("Uploading video…");
        setBar(0);
        setActiveStage(null);
        setVideoPreviewStage("");
        showVideoPreview();

        const form = new FormData();
        form.append("video", file);
        try {
            const res = await fetch("/api/inspection/upload-video", { method: "POST", body: form });
            const data = await res.json();
            if (!data.ok) {
                setStatusText("Error: " + (data.error || "could not start"));
                hideVideoPreview();
                return;
            }
            setStatusText("Processing video…");
            startPolling();
        } catch (e) {
            setStatusText("Error: " + e.message);
            hideVideoPreview();
        }
    }

    async function cancelInspection() {
        try { await fetch("/api/inspection/cancel", { method: "POST" }); } catch (_) {}
        stopPolling();
        setStatusText("Stopping…");
        setBar(100);

        // The server thread aggregates synchronously before marking inactive.
        // Poll until done (usually < 300 ms), then render whatever was collected.
        const maxWait = 4000;
        const tick    = 150;
        let waited    = 0;
        while (waited < maxWait) {
            await new Promise(r => setTimeout(r, tick));
            waited += tick;
            try {
                const res  = await fetch("/api/inspection/status");
                const data = await res.json();
                if (!data.active) {
                    if (data.result) {
                        renderResult(data.result);
                    } else {
                        setStatusText("Stopped (no data collected)");
                    }
                    const manCtrl = $("manualStageControls");
                    if (manCtrl) manCtrl.style.display = "none";
                    const pauseBtn = $("btnPauseInspection");
                    if (pauseBtn) pauseBtn.style.display = "none";
                    setPausedUI(false);
                    return;
                }
            } catch (_) {}
        }
        setStatusText("Stopped");
    }

    function startPolling() {
        stopPolling();
        pollTimer = setInterval(pollStatus, 200);
        liveMeasTimer = setInterval(pollLiveMeasurements, 500);
    }

    function stopPolling() {
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        if (liveMeasTimer) { clearInterval(liveMeasTimer); liveMeasTimer = null; }
        hideLiveMeasurements();
    }

    // ── Live measurements polling ────────────────────────────────────────
    function hideLiveMeasurements() {
        const panel = $("liveMeasurementsPanel");
        if (panel) panel.style.display = "none";
    }

    async function pollLiveMeasurements() {
        try {
            const res = await fetch("/api/inspection/live-measurements");
            const data = await res.json();
            if (!data.active) { hideLiveMeasurements(); return; }
            renderLiveMeasurements(data);
        } catch (_) {}
    }

    function renderLiveMeasurements(data) {
        const panel = $("liveMeasurementsPanel");
        const body  = $("liveMeasBody");
        const stageEl = $("liveMeasStage");
        if (!panel || !body) return;

        panel.style.display = "block";
        const stageLabel = data.stage === "stator_file" ? "Stator + File" : (data.stage || "");
        if (stageEl) stageEl.textContent = stageLabel;

        // Show/hide manual stage-switch buttons based on whether a live run is active
        const manCtrl = $("manualStageControls");
        if (manCtrl) manCtrl.style.display = data.active ? "flex" : "none";

        const parts = [];

        // Stator distances
        if (data.stator && data.stator.length > 0) {
            parts.push('<p class="live-section-title">Stator distances</p>');
            parts.push('<table class="live-table"><thead><tr><th>Measurement</th><th>Current</th><th>AVG</th><th>Variance</th><th>n</th></tr></thead><tbody>');
            for (const r of data.stator) {
                const cur  = r.current  != null ? r.current.toFixed(2)  : "—";
                const mean = r.mean     != null ? r.mean.toFixed(2)     : "—";
                const vari = r.variance != null ? r.variance.toFixed(4) : "—";
                parts.push(`<tr><td>${r.label}</td><td>${cur}</td><td>${mean}</td><td>${vari}</td><td>${r.count}</td></tr>`);
            }
            parts.push('</tbody></table>');
        }

        // File position
        const f = data.file || {};
        if (f.samples > 0) {
            const cls = f.decision === "OK" ? "badge-ok" : (f.decision === "NOT OK" ? "badge-bad" : "badge-muted");
            parts.push(`<p class="live-section-title">File position &nbsp;<span class="badge ${cls}">${f.decision || "—"}</span> &nbsp;<small style="opacity:0.6">${Math.round((f.ok_ratio || 0) * 100)}% correct · ${f.samples} frames</small></p>`);
        }

        // Chignon areas
        const hasChignon = data.chignon && data.chignon.some(r => r.count > 0);
        if (hasChignon) {
            parts.push('<p class="live-section-title">Chignon areas</p>');
            parts.push('<table class="live-table"><thead><tr><th>Chignon</th><th>Current (mm²)</th><th>AVG</th><th>Variance</th><th>n</th></tr></thead><tbody>');
            for (const r of data.chignon) {
                const cur  = r.current  != null ? r.current.toFixed(1)  : "—";
                const mean = r.mean     != null ? r.mean.toFixed(1)     : "—";
                const vari = r.variance != null ? r.variance.toFixed(2) : "—";
                parts.push(`<tr><td>${r.label}</td><td>${cur}</td><td>${mean}</td><td>${vari}</td><td>${r.count}</td></tr>`);
            }
            parts.push('</tbody></table>');
        }

        body.innerHTML = parts.join("");
    }

    async function pollStatus() {
        try {
            const res = await fetch("/api/inspection/status");
            const data = await res.json();
            if (data.active) {
                setActiveStage(data.stage);
                const pct = data.stage_total
                    ? Math.min(100, (data.stage_elapsed / data.stage_total) * 100)
                    : 0;
                setBar(pct);
                const stageLabel = data.stage === "stator_file" ? "Stator + File" : (data.stage ? data.stage.charAt(0).toUpperCase() + data.stage.slice(1) : "");
                const elapsed = data.stage_elapsed != null ? data.stage_elapsed.toFixed(1) + "s" : "";
                const pauseSuffix = data.paused ? " · PAUSED" : "";
                setStatusText(`Stage: ${stageLabel} · ${elapsed} · ${data.frame_count} frames${pauseSuffix}`);
                setVideoPreviewStage(stageLabel);
                // Sync pause button
                const pauseBtn = $("btnPauseInspection");
                if (pauseBtn) pauseBtn.style.display = "inline-block";
                setPausedUI(!!data.paused);
            } else {
                stopPolling();
                setBar(100);
                setStatusText("Done");
                setVideoPreviewStage("Done");
                const manCtrl = $("manualStageControls");
                if (manCtrl) manCtrl.style.display = "none";
                const pauseBtn = $("btnPauseInspection");
                if (pauseBtn) pauseBtn.style.display = "none";
                setPausedUI(false);
                if (data.result) renderResult(data.result);
                // Keep preview visible so user can see the last annotated frame,
                // but stop streaming to save bandwidth after a short delay.
                setTimeout(() => {
                    const stream = $("videoTestStream");
                    if (stream) stream.src = "";
                }, 3000);
                window._videoInspectionActive = false;
            }
        } catch (e) {
            // Keep polling on transient errors
        }
    }

    function setActiveStage(stage) {
        document.querySelectorAll(".stage-chip").forEach(el => {
            el.classList.toggle("active", el.dataset.stage === stage);
        });
    }

    function setBar(pct) {
        const bar = $("inspectionBarFill");
        if (bar) bar.style.width = pct + "%";
    }

    function setStatusText(text) {
        const el = $("inspectionStatusText");
        if (el) el.textContent = text;
    }

    // ── Rendering results ────────────────────────────────────────────────
    function fmt(v, digits) {
        if (v === null || v === undefined) return "—";
        return Number(v).toFixed(digits != null ? digits : 2);
    }

    function rowBadge(validation) {
        if (!validation) return '<span class="badge badge-muted">no threshold</span>';
        return validation.valid
            ? '<span class="badge badge-ok">valid</span>'
            : `<span class="badge badge-bad">non valid${validation.reason ? " (" + validation.reason + ")" : ""}</span>`;
    }

    function renderResult(result) {
        const box = $("inspectionResult");
        if (!box) return;

        if (result.cancelled && !result.partial) {
            box.innerHTML = '<p class="inspection-cancelled">Inspection stopped before any data was collected.</p>';
            box.style.display = "block";
            return;
        }
        if (result.error) {
            box.innerHTML = `<p class="inspection-error">Error: ${result.error}</p>`;
            box.style.display = "block";
            return;
        }

        const isPartial = !!result.partial;
        const parts = [];
        const title = isPartial ? "Partial results" : "Inspection results";
        const dur   = result.duration_s != null ? ` <span class="result-duration">(${fmt(result.duration_s, 1)}s)</span>` : "";
        parts.push(`<h4>${title}${dur}</h4>`);
        if (isPartial) {
            parts.push('<p class="result-partial-note">Stopped early — results reflect data collected up to that point.</p>');
        }

        // ── Stage 1: Stator distances ────────────────────────────────────
        parts.push('<div class="result-section"><h5>Stator — diagonal distances</h5>');
        if (!result.stator || result.stator.length === 0) {
            parts.push('<p class="result-empty">No stator measurements captured.</p>');
        } else {
            parts.push('<table class="result-table"><thead><tr><th>Measurement</th><th>Mean</th><th>Variance</th><th>Samples</th><th>Status</th></tr></thead><tbody>');
            for (const row of result.stator) {
                parts.push(`<tr>
                    <td>${row.label}</td>
                    <td>${fmt(row.mean, 2)} ${row.unit}</td>
                    <td>${fmt(row.variance, 3)}</td>
                    <td>${row.count}</td>
                    <td>${rowBadge(row.validation)}</td>
                </tr>`);
            }
            parts.push('</tbody></table>');
        }
        parts.push('</div>');

        // ── Stage 1: File position ───────────────────────────────────────
        const f = result.file || {};
        const decisionClass = f.decision === "OK" ? "badge-ok"
            : (f.decision === "NOT OK" ? "badge-bad" : "badge-muted");
        parts.push(`<div class="result-section"><h5>File — position</h5>
            <p>Decision: <span class="badge ${decisionClass}">${f.decision || "—"}</span>
            ${f.ok_ratio != null
                ? ` &nbsp;·&nbsp; ${Math.round(f.ok_ratio * 100)}% correct &nbsp;·&nbsp; ${f.samples} samples`
                : ""}
            </p>
        </div>`);

        // ── Stage 2: Chignon areas ───────────────────────────────────────
        parts.push('<div class="result-section"><h5>Chignon — surface areas</h5>');
        if (!result.chignon || result.chignon.every(r => r.count === 0)) {
            parts.push('<p class="result-empty">No chignon detections captured.</p>');
        } else {
            parts.push('<table class="result-table"><thead><tr><th>Chignon</th><th>Mean area</th><th>Variance</th><th>Samples</th><th>Status</th></tr></thead><tbody>');
            for (const row of result.chignon) {
                parts.push(`<tr>
                    <td>${row.label}</td>
                    <td>${fmt(row.mean, 2)} ${row.unit}</td>
                    <td>${fmt(row.variance, 3)}</td>
                    <td>${row.count}</td>
                    <td>${rowBadge(row.validation)}</td>
                </tr>`);
            }
            parts.push('</tbody></table>');
        }
        parts.push('</div>');

        box.innerHTML = parts.join("");
        box.style.display = "block";
    }

    // ── Threshold configuration ──────────────────────────────────────────
    async function loadThresholds() {
        try {
            const res = await fetch("/api/inspection/thresholds");
            const data = await res.json();
            for (const key of THRESHOLD_KEYS) {
                const bounds = data[key] || {};
                const minInput = document.querySelector(`[data-thr-key="${key}"][data-thr-field="min"]`);
                const maxInput = document.querySelector(`[data-thr-key="${key}"][data-thr-field="max"]`);
                if (minInput) minInput.value = bounds.min != null ? bounds.min : "";
                if (maxInput) maxInput.value = bounds.max != null ? bounds.max : "";
            }
        } catch (_) {}
    }

    async function saveThresholds() {
        const payload = {};
        for (const key of THRESHOLD_KEYS) {
            const minInput = document.querySelector(`[data-thr-key="${key}"][data-thr-field="min"]`);
            const maxInput = document.querySelector(`[data-thr-key="${key}"][data-thr-field="max"]`);
            const entry = {};
            if (minInput && minInput.value !== "") entry.min = parseFloat(minInput.value);
            if (maxInput && maxInput.value !== "") entry.max = parseFloat(maxInput.value);
            if (Object.keys(entry).length > 0) payload[key] = entry;
        }
        try {
            await fetch("/api/inspection/thresholds", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const saved = $("thresholdsSaved");
            if (saved) {
                saved.style.display = "inline";
                setTimeout(() => { saved.style.display = "none"; }, 1600);
            }
        } catch (_) {}
    }

    // ── Stage duration configuration ─────────────────────────────────────
    const DURATION_KEYS = ["stator_file", "chignon"];

    async function loadDurations() {
        try {
            const res = await fetch("/api/inspection/durations");
            const data = await res.json();
            for (const key of DURATION_KEYS) {
                const input = document.querySelector(`[data-dur-key="${key}"]`);
                if (input && data[key] != null) input.value = data[key];
            }
        } catch (_) {}
    }

    async function saveDurations() {
        const payload = {};
        for (const key of DURATION_KEYS) {
            const input = document.querySelector(`[data-dur-key="${key}"]`);
            if (input && input.value !== "") {
                const v = parseFloat(input.value);
                if (!isNaN(v)) payload[key] = v;
            }
        }
        const errEl = $("durationsError");
        if (errEl) { errEl.style.display = "none"; errEl.textContent = ""; }
        try {
            const res = await fetch("/api/inspection/durations", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok || !data.ok) {
                if (errEl) {
                    errEl.textContent = data.error || "Could not save durations";
                    errEl.style.display = "inline";
                }
                return;
            }
            const saved = $("durationsSaved");
            if (saved) {
                saved.style.display = "inline";
                setTimeout(() => { saved.style.display = "none"; }, 1600);
            }
        } catch (e) {
            if (errEl) {
                errEl.textContent = e.message;
                errEl.style.display = "inline";
            }
        }
    }

    // ── Init ─────────────────────────────────────────────────────────────
    document.addEventListener("DOMContentLoaded", () => {
        const runBtn = $("btnRunInspection");
        if (runBtn) runBtn.addEventListener("click", startInspection);

        // Shared hidden file input — triggered by both buttons
        const videoInput = $("videoInspectionInput");
        if (videoInput) {
            videoInput.addEventListener("change", () => {
                const file = videoInput.files && videoInput.files[0];
                if (file) { videoInput.value = ""; uploadVideoInspection(file); }
            });
        }

        // Toolbar button (visible when camera is connected)
        const btnTestVideo = $("btnTestVideo");
        if (btnTestVideo && videoInput) {
            btnTestVideo.addEventListener("click", () => videoInput.click());
        }

        // Placeholder button (visible when no camera)
        const btnTestVideoPlaceholder = $("btnTestVideoPlaceholder");
        if (btnTestVideoPlaceholder && videoInput) {
            btnTestVideoPlaceholder.addEventListener("click", () => videoInput.click());
        }

        const skipBtn = $("btnSkipToChignon");
        if (skipBtn) skipBtn.addEventListener("click", skipToChignon);

        const btnGoStatorFile = $("btnGoStatorFile");
        if (btnGoStatorFile) btnGoStatorFile.addEventListener("click", () => setStage("stator_file"));

        const btnGoChignon = $("btnGoChignon");
        if (btnGoChignon) btnGoChignon.addEventListener("click", () => setStage("chignon"));

        const pauseBtn = $("btnPauseInspection");
        if (pauseBtn) pauseBtn.addEventListener("click", togglePause);

        const cancelBtn = $("btnCancelInspection");
        if (cancelBtn) cancelBtn.addEventListener("click", cancelInspection);

        const saveBtn = $("btnSaveThresholds");
        if (saveBtn) saveBtn.addEventListener("click", saveThresholds);

        const thrToggle = $("thresholdsToggle");
        const thrBody = $("thresholdsBody");
        const thrChevron = $("thresholdsChevron");
        if (thrToggle && thrBody) {
            thrToggle.addEventListener("click", () => {
                const isHidden = thrBody.style.display === "none";
                thrBody.style.display = isHidden ? "block" : "none";
                if (thrChevron) thrChevron.textContent = isHidden ? "▾" : "▸";
            });
        }

        const durSaveBtn = $("btnSaveDurations");
        if (durSaveBtn) durSaveBtn.addEventListener("click", saveDurations);

        const durToggle = $("durationsToggle");
        const durBody = $("durationsBody");
        const durChevron = $("durationsChevron");
        if (durToggle && durBody) {
            durToggle.addEventListener("click", () => {
                const isHidden = durBody.style.display === "none";
                durBody.style.display = isHidden ? "block" : "none";
                if (durChevron) durChevron.textContent = isHidden ? "▾" : "▸";
            });
        }

        loadThresholds();
        loadDurations();
    });
})();
