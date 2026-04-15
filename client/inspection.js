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

    async function cancelInspection() {
        try {
            await fetch("/api/inspection/cancel", { method: "POST" });
        } catch (_) {}
        stopPolling();
        setStatusText("Cancelled");
    }

    function startPolling() {
        stopPolling();
        pollTimer = setInterval(pollStatus, 200);
    }

    function stopPolling() {
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
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
                setStatusText(
                    `Stage: ${data.stage} · ${data.stage_remaining.toFixed(1)}s · ${data.frame_count} frames`
                );
            } else {
                stopPolling();
                setBar(100);
                setStatusText("Done");
                if (data.result) renderResult(data.result);
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

        if (result.cancelled) {
            box.innerHTML = '<p class="inspection-cancelled">Inspection cancelled.</p>';
            box.style.display = "block";
            return;
        }
        if (result.error) {
            box.innerHTML = `<p class="inspection-error">Error: ${result.error}</p>`;
            box.style.display = "block";
            return;
        }

        const parts = [];
        parts.push(`<h4>Inspection results <span class="result-duration">(${fmt(result.duration_s, 1)}s)</span></h4>`);

        // Stator
        parts.push('<div class="result-section"><h5>Stator — cross-diameter distances</h5>');
        if (!result.stator || result.stator.length === 0) {
            parts.push('<p class="result-empty">No measurements captured.</p>');
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

        // Chignon
        parts.push('<div class="result-section"><h5>Chignon — surface areas</h5>');
        if (!result.chignon || result.chignon.every(r => r.count === 0)) {
            parts.push('<p class="result-empty">No chignon detections captured.</p>');
        } else {
            parts.push('<table class="result-table"><thead><tr><th>Chignon</th><th>Mean</th><th>Variance</th><th>Samples</th><th>Status</th></tr></thead><tbody>');
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

        // File
        const f = result.file || {};
        const decisionClass = f.decision === "OK"
            ? "badge-ok"
            : (f.decision === "NOT OK" ? "badge-bad" : "badge-muted");
        parts.push(`<div class="result-section"><h5>File — position</h5>
            <p>Decision: <span class="badge ${decisionClass}">${f.decision || "—"}</span>
            ${f.ok_ratio != null ? ` · ${Math.round(f.ok_ratio * 100)}% correct over ${f.samples} samples` : ""}
            </p>
        </div>`);

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

    // ── Init ─────────────────────────────────────────────────────────────
    document.addEventListener("DOMContentLoaded", () => {
        const runBtn = $("btnRunInspection");
        if (runBtn) runBtn.addEventListener("click", startInspection);

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

        loadThresholds();
    });
})();
