/**
 * MindVision Camera — Web Client Module
 *
 * Polls the server for MindVision camera status and streams
 * annotated frames when the capture script is running.
 */

(function () {
    "use strict";

    // ── DOM Elements ────────────────────────────────────────
    const mvStatusBadge = document.getElementById("mvStatusBadge");
    const mvFeedWrapper = document.getElementById("mvFeedWrapper");
    const mvPlaceholder = document.getElementById("mvPlaceholder");
    const mvStream = document.getElementById("mvStream");
    const mvModel = document.getElementById("mvModel");
    const mvInference = document.getElementById("mvInference");
    const mvDetCount = document.getElementById("mvDetCount");
    const mvDetections = document.getElementById("mvDetections");

    // ── State ───────────────────────────────────────────────
    let polling = false;
    let pollTimer = null;
    let connected = false;
    let useStream = false;  // MJPEG vs polling

    // ── Status Check ────────────────────────────────────────
    async function checkStatus() {
        try {
            const res = await fetch("/api/mindvision/status");
            if (!res.ok) throw new Error("not ok");
            const data = await res.json();
            return data.connected === true;
        } catch {
            return false;
        }
    }

    // ── Start MJPEG Stream ──────────────────────────────────
    function startMJPEGStream() {
        mvStream.src = "/api/mindvision/stream?" + Date.now();
        useStream = true;
    }

    function stopMJPEGStream() {
        mvStream.src = "";
        useStream = false;
    }

    // ── Poll Latest Frame ───────────────────────────────────
    async function pollLatestFrame() {
        try {
            const res = await fetch("/api/mindvision/latest");
            if (!res.ok) {
                setDisconnected();
                return;
            }
            const data = await res.json();

            if (!data.connected) {
                setDisconnected();
                return;
            }

            setConnected();

            // Update image
            if (!useStream && data.image) {
                mvStream.src = data.image;
            }

            // Update stats
            mvModel.textContent = `Model: ${data.model || "—"}`;
            mvInference.textContent = `Inference: ${data.inference_ms || 0}ms`;

            // Mirror into dashboard KPI cards
            const kpiInference = document.getElementById("statInference");
            if (kpiInference) kpiInference.textContent = data.inference_ms != null ? data.inference_ms : "—";

            const dets = data.detections || [];
            mvDetCount.textContent = `Detections: ${dets.length}`;

            // Update detection list
            if (dets.length > 0) {
                mvDetections.innerHTML = dets
                    .map(
                        (d) =>
                            `<div class="detection-item">
                                <span class="det-class">${d.class || "chignon"}</span>
                                <span class="det-conf">${(d.confidence * 100).toFixed(1)}%</span>
                                ${d.bbox ? `<span class="det-bbox">[${d.bbox.map((v) => Math.round(v)).join(", ")}]</span>` : ""}
                            </div>`
                    )
                    .join("");
            } else {
                mvDetections.innerHTML = '<p class="placeholder">No detections in current frame</p>';
            }
        } catch {
            // Server unreachable — keep polling
        }
    }

    // ── UI State ────────────────────────────────────────────
    function setConnected() {
        if (connected) return;
        connected = true;
        mvStatusBadge.textContent = "Connected";
        mvStatusBadge.classList.add("badge-success");
        mvStatusBadge.classList.remove("badge-danger");
        mvFeedWrapper.style.display = "block";
        mvPlaceholder.style.display = "none";
        const kpiConn = document.getElementById("statConnection");
        if (kpiConn) kpiConn.textContent = "MindVision";
    }

    function setDisconnected() {
        if (!connected) return;
        connected = false;
        mvStatusBadge.textContent = "Disconnected";
        mvStatusBadge.classList.remove("badge-success");
        const kpiConn = document.getElementById("statConnection");
        if (kpiConn) kpiConn.textContent = "—";
        mvStatusBadge.classList.add("badge-danger");
        mvFeedWrapper.style.display = "none";
        mvPlaceholder.style.display = "flex";
        stopMJPEGStream();
    }

    // ── Main Poll Loop ──────────────────────────────────────
    async function pollLoop() {
        const isConnected = await checkStatus();

        if (isConnected) {
            setConnected();
            // Use polling mode (more metadata than MJPEG)
            await pollLatestFrame();
        } else {
            setDisconnected();
        }

        // Schedule next poll
        pollTimer = setTimeout(pollLoop, connected ? 200 : 3000);
    }

    // ── Init ────────────────────────────────────────────────
    function init() {
        if (!mvStatusBadge) return; // Elements not in page

        mvStatusBadge.classList.add("badge-danger");
        pollLoop();

        // Allow clicking the stream image to toggle MJPEG mode
        mvStream.addEventListener("dblclick", () => {
            if (useStream) {
                stopMJPEGStream();
            } else {
                startMJPEGStream();
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

    // ── Measure Tool for MindVision ────────────────────────────
    function initMVMeasure() {
        const btnMeasure = document.getElementById("btnMeasureMV");
        const streamImg = document.getElementById("mvStream");
        if (!btnMeasure || !streamImg) return;

        const mt = new MeasureTool(streamImg, {
            onCalibrate: (pxToMm) => {
                const manualInput = document.getElementById("manualPxToMm");
                const methodSelect = document.getElementById("measurementMethod");
                if (manualInput) manualInput.value = pxToMm.toFixed(5);
                if (methodSelect) methodSelect.value = "manual";
                if (typeof syncMethodFields === "function") syncMethodFields();
            },
        });
        window._mvMeasure = mt;

        btnMeasure.addEventListener("click", () => {
            if (mt.active) {
                mt.deactivate();
            } else {
                mt.activate();
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initMVMeasure);
    } else {
        initMVMeasure();
    }
})();
