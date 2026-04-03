/**
 * Chignon Detection — WebRTC Client
 *
 * Captures camera frames, sends them to the Flask+aiortc server via
 * a WebRTC PeerConnection, receives annotated frames back, and
 * displays detection results from a DataChannel.
 */

// ═══════════════════════════════════════════════════════════════
// DOM Elements
// ═══════════════════════════════════════════════════════════════
const localVideo = document.getElementById("localVideo");
const remoteVideo = document.getElementById("remoteVideo");
const videoOverlay = document.getElementById("videoOverlay");
const statusBadge = document.getElementById("statusBadge");
const modelSelect = document.getElementById("modelSelect");
const btnConnect = document.getElementById("btnConnect");
const btnDisconnect = document.getElementById("btnDisconnect");
const btnSnapshot = document.getElementById("btnSnapshot");
const enableDepthViz = document.getElementById("enableDepthViz");
const depthPanel = document.getElementById("depthPanel");
const depthStream = document.getElementById("depthStream");

// Stats
const statInference = document.getElementById("statInference");
const statFPS = document.getElementById("statFPS");
const statModel = document.getElementById("statModel");
const statConnection = document.getElementById("statConnection");

// Live Monitoring Stats
const statAvgLatency = document.getElementById("statAvgLatency");
const statAvgThroughput = document.getElementById("statAvgThroughput");
const statErrorRate = document.getElementById("statErrorRate");
const statTotalRequests = document.getElementById("statTotalRequests");

// Benchmark Stats
const benchIoU = document.getElementById("benchIoU");
const benchDice = document.getElementById("benchDice");
const benchTime = document.getElementById("benchTime");
const benchGPU = document.getElementById("benchGPU");

// Detections
const detectionsList = document.getElementById("detectionsList");

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════
let pc = null;            // RTCPeerConnection
let localStream = null;   // Camera MediaStream
let dataChannel = null;   // DataChannel for results
let lastLiveDetections = [];  // Latest live detection data for save
let lastUploadData = null;    // Latest upload detection response for save

// Upload elements
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const uploadPreview = document.getElementById("uploadPreview");
const uploadLoading = document.getElementById("uploadLoading");
const uploadOriginal = document.getElementById("uploadOriginal");
const uploadResult = document.getElementById("uploadResult");
const uploadInference = document.getElementById("uploadInference");
const uploadModel = document.getElementById("uploadModel");
const uploadDetections = document.getElementById("uploadDetections");
const uploadDepth = document.getElementById("uploadDepth");
const uploadDepthWrap = document.getElementById("uploadDepthWrap");
const btnUploadAnother = document.getElementById("btnUploadAnother");
const btnDownloadResult = document.getElementById("btnDownloadResult");
const btnSaveUpload = document.getElementById("btnSaveUpload");
const btnSaveLive = document.getElementById("btnSaveLive");

// SOTA Enhancements
const enhancementsToggle = document.getElementById("enhancementsToggle");
const enhancementsBody = document.getElementById("enhancementsBody");
const enhancementsChevron = document.getElementById("enhancementsChevron");
const enableTracking = document.getElementById("enableTracking");
const enableEdgeRefinement = document.getElementById("enableEdgeRefinement");
const btnSaveEnhancements = document.getElementById("btnSaveEnhancements");
const enhancementsSaved = document.getElementById("enhancementsSaved");

// ═══════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    loadModels();
    btnConnect.addEventListener("click", connect);
    btnDisconnect.addEventListener("click", disconnect);
    btnSnapshot.addEventListener("click", takeSnapshot);
    btnSaveLive.addEventListener("click", saveLiveDetection);
    modelSelect.addEventListener("change", onModelChange);
    initUpload();
});

// ═══════════════════════════════════════════════════════════════
// Model Management
// ═══════════════════════════════════════════════════════════════

async function loadModels() {
    try {
        const res = await fetch("/api/models");
        const data = await res.json();

        modelSelect.innerHTML = "";
        if (data.models.length === 0) {
            modelSelect.innerHTML = '<option value="">No models found</option>';
            return;
        }

        // Group: trained YOLO, pretrained YOLO, PyTorch
        const yoloTrained = data.models.filter((m) => m.type === "yolo" && m.source === "trained");
        const yoloPretrained = data.models.filter((m) => m.type === "yolo" && m.source === "pretrained");
        const pytorch = data.models.filter((m) => m.type === "pytorch");

        const groups = [
            { label: "YOLO Models (trained)", items: yoloTrained },
            { label: "YOLO Models (pretrained)", items: yoloPretrained },
            { label: "PyTorch Models", items: pytorch },
        ];

        groups.forEach(({ label, items }) => {
            if (items.length === 0) return;
            const group = document.createElement("optgroup");
            group.label = label;
            items.forEach((m) => {
                const opt = document.createElement("option");
                opt.value = m.name;
                opt.textContent = m.display_name || m.name;
                if (m.active) opt.selected = true;
                group.appendChild(opt);
            });
            modelSelect.appendChild(group);
        });

        modelSelect.disabled = false;
        statModel.textContent = data.current || "none";
        if (data.current) loadModelPerformance(data.current);
    } catch (err) {
        console.error("Failed to load models:", err);
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
    }
}

async function loadModelPerformance(modelName) {
    benchIoU.textContent = "—";
    benchDice.textContent = "—";
    benchTime.textContent = "—";
    benchGPU.textContent = "—";
    try {
        const res = await fetch(`/api/performance/${modelName}`);
        if (!res.ok) return;
        const data = await res.json();
        
        if (data.best_val_iou !== null && data.best_val_iou !== undefined) {
            benchIoU.textContent = data.best_val_iou.toFixed(4);
        }
        if (data.best_val_dice !== null && data.best_val_dice !== undefined) {
            benchDice.textContent = data.best_val_dice.toFixed(4);
        }
        if (data.total_train_time_sec !== null && data.total_train_time_sec !== undefined) {
            const mins = data.total_train_time_sec / 60;
            benchTime.textContent = mins >= 60 ? (mins/60).toFixed(1) + ' h' : mins.toFixed(1) + ' m';
        }
        if (data.peak_gpu_memory_mb !== null && data.peak_gpu_memory_mb !== undefined) {
            benchGPU.textContent = Math.round(data.peak_gpu_memory_mb) + ' MB';
        }
    } catch (e) {
        console.warn("Could not load model performance:", e);
    }
}

async function pollLiveMetrics() {
    try {
        const res = await fetch('/api/live-metrics');
        if (!res.ok) return;
        const data = await res.json();
        
        if (statAvgLatency) statAvgLatency.textContent = data.avg_latency_ms.toFixed(1);
        if (statAvgThroughput) statAvgThroughput.textContent = data.throughput_fps.toFixed(1);
        if (statErrorRate) statErrorRate.textContent = data.error_rate_percent.toFixed(1);
        if (statTotalRequests) statTotalRequests.textContent = data.total_requests;
    } catch (err) {
        // silently ignore polling errors to avoid console spam
    }
}
// Poll live metrics every 2 seconds
setInterval(pollLiveMetrics, 2000);

async function onModelChange() {
    const model = modelSelect.value;
    if (!model) return;

    modelSelect.disabled = true;
    try {
        const res = await fetch("/api/select-model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model }),
        });
        const data = await res.json();
        if (data.error) {
            alert(`Failed to load model: ${data.error}`);
        } else {
            statModel.textContent = data.model;
            // Refresh the label list for the reference-label measurement method
            if (typeof loadLabels === "function") loadLabels();
            loadModelPerformance(data.model);
        }
    } catch (err) {
        console.error("Model switch failed:", err);
        alert("Failed to switch model. See console for details.");
    } finally {
        modelSelect.disabled = false;
    }
}

// ═══════════════════════════════════════════════════════════════
// WebRTC Connection
// ═══════════════════════════════════════════════════════════════

async function connect() {
    setStatus("connecting");
    btnConnect.disabled = true;

    try {
        // 1. Get camera
        localStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 15, max: 30 },
            },
            audio: false,
        });
        localVideo.srcObject = localStream;

        // 2. Create PeerConnection
        pc = new RTCPeerConnection({
            iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
        });

        // 3. Create DataChannel for receiving detection results
        dataChannel = pc.createDataChannel("results", { ordered: true });
        dataChannel.onmessage = onDataChannelMessage;
        dataChannel.onopen = () => console.log("DataChannel open");
        dataChannel.onclose = () => console.log("DataChannel closed");

        // 4. Add camera track
        localStream.getTracks().forEach((track) => {
            pc.addTrack(track, localStream);
        });

        // 5. Handle remote track (annotated video from server)
        pc.ontrack = (event) => {
            console.log("Remote track received:", event.track.kind);
            if (event.streams && event.streams[0]) {
                remoteVideo.srcObject = event.streams[0];
                videoOverlay.classList.add("hidden");
            }
        };

        // 6. Monitor connection state
        pc.onconnectionstatechange = () => {
            const state = pc.connectionState;
            console.log("Connection state:", state);
            statConnection.textContent = state;

            if (state === "connected") {
                setStatus("connected");
                btnDisconnect.disabled = false;
                btnSnapshot.disabled = false;
                btnSaveLive.disabled = false;
            } else if (state === "failed" || state === "disconnected") {
                setStatus("disconnected");
            }
        };

        pc.oniceconnectionstatechange = () => {
            console.log("ICE state:", pc.iceConnectionState);
        };

        // 7. Create offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // 8. Wait for ICE gathering to complete
        await waitForIceGathering(pc);

        // 9. Send offer to server
        const response = await fetch("/offer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sdp: pc.localDescription.sdp,
                type: pc.localDescription.type,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const answer = await response.json();

        // 10. Set remote description (server's answer)
        await pc.setRemoteDescription(
            new RTCSessionDescription({
                sdp: answer.sdp,
                type: answer.type,
            })
        );

        console.log("WebRTC connection established");
    } catch (err) {
        console.error("Connection failed:", err);
        setStatus("disconnected");
        btnConnect.disabled = false;
        alert(`Connection failed: ${err.message}`);
        cleanup();
    }
}

function waitForIceGathering(pc) {
    return new Promise((resolve) => {
        if (pc.iceGatheringState === "complete") {
            resolve();
            return;
        }

        const timeout = setTimeout(() => {
            console.warn("ICE gathering timed out, proceeding with partial candidates");
            resolve();
        }, 5000);

        pc.onicegatheringstatechange = () => {
            if (pc.iceGatheringState === "complete") {
                clearTimeout(timeout);
                resolve();
            }
        };
    });
}

async function disconnect() {
    cleanup();
    setStatus("disconnected");
    btnConnect.disabled = false;
    btnDisconnect.disabled = true;
    btnSnapshot.disabled = true;
    btnSaveLive.disabled = true;
    resetStats();
}

function cleanup() {
    if (pc) {
        pc.close();
        pc = null;
    }
    if (localStream) {
        localStream.getTracks().forEach((t) => t.stop());
        localStream = null;
    }
    localVideo.srcObject = null;
    remoteVideo.srcObject = null;
    videoOverlay.classList.remove("hidden");
    dataChannel = null;
}

// ═══════════════════════════════════════════════════════════════
// DataChannel — Receive Detection Results
// ═══════════════════════════════════════════════════════════════

function onDataChannelMessage(event) {
    try {
        const data = JSON.parse(event.data);
        updateStats(data);
        lastLiveDetections = data.detections || [];
        updateDetections(lastLiveDetections);

        // Inference Health tracking
        if (data.inference_ms != null) {
            pushLatency(data.inference_ms);
            drawSparkline();
            updateHealthStats();
        }
        if (data.alerts && data.alerts.length > 0) {
            updateAlertFeed(data.alerts);
        }
    } catch (err) {
        console.warn("Bad DataChannel message:", err);
    }
}

function updateStats(data) {
    statInference.textContent = data.inference_ms ?? "—";
    statFPS.textContent = data.server_fps ?? "—";
    statModel.textContent = data.model ?? "—";
}

function renderMeasurementHtml(measurements) {
    if (!measurements || measurements.length === 0) return "";
    return `
        <div class="measurements">
            ${measurements.map(m => {
                const useMm = m.value_mm != null && m.value_px != null
                              && Math.abs(m.value_mm - m.value_px) > 0.01;
                const val = useMm ? m.value_mm : (m.value_px != null ? m.value_px : 0);
                const isArea = m.type === "area";
                const unit = useMm ? (isArea ? "mm\u00B2" : "mm") : (isArea ? "px\u00B2" : "px");
                const label = m.label || m.type;
                return `<span class="measurement-tag">${label}: ${typeof val === "number" ? val.toFixed(2) : val} ${unit}</span>`;
            }).join("")}
        </div>`;
}

function updateDetections(detections) {
    if (detections.length === 0) {
        detectionsList.innerHTML = '<p class="placeholder">No detections</p>';
        return;
    }

    detectionsList.innerHTML = detections
        .map((d) => {
            const conf = d.confidence ?? 0;
            const confClass =
                conf >= 0.75 ? "high" : conf >= 0.5 ? "medium" : "low";
            const label = d.class_name ?? `Class ${d.class_id}`;

            return `
                <div class="detection-item">
                    <span class="detection-class">${label}</span>
                    <span class="detection-conf ${confClass}">
                        ${(conf * 100).toFixed(1)}%
                    </span>
                    ${d.approx_depth_mm ? `<span class="measurement-tag depth-tag">Depth: ~${d.approx_depth_mm} mm</span>` : ""}
                    ${renderMeasurementHtml(d.measurements)}
                </div>`;
        })
        .join("");
}

// ═══════════════════════════════════════════════════════════════
// Snapshot
// ═══════════════════════════════════════════════════════════════

function takeSnapshot() {
    const video = remoteVideo;
    if (!video.videoWidth) return;

    const canvas = document.getElementById("snapshotCanvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const link = document.createElement("a");
    link.download = `detection_${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
}

// ═══════════════════════════════════════════════════════════════
// UI Helpers
// ═══════════════════════════════════════════════════════════════

function setStatus(status) {
    statusBadge.className = "badge";
    switch (status) {
        case "connected":
            statusBadge.textContent = "Connected";
            statusBadge.classList.add("connected");
            break;
        case "connecting":
            statusBadge.textContent = "Connecting…";
            statusBadge.classList.add("connecting");
            break;
        default:
            statusBadge.textContent = "Disconnected";
            break;
    }
}

function resetStats() {
    statInference.textContent = "—";
    statFPS.textContent = "—";
    statConnection.textContent = "—";
    detectionsList.innerHTML = '<p class="placeholder">No detections yet</p>';
}

// ═══════════════════════════════════════════════════════════════
// Image Upload Detection
// ═══════════════════════════════════════════════════════════════

function initUpload() {
    // Click to browse
    dropzone.addEventListener("click", () => fileInput.click());

    // File selected
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleUpload(fileInput.files[0]);
        }
    });

    // Drag & drop
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleUpload(e.dataTransfer.files[0]);
        }
    });

    // Upload another
    btnUploadAnother.addEventListener("click", resetUpload);

    // Save image + JSON
    btnSaveUpload.addEventListener("click", saveUploadDetection);

    // Download result
    btnDownloadResult.addEventListener("click", () => {
        if (!uploadResult.src) return;
        const link = document.createElement("a");
        link.download = `detection_${Date.now()}.jpg`;
        link.href = uploadResult.src;
        link.click();
    });
}

async function handleUpload(file) {
    if (!file.type.startsWith("image/")) {
        alert("Please select an image file.");
        return;
    }

    // Show loading
    dropzone.style.display = "none";
    uploadPreview.style.display = "none";
    uploadLoading.style.display = "flex";

    // Show original preview
    const objectURL = URL.createObjectURL(file);
    uploadOriginal.src = objectURL;

    // Send to server
    const formData = new FormData();
    formData.append("image", file);

    try {
        const res = await fetch("/api/detect", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (data.error) {
            alert(`Detection failed: ${data.error}`);
            resetUpload();
            return;
        }

        // Show result
        lastUploadData = data;
        uploadResult.src = data.image;
        uploadInference.textContent = `Inference: ${data.inference_ms} ms`;
        uploadModel.textContent = `Model: ${data.model}`;

        // Show depth heatmap if available
        if (data.depth_image) {
            uploadDepth.src = data.depth_image;
            uploadDepthWrap.style.display = "block";
        } else {
            uploadDepth.src = "";
            uploadDepthWrap.style.display = "none";
        }

        // Show detections
        if (data.detections && data.detections.length > 0) {
            uploadDetections.innerHTML = data.detections
                .map((d) => {
                    const conf = d.confidence ?? 0;
                    const confClass =
                        conf >= 0.75 ? "high" : conf >= 0.5 ? "medium" : "low";
                    const label = d.class_name ?? `Class ${d.class_id}`;
                    return `
                        <div class="detection-item">
                            <span class="detection-class">${label}</span>
                            <span class="detection-conf ${confClass}">
                                ${(conf * 100).toFixed(1)}%
                            </span>
                            ${d.approx_depth_mm ? `<span class="measurement-tag depth-tag">Depth: ~${d.approx_depth_mm} mm</span>` : ""}
                            ${renderMeasurementHtml(d.measurements)}
                        </div>`;
                })
                .join("");
        } else {
            uploadDetections.innerHTML =
                '<p class="placeholder">No objects detected</p>';
        }

        uploadLoading.style.display = "none";
        uploadPreview.style.display = "flex";
    } catch (err) {
        console.error("Upload detection failed:", err);
        alert(`Detection failed: ${err.message}`);
        resetUpload();
    }
}

function resetUpload() {
    uploadPreview.style.display = "none";
    uploadLoading.style.display = "none";
    dropzone.style.display = "flex";
    fileInput.value = "";
    uploadOriginal.src = "";
    uploadResult.src = "";
    uploadDepth.src = "";
    uploadDepthWrap.style.display = "none";
    uploadDetections.innerHTML = "";
    lastUploadData = null;
    // Deactivate measure tool if active
    if (window._uploadMeasure) window._uploadMeasure.deactivate();
}

// ═══════════════════════════════════════════════════════════════
// Measure Tool — Upload Section
// ═══════════════════════════════════════════════════════════════

(function initUploadMeasure() {
    const btnMeasure = document.getElementById("btnMeasureUpload");
    const resultImg = document.getElementById("uploadResult");
    if (!btnMeasure || !resultImg) return;

    // Create the MeasureTool once the image element exists
    const mt = new MeasureTool(resultImg, {
        onCalibrate: (pxToMm) => {
            // Sync with the settings panel
            const manualInput = document.getElementById("manualPxToMm");
            const methodSelect = document.getElementById("measurementMethod");
            if (manualInput) manualInput.value = pxToMm.toFixed(5);
            if (methodSelect) methodSelect.value = "manual";
            if (typeof syncMethodFields === "function") syncMethodFields();
        },
    });
    window._uploadMeasure = mt;

    btnMeasure.addEventListener("click", () => {
        if (mt.active) {
            mt.deactivate();
        } else {
            mt.activate();
        }
    });
})();


// ═══════════════════════════════════════════════════════════════
// Save — Download predicted image + JSON coordinates
// ═══════════════════════════════════════════════════════════════

/**
 * Trigger a file download from a data URL or blob URL.
 */
function downloadFile(url, filename) {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
}

/**
 * Download a JSON string as a .json file.
 */
function downloadJSON(obj, filename) {
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    downloadFile(url, filename);
    URL.revokeObjectURL(url);
}

/**
 * Save upload detection result: predicted image + JSON with detections.
 */
function saveUploadDetection() {
    if (!lastUploadData) return;
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const baseName = `detection_${ts}`;

    // Download the predicted image
    downloadFile(lastUploadData.image, `${baseName}.jpg`);

    // Build JSON with coordinates
    const jsonData = {
        timestamp: new Date().toISOString(),
        model: lastUploadData.model,
        inference_ms: lastUploadData.inference_ms,
        detections: (lastUploadData.detections || []).map((d) => ({
            class_id: d.class_id,
            class_name: d.class_name,
            confidence: d.confidence,
            bbox: d.bbox || null,
            has_mask: d.has_mask || false,
            mask_coverage: d.mask_coverage || null,
            measurements: d.measurements || [],
        })),
    };
    downloadJSON(jsonData, `${baseName}.json`);
}

/**
 * Save live WebRTC detection: snapshot of annotated frame + JSON with latest detections.
 */
function saveLiveDetection() {
    const video = remoteVideo;
    if (!video.videoWidth) return;

    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const baseName = `live_detection_${ts}`;

    // Capture current annotated frame
    const canvas = document.getElementById("snapshotCanvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    downloadFile(canvas.toDataURL("image/png"), `${baseName}.png`);

    // Build JSON
    const jsonData = {
        timestamp: new Date().toISOString(),
        model: statModel.textContent || null,
        detections: lastLiveDetections.map((d) => ({
            class_id: d.class_id,
            class_name: d.class_name,
            confidence: d.confidence,
            bbox: d.bbox || null,
            has_mask: d.has_mask || false,
            mask_coverage: d.mask_coverage || null,
            measurements: d.measurements || [],
        })),
    };
    downloadJSON(jsonData, `${baseName}.json`);
}

const API_BASE = "";

// ═══════════════════════════════════════════════════════════════
// Camera Settings & Measurement Post-Processing
// ═══════════════════════════════════════════════════════════════
const settingsToggle = document.getElementById("settingsToggle");
const settingsBody = document.getElementById("settingsBody");
const settingsChevron = document.getElementById("settingsChevron");
const measurementEnabled = document.getElementById("measurementEnabled");
const showEdgeDistances = document.getElementById("showEdgeDistances");
const showCenterDistances = document.getElementById("showCenterDistances");
const showAlignedPairDistances = document.getElementById("showAlignedPairDistances");
const measurementMethod = document.getElementById("measurementMethod");
const sensorWidth = document.getElementById("sensorWidth");
const focalLength = document.getElementById("focalLength");
const objectDistance = document.getElementById("objectDistance");
const refLabelName = document.getElementById("refLabelName");
const refKnownDim = document.getElementById("refKnownDim");
const refDimType = document.getElementById("refDimType");
const manualPxToMm = document.getElementById("manualPxToMm");
const btnSaveSettings = document.getElementById("btnSaveSettings");
const settingsSaved = document.getElementById("settingsSaved");
const fieldsIntrinsics = document.getElementById("fieldsIntrinsics");
const fieldsReference = document.getElementById("fieldsReference");
const fieldsManual = document.getElementById("fieldsManual");

// Toggle settings panel
settingsToggle.addEventListener("click", () => {
    const open = settingsBody.style.display !== "none";
    settingsBody.style.display = open ? "none" : "block";
    settingsChevron.textContent = open ? "▸" : "▾";
});

/** Show/hide the correct parameter group for the chosen method. */
function syncMethodFields() {
    const method = measurementMethod.value;
    fieldsIntrinsics.style.display = method === "camera_intrinsics" ? "" : "none";
    fieldsReference.style.display  = method === "reference_label"   ? "" : "none";
    fieldsManual.style.display     = method === "manual"            ? "" : "none";
}

/** Dim all settings grids when measurements are disabled. */
function syncEnabledState() {
    const on = measurementEnabled.checked;
    [fieldsIntrinsics, fieldsReference, fieldsManual].forEach(el => {
        el.style.opacity = on ? "1" : "0.5";
        el.style.pointerEvents = on ? "auto" : "none";
    });
    measurementMethod.disabled = !on;
    if (showEdgeDistances) showEdgeDistances.disabled = !on;
    if (showCenterDistances) showCenterDistances.disabled = !on;
    if (showAlignedPairDistances) showAlignedPairDistances.disabled = !on;
}

measurementMethod.addEventListener("change", syncMethodFields);
measurementEnabled.addEventListener("change", syncEnabledState);

/** Populate the reference-label <select> with class names from the current model. */
async function loadLabels() {
    try {
        const resp = await fetch(`${API_BASE}/api/labels`);
        if (!resp.ok) return;
        const { labels } = await resp.json();
        const prev = refLabelName.value;
        refLabelName.innerHTML = '<option value="">-- select label --</option>';
        (labels || []).forEach(l => {
            const opt = document.createElement("option");
            opt.value = l;
            opt.textContent = l;
            refLabelName.appendChild(opt);
        });
        // Restore previous selection if still available
        if (prev && labels.includes(prev)) refLabelName.value = prev;
    } catch (e) {
        console.warn("Could not load model labels:", e);
    }
}

// Load settings from server on startup
async function loadSettings() {
    try {
        const resp = await fetch(`${API_BASE}/api/settings`);
        if (!resp.ok) return;
        const s = await resp.json();
        measurementEnabled.checked = s.enabled ?? false;
        showEdgeDistances.checked = s.show_edge_distances ?? true;
        showCenterDistances.checked = s.show_center_distances ?? true;
        showAlignedPairDistances.checked = s.show_aligned_pair_distances ?? false;
        measurementMethod.value = s.method ?? "camera_intrinsics";
        // Camera intrinsics
        sensorWidth.value = s.sensor_width_mm ?? 6.17;
        focalLength.value = s.focal_length_mm ?? 4.0;
        objectDistance.value = s.object_distance_mm ?? 300;
        // Reference label
        refKnownDim.value = s.reference_known_dimension_mm ?? 10;
        refDimType.value = s.reference_dimension_type ?? "diameter";
        // Manual
        manualPxToMm.value = s.manual_px_to_mm ?? 0.1;
        // Depth
        enableDepthViz.checked = s.show_depth_map ?? false;
        syncDepthPanel();
        // Sync UI
        syncMethodFields();
        syncEnabledState();
        // Now load labels (needs active model) and restore saved reference label
        await loadLabels();
        if (s.reference_label_name) refLabelName.value = s.reference_label_name;
    } catch (e) {
        console.warn("Could not load camera settings:", e);
    }
}

// Save settings to server
async function saveSettings(showToast = true) {
    const payload = {
        enabled: measurementEnabled.checked,
        show_edge_distances: showEdgeDistances.checked,
        show_center_distances: showCenterDistances.checked,
        show_aligned_pair_distances: showAlignedPairDistances.checked,
        method: measurementMethod.value,
        // Camera intrinsics
        sensor_width_mm: parseFloat(sensorWidth.value),
        focal_length_mm: parseFloat(focalLength.value),
        object_distance_mm: parseFloat(objectDistance.value),
        // Reference label
        reference_label_name: refLabelName.value,
        reference_known_dimension_mm: parseFloat(refKnownDim.value),
        reference_dimension_type: refDimType.value,
        // Manual
        manual_px_to_mm: parseFloat(manualPxToMm.value),
        // Depth
        show_depth_map: enableDepthViz.checked,
    };
    try {
        const resp = await fetch(`${API_BASE}/api/settings`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (resp.ok && showToast) {
            settingsSaved.style.display = "inline";
            setTimeout(() => (settingsSaved.style.display = "none"), 2000);
        }
    } catch (e) {
        console.error("Failed to save settings:", e);
    }
}

btnSaveSettings.addEventListener("click", () => saveSettings(true));
if (showEdgeDistances) {
    showEdgeDistances.addEventListener("change", () => saveSettings(false));
}
if (showCenterDistances) {
    showCenterDistances.addEventListener("change", () => saveSettings(false));
}
if (showAlignedPairDistances) {
    showAlignedPairDistances.addEventListener("change", () => saveSettings(false));
}
if (enableDepthViz) {
    enableDepthViz.addEventListener("change", () => {
        saveSettings(false);
        syncDepthPanel();
    });
}

function syncDepthPanel() {
    if (!depthPanel || !depthStream) return;
    const enabled = enableDepthViz.checked;
    depthPanel.style.display = enabled ? "block" : "none";
    if (enabled) {
        depthStream.src = `${API_BASE}/api/mindvision/depth-stream?t=${Date.now()}`;
    } else {
        depthStream.src = "";
    }
}

// Load settings on page load
loadSettings();

// ═══════════════════════════════════════════════════════════════
// SOTA Pipeline Enhancements
// ═══════════════════════════════════════════════════════════════

enhancementsToggle.addEventListener("click", () => {
    const open = enhancementsBody.style.display !== "none";
    enhancementsBody.style.display = open ? "none" : "block";
    enhancementsChevron.textContent = open ? "▸" : "▾";
});

async function loadEnhancements() {
    try {
        const resp = await fetch(`${API_BASE}/api/inference-settings`);
        if (!resp.ok) return;
        const s = await resp.json();
        enableTracking.checked = s.enable_tracking ?? false;
        enableEdgeRefinement.checked = s.enable_edge_refinement ?? false;
    } catch (e) {
        console.warn("Could not load enhancements settings:", e);
    }
}

btnSaveEnhancements.addEventListener("click", async () => {
    const payload = {
        enable_tracking: enableTracking.checked,
        enable_edge_refinement: enableEdgeRefinement.checked
    };
    try {
        const resp = await fetch(`${API_BASE}/api/inference-settings`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (resp.ok) {
            enhancementsSaved.style.display = "inline";
            setTimeout(() => (enhancementsSaved.style.display = "none"), 2000);
        }
    } catch (e) {
        console.error("Failed to apply enhancements:", e);
    }
});

loadEnhancements();

// ═══════════════════════════════════════════════════════════════
// Inference Health — Sparkline & Alert Feed
// ═══════════════════════════════════════════════════════════════

const LATENCY_BUFFER_SIZE = 60;
const latencyBuffer = [];
const healthMinEl = document.getElementById("healthMinLatency");
const healthMaxEl = document.getElementById("healthMaxLatency");
const healthP95El = document.getElementById("healthP95Latency");
const alertFeedEl = document.getElementById("alertFeed");
const sparkCanvas = document.getElementById("latencySparkline");
let sparkCtx = null;

if (sparkCanvas) {
    sparkCtx = sparkCanvas.getContext("2d");
}

function pushLatency(ms) {
    latencyBuffer.push(ms);
    if (latencyBuffer.length > LATENCY_BUFFER_SIZE) {
        latencyBuffer.shift();
    }
}

function updateHealthStats() {
    if (latencyBuffer.length === 0) return;

    const sorted = [...latencyBuffer].sort((a, b) => a - b);
    const minVal = sorted[0];
    const maxVal = sorted[sorted.length - 1];
    const p95Idx = Math.floor(sorted.length * 0.95);
    const p95Val = sorted[Math.min(p95Idx, sorted.length - 1)];

    if (healthMinEl) healthMinEl.textContent = minVal.toFixed(1) + " ms";
    if (healthMaxEl) healthMaxEl.textContent = maxVal.toFixed(1) + " ms";
    if (healthP95El) healthP95El.textContent = p95Val.toFixed(1) + " ms";
}

function drawSparkline() {
    if (!sparkCtx || !sparkCanvas || latencyBuffer.length < 2) return;

    // Resolve actual pixel dimensions (handle DPR)
    const rect = sparkCanvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width - 16; // account for padding
    const h = rect.height - 16;
    sparkCanvas.width = w * dpr;
    sparkCanvas.height = h * dpr;
    sparkCtx.scale(dpr, dpr);

    const data = latencyBuffer;
    const maxVal = Math.max(...data) * 1.1 || 1;
    const minVal = Math.min(...data) * 0.9 || 0;
    const range = maxVal - minVal || 1;
    const stepX = w / (LATENCY_BUFFER_SIZE - 1);

    // Clear
    sparkCtx.clearRect(0, 0, w, h);

    // Fill gradient
    const grad = sparkCtx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(123, 104, 238, 0.25)");
    grad.addColorStop(1, "rgba(123, 104, 238, 0.02)");

    sparkCtx.beginPath();
    sparkCtx.moveTo(0, h);
    data.forEach((val, i) => {
        const x = i * stepX;
        const y = h - ((val - minVal) / range) * h;
        sparkCtx.lineTo(x, y);
    });
    sparkCtx.lineTo((data.length - 1) * stepX, h);
    sparkCtx.closePath();
    sparkCtx.fillStyle = grad;
    sparkCtx.fill();

    // Line
    sparkCtx.beginPath();
    data.forEach((val, i) => {
        const x = i * stepX;
        const y = h - ((val - minVal) / range) * h;
        if (i === 0) sparkCtx.moveTo(x, y);
        else sparkCtx.lineTo(x, y);
    });
    sparkCtx.strokeStyle = "#7B68EE";
    sparkCtx.lineWidth = 1.5;
    sparkCtx.lineJoin = "round";
    sparkCtx.stroke();

    // Latest point dot
    const lastX = (data.length - 1) * stepX;
    const lastY = h - ((data[data.length - 1] - minVal) / range) * h;
    sparkCtx.beginPath();
    sparkCtx.arc(lastX, lastY, 3, 0, Math.PI * 2);
    sparkCtx.fillStyle = "#7B68EE";
    sparkCtx.fill();
}

function updateAlertFeed(alerts) {
    if (!alertFeedEl) return;

    // De-duplicate by message text within the last batch
    const seen = new Set();
    const unique = [];
    for (const a of alerts) {
        const key = a.msg;
        if (!seen.has(key)) {
            seen.add(key);
            unique.push(a);
        }
    }

    unique.forEach(alert => {
        const item = document.createElement("div");
        item.className = "alert-item";

        const badge = document.createElement("span");
        badge.className = `alert-badge ${alert.level || "info"}`;

        const msg = document.createElement("span");
        msg.className = "alert-msg";
        msg.textContent = alert.msg;

        const timeEl = document.createElement("span");
        timeEl.className = "alert-time";
        const d = alert.ts ? new Date(alert.ts * 1000) : new Date();
        timeEl.textContent = d.toLocaleTimeString();

        item.appendChild(badge);
        item.appendChild(msg);
        item.appendChild(timeEl);

        alertFeedEl.prepend(item);
    });

    // Cap at 30 items
    while (alertFeedEl.children.length > 30) {
        alertFeedEl.removeChild(alertFeedEl.lastChild);
    }
}

// Polling fallback for alerts when DataChannel is not available
setInterval(async () => {
    if (dataChannel && dataChannel.readyState === "open") return; // DC handles it
    try {
        const res = await fetch("/api/inference-alerts");
        if (!res.ok) return;
        const data = await res.json();
        if (data.alerts && data.alerts.length > 0) {
            updateAlertFeed(data.alerts);
        }
    } catch (e) {
        // silently ignore
    }
}, 3000);
