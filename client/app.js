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
const domainSelect = document.getElementById("domainSelect");
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

// SOTA Enhancements & Post-processing
const enhancementsToggle = document.getElementById("enhancementsToggle");
const enhancementsBody = document.getElementById("enhancementsBody");
const enhancementsChevron = document.getElementById("enhancementsChevron");
const enableTracking = document.getElementById("enableTracking");
const enableEdgeRefinement = document.getElementById("enableEdgeRefinement");
const enablePostprocessing = document.getElementById("enablePostprocessing");
const enableHeuristic = document.getElementById("enableHeuristic");
const enableTopN = document.getElementById("enableTopN");
const drawBoxes = document.getElementById("drawBoxes");
const drawMasks = document.getElementById("drawMasks");
const drawLabels = document.getElementById("drawLabels");
const enableFileColorValidation = document.getElementById("enableFileColorValidation");
const confThreshold = document.getElementById("confThreshold");
const confValue = document.getElementById("confValue");
const rowHeuristic = document.getElementById("rowHeuristic");
const rowFileColorValidation = document.getElementById("rowFileColorValidation");
const btnSaveEnhancements = document.getElementById("btnSaveEnhancements");
const enhancementsSaved = document.getElementById("enhancementsSaved");
const settingsDomainContext = document.getElementById("settingsDomainContext");

// ═══════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    loadModels();
    if(btnConnect) btnConnect.addEventListener("click", connect);
    if(btnDisconnect) btnDisconnect.addEventListener("click", disconnect);
    if(btnSnapshot) btnSnapshot.addEventListener("click", takeSnapshot);
    if(btnSaveLive) btnSaveLive.addEventListener("click", saveLiveDetection);
    modelSelect.addEventListener("change", onModelChange);
    domainSelect.addEventListener("change", async () => {
        await loadModels();
        // Sync setting selector with active domain
        if (settingsDomainContext) {
            settingsDomainContext.value = domainSelect.value;
            loadEnhancements();
        }
        // Sync MindVision camera domain so live frames use the right model
        fetch("/api/mindvision/set-domain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ domain: domainSelect.value }),
        }).catch(() => {});
    });

    if (settingsDomainContext) {
        settingsDomainContext.addEventListener("change", loadEnhancements);
    }
    initUpload();
    if(confThreshold && confValue) {
        confThreshold.addEventListener("input", () => {
            confValue.textContent = confThreshold.value;
        });
    }
});

// ═══════════════════════════════════════════════════════════════
// Model Management
// ═══════════════════════════════════════════════════════════════

async function loadModels() {
    try {
        const res = await fetch("/api/models?domain=" + domainSelect.value);
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

        // Hide stator-specific settings if in chignon domain
        if (rowHeuristic) {
            rowHeuristic.style.display = domainSelect.value === "chignon" ? "none" : "block";
        }
        if (rowFileColorValidation) {
            rowFileColorValidation.style.display = domainSelect.value === "file" ? "block" : "none";
        }
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
        // Also drive the FPS KPI card from live metrics (works for both MindVision and WebRTC)
        if (statFPS && data.throughput_fps != null) statFPS.textContent = data.throughput_fps.toFixed(1);
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
            body: JSON.stringify({ model, domain: domainSelect.value }),
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
                domain: domainSelect.value,
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

    // Open adjustment modal; detection runs when user clicks "Apply & Detect"
    const objectURL = URL.createObjectURL(file);

    openImageAdjust(objectURL, async (adjustedBlob) => {
        URL.revokeObjectURL(objectURL);
        // Show the adjusted image as "original" panel
        const previewBlob = adjustedBlob || file;
        uploadOriginal.src = URL.createObjectURL(previewBlob);
        await runUploadDetection(previewBlob);
    }, () => {
        // Cancelled — clean up object URL and stay on dropzone
        URL.revokeObjectURL(objectURL);
    });
}

async function runUploadDetection(fileOrBlob) {
    // Show loading
    dropzone.style.display = "none";
    uploadPreview.style.display = "none";
    uploadLoading.style.display = "flex";

    // Send to server
    const imgFile = fileOrBlob instanceof File
        ? fileOrBlob
        : new File([fileOrBlob], "adjusted.jpg", { type: "image/jpeg" });
    const formData = new FormData();
    formData.append("image", imgFile);
    formData.append("domain", domainSelect.value);

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
        const uploadAlerts = (data.alerts || []).map(a => a.msg).filter(Boolean);
        const positionMessage = data.position_message || uploadAlerts[uploadAlerts.length - 1] || "";
        const positionHtml = positionMessage
            ? `<div class="detection-item"><span class="detection-class">${positionMessage}</span></div>`
            : "";

        if (data.detections && data.detections.length > 0) {
            uploadDetections.innerHTML = positionHtml + data.detections
                .map((d) => {
                    const conf = d.confidence ?? 0;
                    const confClass =
                        conf >= 0.75 ? "high" : conf >= 0.5 ? "medium" : "low";
                    const label = (d.class_name ?? `Class ${d.class_id}`).replace(/_/g, ' ');
                    const colorTag = d.file_color
                        ? `<span class="measurement-tag">Color: ${d.file_color}</span>`
                        : "";
                    return `
                        <div class="detection-item">
                            <span class="detection-class">${label}</span>
                            <span class="detection-conf ${confClass}">
                                ${(conf * 100).toFixed(1)}%
                            </span>
                            ${colorTag}
                            ${d.approx_depth_mm ? `<span class="measurement-tag depth-tag">Depth: ~${d.approx_depth_mm} mm</span>` : ""}
                            ${renderMeasurementHtml(d.measurements)}
                        </div>`;
                })
                .join("");
        } else {
            uploadDetections.innerHTML = positionHtml || '<p class="placeholder">No objects detected</p>';
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
    const _fia = document.getElementById("fileInputAdjust");
    if (_fia) _fia.value = "";
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
// Dark Mode Toggle
// ═══════════════════════════════════════════════════════════════
(function initTheme() {
    const btn = document.getElementById("themeToggle");
    if (!btn) return;
    btn.addEventListener("click", () => {
        const isDark = document.documentElement.classList.toggle("dark");
        localStorage.setItem("theme", isDark ? "dark" : "light");
    });
})();

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
const showOppositeDistances = document.getElementById("showOppositeDistances");
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
    if (showOppositeDistances) showOppositeDistances.disabled = !on;
}

measurementMethod.addEventListener("change", syncMethodFields);
measurementEnabled.addEventListener("change", syncEnabledState);

// Cross-diametric mode is exclusive: it should draw only opposite lines.
if (showOppositeDistances) {
    showOppositeDistances.addEventListener("change", () => {
        if (!showOppositeDistances.checked) return;
        if (showEdgeDistances) showEdgeDistances.checked = false;
        if (showCenterDistances) showCenterDistances.checked = false;
        if (showAlignedPairDistances) showAlignedPairDistances.checked = false;
    });
}

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
            opt.textContent = l.replace(/_/g, " ");
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
        if(showOppositeDistances) showOppositeDistances.checked = s.show_opposite_distances ?? false;
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
        show_opposite_distances: showOppositeDistances ? showOppositeDistances.checked : false,
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
if (showOppositeDistances) {
    showOppositeDistances.addEventListener("change", () => saveSettings(false));
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
    const domain = settingsDomainContext ? settingsDomainContext.value : "stator";
    try {
        const resp = await fetch(`${API_BASE}/api/inference-enhancements?domain=${domain}`);
        if (!resp.ok) return;
        const s = await resp.json();
        if (enableTracking) enableTracking.checked = s.enable_tracking ?? false;
        if (enableEdgeRefinement) enableEdgeRefinement.checked = s.enable_edge_refinement ?? false;
        if (enablePostprocessing) enablePostprocessing.checked = s.enable_postprocessing ?? true;
        if (enableHeuristic) enableHeuristic.checked = s.enable_heuristic ?? true;
        if (enableTopN) enableTopN.checked = s.enable_top_n ?? true;
        if (drawBoxes) drawBoxes.checked = s.draw_boxes ?? true;
        if (drawMasks) drawMasks.checked = s.draw_masks ?? true;
        if (drawLabels) drawLabels.checked = s.draw_labels ?? true;
        if (enableFileColorValidation) {
            enableFileColorValidation.checked = s.enable_file_color_validation ?? true;
        }
        // Hide/Show heuristic based on domain (Stator only)
        if (rowHeuristic) {
            rowHeuristic.style.display = domain === "stator" ? "block" : "none";
        }
        if (rowFileColorValidation) {
            rowFileColorValidation.style.display = domain === "file" ? "block" : "none";
        }

        if (confThreshold) {
            confThreshold.value = s.conf_threshold ?? 0.05;
            confValue.textContent = confThreshold.value;
        }
    } catch (e) {
        console.warn(`Could not load enhancements settings for ${domain}:`, e);
    }
}

btnSaveEnhancements.addEventListener("click", async () => {
    const domain = settingsDomainContext ? settingsDomainContext.value : "stator";
    const payload = {
        enable_tracking: enableTracking.checked,
        enable_edge_refinement: enableEdgeRefinement.checked,
        enable_postprocessing: enablePostprocessing.checked,
        enable_heuristic: enableHeuristic.checked,
        enable_top_n: enableTopN.checked,
        draw_boxes: drawBoxes.checked,
        draw_masks: drawMasks.checked,
        draw_labels: drawLabels.checked,
        enable_file_color_validation: enableFileColorValidation ? enableFileColorValidation.checked : true,
        conf_threshold: parseFloat(confThreshold.value)
    };
    try {
        const resp = await fetch(`${API_BASE}/api/inference-enhancements?domain=${domain}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (resp.ok) {
            enhancementsSaved.style.display = "inline";
            setTimeout(() => (enhancementsSaved.style.display = "none"), 2000);
        }
    } catch (e) {
        console.error(`Failed to apply enhancements for ${domain}:`, e);
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

// ═══════════════════════════════════════════════════════════════
// Dashboard — Page Navigation
// ═══════════════════════════════════════════════════════════════

(function initNav() {
    const burgerBtn     = document.getElementById("burgerBtn");
    const drawer        = document.getElementById("drawer");
    const drawerOverlay = document.getElementById("drawerOverlay");
    const drawerClose   = document.getElementById("drawerClose");
    const navItems      = document.querySelectorAll(".nav-item[data-page]");
    const pages         = document.querySelectorAll(".page[id^='page-']");

    // ── Drawer open/close ──────────────────────────────────
    function openDrawer() {
        drawer.classList.add("open");
        drawerOverlay.classList.add("open");
        burgerBtn.classList.add("open");
        document.body.style.overflow = "hidden";
    }

    function closeDrawer() {
        drawer.classList.remove("open");
        drawerOverlay.classList.remove("open");
        burgerBtn.classList.remove("open");
        document.body.style.overflow = "";
    }

    if (burgerBtn)     burgerBtn.addEventListener("click", openDrawer);
    if (drawerClose)   drawerClose.addEventListener("click", closeDrawer);
    if (drawerOverlay) drawerOverlay.addEventListener("click", closeDrawer);

    // Close on Escape
    document.addEventListener("keydown", e => { if (e.key === "Escape") closeDrawer(); });

    // ── Page switching ────────────────────────────────────
    function showPage(pageId) {
        pages.forEach(p => {
            p.classList.toggle("active", p.id === "page-" + pageId);
            p.classList.toggle("hidden", p.id !== "page-" + pageId);
        });
        navItems.forEach(n => {
            n.classList.toggle("active", n.dataset.page === pageId);
        });
        closeDrawer();
        if (pageId === "dashboard") loadPastSessions();
    }

    navItems.forEach(btn => btn.addEventListener("click", () => showPage(btn.dataset.page)));

    // ── Status dot mirrors ────────────────────────────────
    const topDot   = document.getElementById("sidebarDot");   // topbar dot
    const topMvDot = document.getElementById("mvDot");        // topbar mv dot
    const drwDot   = document.getElementById("drawerDot");
    const drwMvDot = document.getElementById("drawerMvDot");
    const drwRtc   = document.getElementById("drawerRtcStatus");
    const drwMvSt  = document.getElementById("drawerMvStatus");
    const mvBadge  = document.getElementById("mvStatusBadge");

    function syncRtcDots(status) {
        [topDot, drwDot].forEach(d => {
            if (!d) return;
            d.className = "status-dot";
            if (status === "connected")  d.classList.add("connected");
            if (status === "connecting") d.classList.add("connecting");
        });
        if (drwRtc) drwRtc.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        const badge = document.getElementById("statusBadge");
        if (badge) badge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }

    // Patch setStatus used by WebRTC code
    const _origSet = window.setStatus;
    if (typeof _origSet === "function") {
        window.setStatus = function(s) { _origSet(s); syncRtcDots(s); };
    }

    // Watch mvStatusBadge text for camera connection state
    if (mvBadge) {
        new MutationObserver(() => {
            const connected = mvBadge.textContent.toLowerCase().includes("connected");
            [topMvDot, drwMvDot].forEach(d => {
                if (!d) return;
                d.className = "status-dot mv-dot";
                if (connected) d.classList.add("connected");
            });
            if (drwMvSt) drwMvSt.textContent = connected ? "Connected" : "Off";
        }).observe(mvBadge, { childList: true, characterData: true, subtree: true });
    }
})();

// ═══════════════════════════════════════════════════════════════
// Dashboard — Tab Switching (Live Camera / Upload)
// ═══════════════════════════════════════════════════════════════

(function initTabs() {
    const tabBtns  = document.querySelectorAll(".tab-btn[data-tab]");
    const tabPanes = document.querySelectorAll(".tab-pane[id^='tab-']");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.toggle("active", b.dataset.tab === target));
            tabPanes.forEach(p => {
                p.classList.toggle("active", p.id === "tab-" + target);
                p.classList.toggle("hidden", p.id !== "tab-" + target);
            });
        });
    });
})();

// ═══════════════════════════════════════════════════════════════
// Dashboard — Domain Switcher
// ═══════════════════════════════════════════════════════════════

(function initDomainSwitcher() {
    const switcher = document.getElementById("domainSwitcher");
    if (!switcher) return;

    const btns = switcher.querySelectorAll(".domain-btn[data-domain]");

    // Sync a pill's active state from the hidden domainSelect value
    function syncActive(value) {
        btns.forEach(b => b.classList.toggle("active", b.dataset.domain === value));
    }

    btns.forEach(btn => {
        btn.addEventListener("click", () => {
            const domain = btn.dataset.domain;
            if (!domainSelect) return;

            domainSelect.value = domain;
            syncActive(domain);

            // Sync MindVision camera domain so live frames use the right model
            fetch("/api/mindvision/set-domain", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ domain }),
            }).catch(() => {});

            // If WebRTC is connected, reconnect so the new domain takes effect
            if (pc && pc.connectionState === "connected") {
                disconnect().then(() => connect());
            }

            // Trigger the same logic as the Settings select
            loadModels();
            if (settingsDomainContext) {
                settingsDomainContext.value = domain;
                loadEnhancements();
            }
        });
    });

    // Keep switcher in sync if Settings domainSelect is changed separately
    if (domainSelect) {
        domainSelect.addEventListener("change", () => syncActive(domainSelect.value));
    }

    // Init active state from current domainSelect value
    if (domainSelect) syncActive(domainSelect.value);
})();

// ═══════════════════════════════════════════════════════════════
// Dashboard — Mode Switcher (Normal vs Inspection)
// ═══════════════════════════════════════════════════════════════

(function initModeSwitcher() {
    const switcher = document.getElementById("modeSwitcher");
    if (!switcher) return;

    const btns = switcher.querySelectorAll(".domain-btn[data-mode]");
    const domainSwitcher = document.getElementById("domainSwitcher");
    const btnRunInspection = document.getElementById("btnRunInspection");
    const inspectionPanel = document.getElementById("inspectionPanel");

    function applyMode(mode) {
        btns.forEach(b => b.classList.toggle("active", b.dataset.mode === mode));

        const isInspection = mode === "inspection";
        if (domainSwitcher) domainSwitcher.classList.toggle("disabled", isInspection);
        if (btnRunInspection) btnRunInspection.style.display = isInspection ? "" : "none";
        if (!isInspection && inspectionPanel) inspectionPanel.style.display = "none";
    }

    btns.forEach(btn => {
        btn.addEventListener("click", () => applyMode(btn.dataset.mode));
    });

    // Default to Normal
    applyMode("normal");

    // Expose so inspection.js can re-enable domain switcher when a run finishes
    window.setDetectionMode = applyMode;
})();

// ═══════════════════════════════════════════════════════════════
// Dashboard — Past Sessions
// ═══════════════════════════════════════════════════════════════

async function loadPastSessions() {
    const container = document.getElementById("pastSessionsList");
    if (!container) return;

    try {
        const res = await fetch("/api/inference-logs?n=200");
        if (!res.ok) throw new Error("not ok");
        const data = await res.json();
        const entries = data.entries || [];

        if (entries.length === 0) {
            container.innerHTML = '<p class="placeholder" style="padding:20px 0;">No sessions recorded yet.</p>';
            return;
        }

        // Aggregate entries into pseudo-sessions by gaps > 5 min
        const GAP_S = 300;
        const sessions = [];
        let cur = null;

        for (const e of entries) {
            if (!cur || (e.ts - cur.lastTs) > GAP_S) {
                if (cur) sessions.push(cur);
                cur = { startTs: e.ts, lastTs: e.ts, model: e.model, entries: [] };
            }
            cur.lastTs = e.ts;
            cur.entries.push(e);
        }
        if (cur) sessions.push(cur);
        sessions.reverse(); // newest first

        container.innerHTML = sessions.map(s => {
            const count  = s.entries.length;
            const avgLat = s.entries.reduce((a, e) => a + (e.latency_ms || 0), 0) / count;
            const avgDet = s.entries.reduce((a, e) => a + (e.detections || 0), 0) / count;
            const avgConf = s.entries.reduce((a, e) => a + (e.avg_conf || 0), 0) / count;
            const start  = new Date(s.startTs * 1000);
            const durationMin = Math.round((s.lastTs - s.startTs) / 60);
            const timeStr = start.toLocaleDateString(undefined, { month: "short", day: "numeric" })
                          + " " + start.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });

            return `
                <div class="session-card">
                    <div class="session-card-header">
                        <span class="session-model">${s.model || "Unknown model"}</span>
                        <span class="session-time">${timeStr}</span>
                    </div>
                    <div class="session-stats">
                        <div class="session-stat">
                            <span class="session-stat-val">${avgLat.toFixed(0)}</span>
                            <span class="session-stat-lbl">Avg latency ms</span>
                        </div>
                        <div class="session-stat">
                            <span class="session-stat-val">${avgDet.toFixed(1)}</span>
                            <span class="session-stat-lbl">Avg detections</span>
                        </div>
                        <div class="session-stat">
                            <span class="session-stat-val">${(avgConf * 100).toFixed(1)}%</span>
                            <span class="session-stat-lbl">Avg confidence</span>
                        </div>
                        <div class="session-stat">
                            <span class="session-stat-val">${count}</span>
                            <span class="session-stat-lbl">Frames</span>
                        </div>
                    </div>
                    <span class="session-badge">
                        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                        ${durationMin < 1 ? "< 1 min" : durationMin + " min"}
                    </span>
                </div>`;
        }).join("");

    } catch (e) {
        container.innerHTML = '<p class="placeholder" style="padding:20px 0;">Could not load sessions.</p>';
    }
}

// Refresh button
const _btnRefresh = document.getElementById("btnRefreshSessions");
if (_btnRefresh) _btnRefresh.addEventListener("click", loadPastSessions);

// Load on startup
document.addEventListener("DOMContentLoaded", loadPastSessions);

// ═══════════════════════════════════════════════════════════════
// Settings — MindVision Camera Toggle
// ═══════════════════════════════════════════════════════════════

(function initMvCameraToggle() {
    const toggle = document.getElementById("mvCameraToggle");
    const statusEl = document.getElementById("mvCameraStatus");
    if (!toggle || !statusEl) return;

    function setStatus(running, transitioning) {
        if (transitioning) {
            statusEl.textContent = "Starting…";
            statusEl.className = "mv-camera-status starting";
        } else if (running) {
            statusEl.textContent = "Running";
            statusEl.className = "mv-camera-status running";
        } else {
            statusEl.textContent = "Stopped";
            statusEl.className = "mv-camera-status";
        }
    }

    // Poll proc-status once to sync the toggle with server reality
    async function syncState() {
        try {
            const res = await fetch("/api/mindvision/proc-status");
            if (!res.ok) return;
            const data = await res.json();
            toggle.checked = data.running;
            setStatus(data.running, false);
        } catch { /* server may not be up yet */ }
    }

    toggle.addEventListener("change", async () => {
        const want = toggle.checked;
        setStatus(false, want);   // show "Starting…" when turning on
        try {
            const res = await fetch(want ? "/api/mindvision/start" : "/api/mindvision/stop", {
                method: "POST",
            });
            const data = await res.json();
            if (!data.ok) {
                console.warn("MindVision toggle failed:", data.msg);
                toggle.checked = !want;  // revert
                setStatus(!want, false);
                return;
            }
            setStatus(data.running, false);
            toggle.checked = data.running;
        } catch (err) {
            console.error("MindVision toggle error:", err);
            toggle.checked = !want;
            setStatus(!want, false);
        }
    });

    // Sync on page load and keep in sync with the live camera status
    syncState();
    // Re-sync every 5s so the toggle reflects process crashes / external stops
    setInterval(syncState, 5000);
})();

// ═══════════════════════════════════════════════════════════════
// Image Adjustment Modal
// ═══════════════════════════════════════════════════════════════

let _iaOnConfirm = null;
let _iaOnCancel  = null;
let _iaSourceImg = new Image();
let _iaRotation  = 0;   // degrees: 0, 90, 180, 270

/**
 * Open the image adjustment modal.
 * @param {string}   src        - Image URL (objectURL or data URL)
 * @param {Function} onConfirm  - Called with (Blob) of the adjusted JPEG
 * @param {Function} [onCancel] - Called when user dismisses without action
 */
function openImageAdjust(src, onConfirm, onCancel) {
    _iaOnConfirm = onConfirm;
    _iaOnCancel  = onCancel || null;

    const overlay = document.getElementById("iaOverlay");
    if (!overlay) return;

    // Reset sliders and rotation to neutral
    ["iaBrightness", "iaContrast", "iaSaturation", "iaSharpness"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = id === "iaSharpness" ? 0 : 100;
    });
    _iaRotation = 0;
    _iaUpdateLabels();

    _iaSourceImg = new Image();
    _iaSourceImg.crossOrigin = "anonymous";
    _iaSourceImg.onload = () => {
        _iaRender();
        overlay.classList.add("open");
        overlay.setAttribute("aria-hidden", "false");
    };
    _iaSourceImg.onerror = () => {
        // Fallback: just confirm with null so detection runs on the original
        if (_iaOnConfirm) _iaOnConfirm(null);
    };
    _iaSourceImg.src = src;
}

function _iaClose(confirmed) {
    const overlay = document.getElementById("iaOverlay");
    if (overlay) {
        overlay.classList.remove("open");
        overlay.setAttribute("aria-hidden", "true");
    }
    if (confirmed) {
        const canvas = document.getElementById("iaCanvas");
        canvas.toBlob(blob => {
            if (_iaOnConfirm) _iaOnConfirm(blob);
        }, "image/jpeg", 0.95);
    } else {
        if (_iaOnCancel) _iaOnCancel();
    }
}

function _iaRender() {
    const canvas = document.getElementById("iaCanvas");
    if (!canvas || !_iaSourceImg.naturalWidth) return;
    const ctx = canvas.getContext("2d");

    const sw = _iaSourceImg.naturalWidth;
    const sh = _iaSourceImg.naturalHeight;
    const rotated90 = _iaRotation === 90 || _iaRotation === 270;

    canvas.width  = rotated90 ? sh : sw;
    canvas.height = rotated90 ? sw : sh;

    const b = document.getElementById("iaBrightness")?.value ?? 100;
    const c = document.getElementById("iaContrast")?.value   ?? 100;
    const s = document.getElementById("iaSaturation")?.value ?? 100;

    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate((_iaRotation * Math.PI) / 180);
    ctx.filter = `brightness(${b}%) contrast(${c}%) saturate(${s}%)`;
    ctx.drawImage(_iaSourceImg, -sw / 2, -sh / 2);
    ctx.filter = "none";
    ctx.restore();

    // Simple unsharp mask via sharpness slider
    const sharp = parseFloat(document.getElementById("iaSharpness")?.value ?? 0);
    if (sharp > 0) {
        _iaApplyUnsharpMask(ctx, canvas, sharp / 100);
    }
}

function _iaApplyUnsharpMask(ctx, canvas, amount) {
    // Fast unsharp: draw blurred copy then blend
    const w = canvas.width, h = canvas.height;
    const offscreen = document.createElement("canvas");
    offscreen.width = w; offscreen.height = h;
    const octx = offscreen.getContext("2d");
    octx.filter = "blur(1.5px)";
    octx.drawImage(canvas, 0, 0);
    octx.filter = "none";

    const orig = ctx.getImageData(0, 0, w, h);
    const blur = octx.getImageData(0, 0, w, h);
    const out  = ctx.createImageData(w, h);
    const a    = amount * 1.5;
    for (let i = 0; i < orig.data.length; i += 4) {
        out.data[i]   = Math.min(255, Math.max(0, orig.data[i]   + a * (orig.data[i]   - blur.data[i])));
        out.data[i+1] = Math.min(255, Math.max(0, orig.data[i+1] + a * (orig.data[i+1] - blur.data[i+1])));
        out.data[i+2] = Math.min(255, Math.max(0, orig.data[i+2] + a * (orig.data[i+2] - blur.data[i+2])));
        out.data[i+3] = orig.data[i+3];
    }
    ctx.putImageData(out, 0, 0);
}

function _iaUpdateLabels() {
    const pairs = [
        ["iaBrightness", "iaBrightnessVal", "%"],
        ["iaContrast",   "iaContrastVal",   "%"],
        ["iaSaturation", "iaSaturationVal", "%"],
        ["iaSharpness",  "iaSharpnessVal",  ""],
    ];
    pairs.forEach(([sliderId, labelId, suffix]) => {
        const s = document.getElementById(sliderId);
        const l = document.getElementById(labelId);
        if (s && l) l.textContent = s.value + suffix;
    });
    const rotLabel = document.getElementById("iaRotationVal");
    if (rotLabel) rotLabel.textContent = _iaRotation + "°";
}

(function initImageAdjust() {
    // Wire sliders → re-render
    ["iaBrightness", "iaContrast", "iaSaturation", "iaSharpness"].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener("input", () => {
            _iaUpdateLabels();
            _iaRender();
        });
    });

    document.getElementById("iaReset")?.addEventListener("click", () => {
        ["iaBrightness", "iaContrast", "iaSaturation"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.value = 100;
        });
        const sharp = document.getElementById("iaSharpness");
        if (sharp) sharp.value = 0;
        _iaRotation = 0;
        _iaUpdateLabels();
        _iaRender();
    });

    document.getElementById("iaRotLeft")?.addEventListener("click", () => {
        _iaRotation = (_iaRotation - 90 + 360) % 360;
        _iaUpdateLabels();
        _iaRender();
    });

    document.getElementById("iaRotRight")?.addEventListener("click", () => {
        _iaRotation = (_iaRotation + 90) % 360;
        _iaUpdateLabels();
        _iaRender();
    });

    document.getElementById("iaDetect")?.addEventListener("click", () => _iaClose(true));
    document.getElementById("iaClose")?.addEventListener("click",  () => _iaClose(false));
    document.getElementById("iaCancel")?.addEventListener("click", () => _iaClose(false));

    // Close on overlay backdrop click
    document.getElementById("iaOverlay")?.addEventListener("click", (e) => {
        if (e.target === document.getElementById("iaOverlay")) _iaClose(false);
    });

    // Escape key
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && document.getElementById("iaOverlay")?.classList.contains("open")) {
            _iaClose(false);
        }
    });

    // "Adjust Before Detect" button in dropzone — opens file picker
    const btnAdjOpen = document.getElementById("btnAdjustOpen");
    const fileInputAdjust = document.getElementById("fileInputAdjust");
    if (btnAdjOpen && fileInputAdjust) {
        btnAdjOpen.addEventListener("click", (e) => {
            e.stopPropagation(); // prevent dropzone's own click handler
            fileInputAdjust.click();
        });
        fileInputAdjust.addEventListener("change", () => {
            if (fileInputAdjust.files.length > 0) {
                handleUpload(fileInputAdjust.files[0]);
            }
        });
    }

    // "Adjust Frame" button in MindVision meta row
    document.getElementById("btnAdjustMV")?.addEventListener("click", () => {
        const mvImg = document.getElementById("mvStream");
        if (!mvImg || !mvImg.src || mvImg.src === window.location.href) return;

        openImageAdjust(mvImg.src, async (blob) => {
            if (!blob) return;
            const formData = new FormData();
            formData.append("image", new File([blob], "mv_frame.jpg", { type: "image/jpeg" }));
            formData.append("domain", domainSelect ? domainSelect.value : "stator");
            try {
                const res = await fetch("/api/detect", { method: "POST", body: formData });
                const data = await res.json();
                if (data.image) {
                    // Show result in a temporary overlay on the stream
                    mvImg.src = data.image;
                }
            } catch (err) {
                console.error("MV adjust detect failed:", err);
            }
        });
    });
})();
