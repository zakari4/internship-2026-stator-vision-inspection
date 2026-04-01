/**
 * Measure Tool — Two-Point Calibration & Distance Measurement
 *
 * Provides a reusable overlay that lets users click two points on an
 * image, see the pixel distance, calibrate with a known real-world
 * distance, and then measure further distances in millimeters.
 *
 * Usage:
 *   const mt = new MeasureTool(imageElement, { onCalibrate: (pxToMm) => {} });
 *   mt.activate();   // enter drawing mode
 *   mt.deactivate(); // exit drawing mode
 */

(function () {
    "use strict";

    class MeasureTool {
        /**
         * @param {HTMLImageElement} imgEl  — the image to measure on
         * @param {Object} opts
         * @param {Function} opts.onCalibrate — called with (pxToMm) after calibration
         */
        constructor(imgEl, opts = {}) {
            this.imgEl = imgEl;
            this.onCalibrate = opts.onCalibrate || (() => {});
            this.pxToMm = 0;          // 0 = not calibrated
            this.points = [];          // [{x, y}] in image-space coords
            this.lines = [];           // completed pairs [{p1, p2, pxDist, mmDist?}]
            this.active = false;

            // Build DOM
            this._buildOverlay();
            this._buildToolbar();
        }

        /* ── DOM Construction ──────────────────────────────────── */

        _buildOverlay() {
            // Wrap the image in a relative container if not already
            const parent = this.imgEl.parentElement;
            let wrapper = parent;
            if (!parent.classList.contains("measure-wrapper")) {
                wrapper = document.createElement("div");
                wrapper.className = "measure-wrapper";
                parent.insertBefore(wrapper, this.imgEl);
                wrapper.appendChild(this.imgEl);
            }
            this.wrapper = wrapper;

            // Canvas overlay
            this.canvas = document.createElement("canvas");
            this.canvas.className = "measure-canvas";
            this.wrapper.appendChild(this.canvas);
            this.ctx = this.canvas.getContext("2d");

            // Click handler
            this.canvas.addEventListener("click", (e) => this._onClick(e));
        }

        _buildToolbar() {
            const tb = document.createElement("div");
            tb.className = "measure-toolbar";
            tb.style.display = "none";
            tb.innerHTML = `
                <div class="measure-toolbar-row">
                    <span class="measure-status" id="mStatus_${this._uid()}">Click two points on the image</span>
                </div>
                <div class="measure-toolbar-row measure-calibrate-row" style="display:none;">
                    <label>Real distance:</label>
                    <input type="number" class="measure-input" step="0.01" min="0.01" placeholder="mm" />
                    <button class="btn btn-primary btn-sm measure-btn-calibrate">Calibrate</button>
                </div>
                <div class="measure-toolbar-actions">
                    <button class="btn btn-secondary btn-sm measure-btn-clear">Clear</button>
                    <button class="btn btn-danger btn-sm measure-btn-done">Done</button>
                </div>
            `;
            // Insert toolbar after the wrapper
            this.wrapper.parentElement.insertBefore(tb, this.wrapper.nextSibling);
            this.toolbar = tb;

            // References
            this.statusEl = tb.querySelector(".measure-status");
            this.calibrateRow = tb.querySelector(".measure-calibrate-row");
            this.distInput = tb.querySelector(".measure-input");
            this.btnCalibrate = tb.querySelector(".measure-btn-calibrate");
            this.btnClear = tb.querySelector(".measure-btn-clear");
            this.btnDone = tb.querySelector(".measure-btn-done");

            // Events
            this.btnCalibrate.addEventListener("click", () => this._calibrate());
            this.btnClear.addEventListener("click", () => this._clear());
            this.btnDone.addEventListener("click", () => this.deactivate());
        }

        _uid() {
            return Math.random().toString(36).substr(2, 6);
        }

        /* ── Activation ────────────────────────────────────────── */

        activate() {
            this.active = true;
            this.canvas.style.display = "block";
            this.toolbar.style.display = "flex";
            this._syncCanvasSize();
            this._clear();
            this.statusEl.textContent = "Click two points on the image";

            // Observe image size changes
            this._resizeObs = new ResizeObserver(() => this._syncCanvasSize());
            this._resizeObs.observe(this.imgEl);
        }

        deactivate() {
            this.active = false;
            this.canvas.style.display = "none";
            this.toolbar.style.display = "none";
            if (this._resizeObs) this._resizeObs.disconnect();
        }

        _syncCanvasSize() {
            const rect = this.imgEl.getBoundingClientRect();
            this.canvas.width = rect.width;
            this.canvas.height = rect.height;
            this._redraw();
        }

        /* ── Click / Drawing ───────────────────────────────────── */

        _onClick(e) {
            if (!this.active) return;

            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Store in image-pixel coordinates (scale from display to natural)
            const scaleX = this.imgEl.naturalWidth / rect.width;
            const scaleY = this.imgEl.naturalHeight / rect.height;
            const imgX = x * scaleX;
            const imgY = y * scaleY;

            this.points.push({ x: imgX, y: imgY, dispX: x, dispY: y });

            if (this.points.length === 2) {
                const p1 = this.points[0];
                const p2 = this.points[1];
                const pxDist = Math.sqrt(
                    Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2)
                );

                const line = { p1, p2, pxDist, mmDist: null };
                if (this.pxToMm > 0) {
                    line.mmDist = pxDist * this.pxToMm;
                }
                this.lines.push(line);

                // Show calibrate row if not yet calibrated
                if (this.pxToMm === 0) {
                    this.calibrateRow.style.display = "flex";
                    this.statusEl.textContent = `Distance: ${pxDist.toFixed(1)} px — Enter real distance to calibrate`;
                } else {
                    this.statusEl.textContent = `Distance: ${line.mmDist.toFixed(2)} mm (${pxDist.toFixed(1)} px)`;
                }

                // Reset points for next pair
                this.points = [];
            } else {
                this.statusEl.textContent = "Click a second point…";
            }

            this._redraw();
        }

        _redraw() {
            const ctx = this.ctx;
            const w = this.canvas.width;
            const h = this.canvas.height;
            ctx.clearRect(0, 0, w, h);

            const rect = this.imgEl.getBoundingClientRect();
            const scaleX = rect.width / (this.imgEl.naturalWidth || rect.width);
            const scaleY = rect.height / (this.imgEl.naturalHeight || rect.height);

            // Draw completed lines
            this.lines.forEach((line) => {
                const x1 = line.p1.x * scaleX;
                const y1 = line.p1.y * scaleY;
                const x2 = line.p2.x * scaleX;
                const y2 = line.p2.y * scaleY;

                // Dashed line
                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = "#00e5ff";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
                ctx.setLineDash([]);

                // End points
                this._drawPoint(ctx, x1, y1);
                this._drawPoint(ctx, x2, y2);

                // Distance label at midpoint
                const mx = (x1 + x2) / 2;
                const my = (y1 + y2) / 2;
                const label = line.mmDist != null
                    ? `${line.mmDist.toFixed(2)} mm`
                    : `${line.pxDist.toFixed(1)} px`;
                this._drawLabel(ctx, mx, my, label);
            });

            // Draw pending point (first click waiting for second)
            if (this.points.length === 1) {
                const p = this.points[0];
                const dx = p.x * scaleX;
                const dy = p.y * scaleY;
                this._drawPoint(ctx, dx, dy, "#ff4081");
            }
        }

        _drawPoint(ctx, x, y, color = "#00e5ff") {
            // Outer glow
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fillStyle = color + "33";
            ctx.fill();

            // Inner dot
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();

            // White border
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        _drawLabel(ctx, x, y, text) {
            ctx.font = "bold 13px Inter, sans-serif";
            const metrics = ctx.measureText(text);
            const tw = metrics.width + 12;
            const th = 22;

            // Background pill
            const rx = x - tw / 2;
            const ry = y - th - 10;
            ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
            this._roundRect(ctx, rx, ry, tw, th, 6);
            ctx.fill();

            // Border
            ctx.strokeStyle = "#00e5ff";
            ctx.lineWidth = 1;
            this._roundRect(ctx, rx, ry, tw, th, 6);
            ctx.stroke();

            // Text
            ctx.fillStyle = "#fff";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(text, x, ry + th / 2);
        }

        _roundRect(ctx, x, y, w, h, r) {
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.lineTo(x + w - r, y);
            ctx.arcTo(x + w, y, x + w, y + r, r);
            ctx.lineTo(x + w, y + h - r);
            ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
            ctx.lineTo(x + r, y + h);
            ctx.arcTo(x, y + h, x, y + h - r, r);
            ctx.lineTo(x, y + r);
            ctx.arcTo(x, y, x + r, y, r);
            ctx.closePath();
        }

        /* ── Calibration ───────────────────────────────────────── */

        _calibrate() {
            const realMm = parseFloat(this.distInput.value);
            if (!realMm || realMm <= 0) {
                this.statusEl.textContent = "⚠ Enter a valid distance in mm";
                return;
            }

            // Use the last drawn line for calibration
            const lastLine = this.lines[this.lines.length - 1];
            if (!lastLine) {
                this.statusEl.textContent = "⚠ Draw two points first";
                return;
            }

            this.pxToMm = realMm / lastLine.pxDist;

            // Update all existing lines with mm values
            this.lines.forEach((line) => {
                line.mmDist = line.pxDist * this.pxToMm;
            });

            this.calibrateRow.style.display = "none";
            this.statusEl.textContent =
                `✓ Calibrated: 1 px = ${this.pxToMm.toFixed(5)} mm — Draw more points to measure`;

            this._redraw();

            // Persist to server
            this._saveCalibration();

            // Notify callback
            this.onCalibrate(this.pxToMm);
        }

        async _saveCalibration() {
            try {
                await fetch("/api/settings", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        method: "manual",
                        manual_px_to_mm: this.pxToMm,
                        enabled: true,
                    }),
                });
            } catch (e) {
                console.warn("Failed to save calibration:", e);
            }
        }

        /* ── Clear ─────────────────────────────────────────────── */

        _clear() {
            this.points = [];
            this.lines = [];
            this.calibrateRow.style.display = "none";
            this.distInput.value = "";
            this.statusEl.textContent = this.pxToMm > 0
                ? `Calibrated (1 px = ${this.pxToMm.toFixed(5)} mm) — Click two points to measure`
                : "Click two points on the image";
            this._redraw();
        }
    }

    // Export globally
    window.MeasureTool = MeasureTool;
})();
