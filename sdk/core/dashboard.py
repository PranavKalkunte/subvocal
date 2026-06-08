"""Local observability dashboard server for Subvocal Middleware."""

import os
import sys
import json
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer

# Add package paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Subvocal Observability Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(18, 25, 41, 0.6);
            --card-border: rgba(255, 255, 255, 0.05);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --color-primary: #3b82f6;
            --color-success: #10b981;
            --color-warning: #f59e0b;
            --color-danger: #ef4444;
            --font-family: 'Outfit', sans-serif;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: var(--font-family);
            min-height: 100vh;
            overflow-x: hidden;
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.1) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%);
            background-attachment: fixed;
        }

        header {
            padding: 2rem 4rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--card-border);
            backdrop-filter: blur(10px);
            background: rgba(11, 15, 25, 0.5);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        header h1 {
            font-weight: 700;
            font-size: 1.8rem;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, #3b82f6, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        header .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
            background: var(--card-bg);
            padding: 0.5rem 1rem;
            border-radius: 99px;
            border: 1px solid var(--card-border);
        }

        header .dot {
            width: 8px;
            height: 8px;
            background-color: var(--color-success);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--color-success);
        }

        main {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 2rem;
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
        }

        .stat-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(16px);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            border-color: rgba(59, 130, 246, 0.2);
        }

        .stat-card .label {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .stat-card .value {
            font-size: 2rem;
            font-weight: 700;
        }

        .content-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .content-grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 2rem;
            backdrop-filter: blur(16px);
        }

        .card h2 {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            border-left: 4px solid var(--color-primary);
            padding-left: 0.75rem;
        }

        .traces-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            max-height: 600px;
            overflow-y: auto;
            padding-right: 0.5rem;
        }

        .traces-list::-webkit-scrollbar {
            width: 6px;
        }

        .traces-list::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 99px;
        }

        .trace-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            transition: background 0.2s ease;
        }

        .trace-item:hover {
            background: rgba(255, 255, 255, 0.04);
        }

        .trace-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .trace-app {
            font-weight: 600;
            font-size: 0.95rem;
        }

        .trace-time {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .trace-pill {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.25rem 0.75rem;
            border-radius: 99px;
            text-transform: uppercase;
        }

        .status-success {
            background: rgba(16, 185, 129, 0.15);
            color: var(--color-success);
        }

        .status-rejected {
            background: rgba(239, 68, 68, 0.15);
            color: var(--color-danger);
        }

        .status-dry_run {
            background: rgba(245, 158, 11, 0.15);
            color: var(--color-warning);
        }

        .status-error {
            background: rgba(239, 68, 68, 0.15);
            color: var(--color-danger);
        }

        .trace-body {
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
        }

        .trace-body .token-box {
            background: rgba(59, 130, 246, 0.1);
            color: var(--color-primary);
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            font-family: monospace;
        }

        .trace-body .intent-text {
            color: var(--text-secondary);
        }

        .trace-details {
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            display: flex;
            justify-content: space-between;
        }

        .no-data {
            text-align: center;
            padding: 3rem;
            color: var(--text-secondary);
        }

        .chart-container {
            position: relative;
            height: 250px;
            width: 100%;
        }
    </style>
</head>
<body>

    <header>
        <h1>Subvocal Middleware Telemetry</h1>
        <div class="status-indicator">
            <span class="dot"></span>
            <span>Local Node: Active</span>
        </div>
    </header>

    <main>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Total Actions Logs</div>
                <div class="value" id="stat-total">0</div>
            </div>
            <div class="stat-card" style="border-left: 4px solid var(--color-success);">
                <div class="label">Executed Actions</div>
                <div class="value" id="stat-executed" style="color: var(--color-success);">0</div>
            </div>
            <div class="stat-card" style="border-left: 4px solid var(--color-warning);">
                <div class="label">Dry-Run Simulations</div>
                <div class="value" id="stat-dryrun" style="color: var(--color-warning);">0</div>
            </div>
            <div class="stat-card" style="border-left: 4px solid var(--color-danger);">
                <div class="label">Rejected Intents</div>
                <div class="value" id="stat-rejected" style="color: var(--color-danger);">0</div>
            </div>
            <div class="stat-card">
                <div class="label">Avg Intent Confidence</div>
                <div class="value" id="stat-confidence">0.0%</div>
            </div>
        </div>

        <div class="content-grid">
            <div class="card">
                <h2>Real-Time Trace History</h2>
                <div class="traces-list" id="traces-container">
                    <div class="no-data">Listening for subvocal intents...</div>
                </div>
            </div>

            <div style="display: flex; flex-direction: column; gap: 2rem;">
                <div class="card">
                    <h2>Confidence Distribution</h2>
                    <div class="chart-container">
                        <canvas id="confidence-chart"></canvas>
                    </div>
                </div>

                <div class="card">
                    <h2>Recent Latency Trend</h2>
                    <div class="chart-container">
                        <canvas id="latency-chart"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        let confidenceChart = null;
        let latencyChart = null;

        async function fetchTelemetry() {
            try {
                // Fetch stats
                const statsRes = await fetch('/api/stats');
                const stats = await statsRes.json();
                
                document.getElementById('stat-total').innerText = stats.total;
                document.getElementById('stat-executed').innerText = stats.success;
                document.getElementById('stat-dryrun').innerText = stats.dry_run;
                document.getElementById('stat-rejected').innerText = stats.rejected;
                document.getElementById('stat-confidence').innerText = (stats.avg_confidence * 100).toFixed(1) + '%';

                // Fetch traces
                const tracesRes = await fetch('/api/traces');
                const traces = await tracesRes.json();

                renderTraces(traces);
                renderCharts(traces);

            } catch (err) {
                console.error("Failed to load telemetry:", err);
            }
        }

        function renderTraces(traces) {
            const container = document.getElementById('traces-container');
            if (traces.length === 0) {
                container.innerHTML = '<div class="no-data">Listening for subvocal intents...</div>';
                return;
            }

            container.innerHTML = '';
            traces.forEach(trace => {
                const item = document.createElement('div');
                item.className = 'trace-item';
                
                const timeString = new Date(trace.timestamp * 1000).toLocaleTimeString();
                const tokenString = trace.tokens.map(t => t.text).join(' → ');
                const appName = trace.context?.active_application || 'Unknown System';
                const resolvedText = trace.intent?.resolved_text || 'None';
                const confidence = trace.intent?.confidence ? (trace.intent.confidence * 100).toFixed(0) + '%' : 'N/A';

                item.innerHTML = `
                    <div class="trace-header">
                        <span class="trace-app">${appName}</span>
                        <span class="trace-time">${timeString}</span>
                    </div>
                    <div class="trace-body">
                        <span class="token-box">${tokenString || 'None'}</span>
                        <span class="intent-text">${resolvedText}</span>
                    </div>
                    <div class="trace-details">
                        <span>Confidence: <strong>${confidence}</strong></span>
                        <span class="trace-pill status-${trace.status.toLowerCase()}">${trace.status}</span>
                    </div>
                `;
                container.appendChild(item);
            });
        }

        function renderCharts(traces) {
            const last10 = [...traces].reverse().slice(-10);
            const labels = last10.map((_, i) => `Run ${i+1}`);
            const confidences = last10.map(t => t.intent?.confidence || 0);
            
            // Random simulated latencies for metrics visualization
            const latencies = last10.map(t => t.status === "ERROR" ? 0 : 25 + Math.random() * 10);

            if (!confidenceChart) {
                const ctx1 = document.getElementById('confidence-chart').getContext('2d');
                confidenceChart = new Chart(ctx1, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Confidence Score',
                            data: confidences,
                            backgroundColor: 'rgba(16, 185, 129, 0.4)',
                            borderColor: '#10b981',
                            borderWidth: 1.5,
                            borderRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.05)' } },
                            x: { grid: { display: false } }
                        }
                    }
                });
            } else {
                confidenceChart.data.labels = labels;
                confidenceChart.data.datasets[0].data = confidences;
                confidenceChart.update();
            }

            if (!latencyChart) {
                const ctx2 = document.getElementById('latency-chart').getContext('2d');
                latencyChart = new Chart(ctx2, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Pipeline Latency (ms)',
                            data: latencies,
                            borderColor: '#3b82f6',
                            borderWidth: 2,
                            tension: 0.3,
                            fill: true,
                            backgroundColor: 'rgba(59, 130, 246, 0.1)'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { grid: { color: 'rgba(255,255,255,0.05)' } },
                            x: { grid: { display: false } }
                        }
                    }
                });
            } else {
                latencyChart.data.labels = labels;
                latencyChart.data.datasets[0].data = latencies;
                latencyChart.update();
            }
        }

        // Poll every 1.5s
        fetchTelemetry();
        setInterval(fetchTelemetry, 1500);
    </script>
</body>
</html>
"""


class ObservabilityDashboardHandler(BaseHTTPRequestHandler):
    """Observability Dashboard API and HTML endpoints."""

    def log_message(self, format, *args):
        # Override to suppress standard terminal request logs
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
            
        elif self.path == "/api/traces":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            traces = self._get_traces()
            self.wfile.write(json.dumps(traces).encode("utf-8"))
            
        elif self.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            stats = self._get_stats()
            self.wfile.write(json.dumps(stats).encode("utf-8"))
            
        else:
            self.send_response(404)
            self.end_headers()

    def _get_traces(self) -> list:
        from emg_core import config
        trace_path = os.path.join(config.DATA_DIR, "pipeline_traces.jsonl")
        if not os.path.exists(trace_path):
            return []
        
        traces = []
        with open(trace_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        traces.append(json.loads(line))
                    except Exception:
                        pass
        return list(reversed(traces))[-100:]

    def _get_stats(self) -> dict:
        traces = self._get_traces()
        total = len(traces)
        if total == 0:
            return {
                "total": 0,
                "success": 0,
                "rejected": 0,
                "dry_run": 0,
                "error": 0,
                "avg_confidence": 0.0
            }
        
        success = sum(1 for t in traces if t.get("status") == "SUCCESS")
        rejected = sum(1 for t in traces if t.get("status") == "REJECTED_UNAUTHORIZED")
        dry_run = sum(1 for t in traces if t.get("status") == "DRY_RUN")
        error = sum(1 for t in traces if t.get("status") == "ERROR")
        
        confidences = []
        for t in traces:
            intent = t.get("intent")
            if intent:
                confidences.append(float(intent.get("confidence", 0.0)))
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total": total,
            "success": success,
            "rejected": rejected,
            "dry_run": dry_run,
            "error": error,
            "avg_confidence": avg_conf
        }


def start_dashboard(port: int = 8000):
    server = HTTPServer(("localhost", port), ObservabilityDashboardHandler)
    print(f"Observability Dashboard listening on http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Subvocal Telemetry Observability Dashboard.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run HTTP server on")
    args = parser.parse_args()
    start_dashboard(args.port)
