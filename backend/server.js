/**
 * server.js — NSY107 Project 6 Backend
 * Express API with JWT auth, request validation, CORS, and JSON request logging.
 *
 * Routes:
 *   GET  /         — health check
 *   GET  /public   — public endpoint, no auth
 *   POST /login    — returns JWT token
 *   GET  /secure   — requires valid JWT
 *   POST /orders   — requires JWT + validates itemName and quantity
 */

const express = require("express");
const cors    = require("cors");
const fs = require("fs");
const path = require("path");
const multer = require("multer");
const { spawn } = require("child_process");
const { login, verifyToken } = require("./auth");
const jwt = require('jsonwebtoken');

const app  = express();
const PORT = process.env.PORT || 3000;
const PROJECT_ROOT = path.join(__dirname, "..");
const UPLOAD_DIR = path.join(__dirname, "uploads");
const pythonCmd = process.platform === "win32" ? "python" : "python3";
const PYTHON_COMMAND = process.env.PYTHON || pythonCmd;

fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const upload = multer({
  dest: UPLOAD_DIR,
  limits: {
    fileSize: 10 * 1024 * 1024,
  },
  fileFilter: (req, file, cb) => {
    const isCsv =
      path.extname(file.originalname).toLowerCase() === ".csv" ||
      file.mimetype === "text/csv" ||
      file.mimetype === "application/vnd.ms-excel";

    if (!isCsv) {
      return cb(new Error("Only CSV files are allowed"));
    }

    return cb(null, true);
  },
});

app.use(cors());
app.use(express.json());

// ── Request logger ────────────────────────────────────────────────────────────
// Writes one JSON line per request to stdout.
// On EC2 this is captured by CloudWatch Logs Agent / the ML anomaly pipeline.
app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    console.log(JSON.stringify({
      timestamp:      new Date().toISOString(),
      ip:             req.ip || req.socket.remoteAddress,
      method:         req.method,
      route:          req.path,
      status:         res.statusCode,
      latency_ms:     Date.now() - start,
      content_length: res.getHeader('content-length') || 0
    }));
  });
  next();
});

// ── Routes ────────────────────────────────────────────────────────────────────

// Health check — required by API Gateway and load balancers
app.get("/", (req, res) => {
  res.json({ status: "ok", message: "NSY107 Project 6 API is running" });
});

// Public — no authentication required
app.get('/public', (req, res) => {
  res.json({ message: "Public endpoint — no authentication required" });
});

// Login — validates credentials and returns a signed JWT
app.post("/login", login)

// Secure — requires a valid JWT in Authorization: Bearer <token>
app.get("/secure", verifyToken, (req, res) => {
  res.json({
    message:   "Access granted to secure resource",
    user:      req.user.username,
    role:      req.user.role,
    timestamp: new Date().toISOString()
  });
});

// Orders — requires JWT + validates body fields
app.post("/orders", verifyToken, (req, res) => {
  const { itemName, quantity } = req.body || {};

  if (!itemName || typeof itemName !== "string" || itemName.trim() === "") {
    return res.status(400).json({
      message: "Validation failed: itemName must be a non-empty string",
    });
  }
    if (
    quantity === undefined ||
    quantity === null ||
    typeof quantity !== "number" ||
    !Number.isInteger(quantity) ||
    quantity <= 0
  ) {
    return res.status(400).json({
      message: "Validation failed: quantity must be a positive integer",
    });
  }

  res.status(201).json({
    message: "Order created successfully",
    order: {
      id: Date.now(),
      itemName: itemName.trim(),
      quantity,
      createdBy: req.user.username,
      timestamp: new Date().toISOString(),
    },
  });
});

function runLogAnalysis(req, res) {
  if (!req.file) {
    return res.status(400).json({
      success: false,
      output: "",
      error: "CSV file is required",
    });
  }

  const scriptPath = path.join(PROJECT_ROOT, "src", "aws_logs_to_predictions.py");
  const csvPath = req.file.path;
  let stdout = "";
  let stderr = "";
  let responseSent = false;

  const child = spawn(PYTHON_COMMAND, [scriptPath, "--csv", csvPath], {
    cwd: PROJECT_ROOT,
    windowsHide: true,
  });

  child.stdout.on("data", (chunk) => {
    stdout += chunk.toString();
  });

  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  child.on("error", (error) => {
    responseSent = true;
    fs.unlink(csvPath, () => {});
    return res.status(500).json({
      success: false,
      output: stdout,
      error: `Failed to start Python analysis: ${error.message}`,
    });
  });

  child.on("close", (code) => {
    if (responseSent) {
      return;
    }

    fs.unlink(csvPath, () => {});

    if (code !== 0) {
      return res.status(500).json({
        success: false,
        output: stdout,
        error: stderr || `Python analysis exited with code ${code}`,
      });
    }

    return res.json({
      success: true,
      output: stdout,
      error: stderr,
    });
  });
}

app.post("/analyze-logs", (req, res) => {
  upload.single("csvFile")(req, res, (error) => {
    if (error) {
      return res.status(400).json({
        success: false,
        output: "",
        error: error.message,
      });
    }

    return runLogAnalysis(req, res);
  });
});

// 404 fallback
app.use((req, res) => {
  res.status(404).json({
    message: `Route ${req.method} ${req.path} not found`,
  });
});

// Start server
// 0.0.0.0 is important on EC2 so API Gateway can reach the backend.
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Backend running on port ${PORT}`);
  console.log(
    "Routes: GET / | GET /public | POST /login | GET /secure | POST /orders"
  );
});
