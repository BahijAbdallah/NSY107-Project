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
const { login, verifyToken } = require("./auth");
const jwt = require('jsonwebtoken');

const app  = express();
const PORT = process.env.PORT || 3000;

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