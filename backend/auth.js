/**
 * auth.js — JWT middleware and secret
 * Exported so server.js stays clean and auth logic is reusable.
 */

const jwt = require('jsonwebtoken');

// In production: set JWT_SECRET as an EC2 environment variable.
// The fallback string is for local demo only — never use it in production.
const SECRET = process.env.JWT_SECRET || 'nsy107-secret-key';

function verifyToken(req, res, next) {
  const header = req.headers['authorization'] || '';
  const token  = header.startsWith('Bearer ') ? header.slice(7) : null;

  if (!token) {
    return res.status(401).json({ error: 'No token provided' });
  }
  try {
    req.user = jwt.verify(token, SECRET);
    next();
  } catch {
    return res.status(403).json({ error: 'Invalid or expired token' });
  }
}

module.exports = { verifyToken, SECRET };
