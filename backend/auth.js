/**
 * auth.js — JWT middleware and secret
 * Exported so server.js stays clean and auth logic is reusable.
 */

const jwt = require('jsonwebtoken');

// In production: set JWT_SECRET as an EC2 environment variable.
// The fallback string is for local demo only — never use it in production.
const SECRET = process.env.JWT_SECRET || 'nsy107-secret-key';

const USERS = {
  admin: "123456",
};

function login(req,res){
  const{username, password} =req.body || {};
  if(!username || !password){
    return res.status(400).json({
      message: "Username and password are required",
    });
  }

  if (USERS[username] !==password){
    return res.status(401).json({
      message: "Invalid username or password",
    });
  }
  const token = jwt.sign(
    {
      username,
      role: "admin",
    },
    SECRET,
    {
      expiresIn: "1h",
    }
  );
  
  return res.json({
    message: "Login successful",
    token,
  });
}


function verifyToken(req, res, next) {
  const header = req.headers['authorization'] || '';

  if (!header.startsWith("Bearer ")) {
    return res.status(401).json({
      message: "Missing or invalid Authorization header",
    });
  }

  const token = header.slice(7);
  
  try {
    req.user = jwt.verify(token, SECRET);
    next();
  } catch {
    return res.status(403).json({
      message: "Invalid or expired token",
    });
  }
}

module.exports = { login, verifyToken };
