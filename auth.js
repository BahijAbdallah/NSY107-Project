const jwt = require("jsonwebtoken");

const JWT_SECRET = "nsy107-secret-key"; // demo

function login(req, res) {
  const { username, password } = req.body;

  if (username !== "admin" || password !== "123456") {
    return res.status(401).json({
      message: "Invalid username or password",
    });
  }

  const token = jwt.sign(
    {
      username,
      role: "admin",
    },
    JWT_SECRET,
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
  const authHeader = req.headers.authorization;

  if (!authHeader) {
    return res.status(401).json({
      message: "Missing Authorization header",
    });
  }

  const token = authHeader.split(" ")[1];

  if (!token) {
    return res.status(401).json({
      message: "Missing token",
    });
  }

  jwt.verify(token, JWT_SECRET, (err, decoded) => {
    if (err) {
      return res.status(403).json({
        message: "Invalid or expired token",
      });
    }

    req.user = decoded;
    next();
  });
}

module.exports = {
  login,
  verifyToken,
};