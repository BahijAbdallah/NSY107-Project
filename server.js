const express = require("express");
const cors = require("cors");
const { login, verifyToken } = require("./auth");

const app = express();

app.use(cors());
app.use(express.json());

const PORT = 3000;

app.get("/", (req, res) => {
  res.json({
    message: "NSY107 Project Backend is running",
  });
});

app.get("/public", (req, res) => {
  res.json({
    message: "Public API endpoint is working",
  });
});

app.post("/login", login);

app.get("/secure", verifyToken, (req, res) => {
  res.json({
    message: "Secure API endpoint accessed successfully",
    user: req.user,
  });
});

app.post("/orders", verifyToken, (req, res) => {
  const { itemName, quantity } = req.body;

  if (!itemName || !quantity) {
    return res.status(400).json({
      message: "itemName and quantity are required",
    });
  }

  res.status(201).json({
    message: "Order created successfully",
    order: {
      itemName,
      quantity,
    },
  });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on port ${PORT}`);
});