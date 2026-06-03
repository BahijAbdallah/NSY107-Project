import { useMemo, useState } from "react";
import { API_BASE_URL, apiRequest } from "../api";

const initialResponses = {
  health: null,
  publicApi: null,
  login: null,
  secure: null,
  orders: null,
};

function formatError(error) {
  return {
    ok: false,
    status: error.status || "Network error",
    message: error.message,
    details: error.data || null,
  };
}

function ResultBlock({ result }) {
  if (!result) {
    return <pre className="response muted">No request sent yet.</pre>;
  }

  return <pre className="response">{JSON.stringify(result, null, 2)}</pre>;
}

function Card({ title, description, children, result }) {
  return (
    <section className="card">
      <div className="card-header">
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
      </div>
      <div className="card-body">{children}</div>
      <ResultBlock result={result} />
    </section>
  );
}

function Home() {
  const [token, setToken] = useState(() => localStorage.getItem("jwtToken") || "");
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("123456");
  const [itemName, setItemName] = useState("Laptop");
  const [quantity, setQuantity] = useState(1);
  const [responses, setResponses] = useState(initialResponses);
  const [loading, setLoading] = useState(null);

  const tokenPreview = useMemo(() => {
    if (!token) return "No token saved";
    return `${token.slice(0, 24)}...${token.slice(-12)}`;
  }, [token]);

  async function runRequest(key, request) {
    setLoading(key);
    setResponses((current) => ({ ...current, [key]: null }));

    try {
      const data = await request();
      setResponses((current) => ({ ...current, [key]: { ok: true, data } }));
      return data;
    } catch (error) {
      setResponses((current) => ({ ...current, [key]: formatError(error) }));
      return null;
    } finally {
      setLoading(null);
    }
  }

  function authHeaders() {
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  async function handleLogin(event) {
    event.preventDefault();

    const data = await runRequest("login", () =>
      apiRequest("/login", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      })
    );

    if (data?.token) {
      setToken(data.token);
      localStorage.setItem("jwtToken", data.token);
    }
  }

  function clearToken() {
    setToken("");
    localStorage.removeItem("jwtToken");
  }

  function handleCreateOrder(event) {
    event.preventDefault();

    return runRequest("orders", () =>
      apiRequest("/orders", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          itemName,
          quantity: Number(quantity),
        }),
      })
    );
  }

  return (
    <>
      <header className="topbar">
        <div className="api-base">
          API base URL: <code>{API_BASE_URL}</code>
        </div>
        <div className="status-panel">
          <span className={token ? "status online" : "status offline"}>
            {token ? "Logged in" : "Not logged in"}
          </span>
          <code>{tokenPreview}</code>
          <button className="secondary" type="button" onClick={clearToken} disabled={!token}>
            Clear Token
          </button>
        </div>
      </header>

      <div className="grid">
        <Card
          title="Health Check"
          description="GET / confirms the backend server is running."
          result={responses.health}
        >
          <button
            type="button"
            onClick={() => runRequest("health", () => apiRequest("/"))}
            disabled={loading === "health"}
          >
            {loading === "health" ? "Testing..." : "Test Backend"}
          </button>
        </Card>

        <Card
          title="Public Endpoint"
          description="GET /public does not require authentication."
          result={responses.publicApi}
        >
          <button
            type="button"
            onClick={() => runRequest("publicApi", () => apiRequest("/public"))}
            disabled={loading === "publicApi"}
          >
            {loading === "publicApi" ? "Calling..." : "Call Public API"}
          </button>
        </Card>

        <Card
          title="Login"
          description="POST /login returns a JWT for valid credentials."
          result={responses.login}
        >
          <form className="form" onSubmit={handleLogin}>
            <label>
              Username
              <input value={username} onChange={(event) => setUsername(event.target.value)} />
            </label>
            <label>
              Password
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
            </label>
            <button type="submit" disabled={loading === "login"}>
              {loading === "login" ? "Logging in..." : "Login"}
            </button>
          </form>
        </Card>

        <Card
          title="Secure Endpoint"
          description="GET /secure requires Authorization: Bearer <token>."
          result={responses.secure}
        >
          <button
            type="button"
            onClick={() =>
              runRequest("secure", () =>
                apiRequest("/secure", {
                  headers: authHeaders(),
                })
              )
            }
            disabled={loading === "secure"}
          >
            {loading === "secure" ? "Calling..." : "Call Secure API"}
          </button>
        </Card>

        <Card
          title="Orders"
          description="POST /orders requires a JWT and validates the order body."
          result={responses.orders}
        >
          <form className="form" onSubmit={handleCreateOrder}>
            <label>
              Item name
              <input value={itemName} onChange={(event) => setItemName(event.target.value)} />
            </label>
            <label>
              Quantity
              <input
                type="number"
                min="1"
                step="1"
                value={quantity}
                onChange={(event) => setQuantity(event.target.value)}
              />
            </label>
            <button type="submit" disabled={loading === "orders"}>
              {loading === "orders" ? "Creating..." : "Create Order"}
            </button>
          </form>
        </Card>
      </div>
    </>
  );
}

export default Home;
