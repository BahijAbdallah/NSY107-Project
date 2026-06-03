import { NavLink, Route, Routes } from "react-router-dom";
import Home from "./pages/Home.jsx";
import LogAnalysis from "./pages/LogAnalysis.jsx";

function App() {
  return (
    <main className="app">
      <nav className="navbar">
        <div>
          <p className="eyebrow">NSY107 Project</p>
          <h1>API Route Dashboard</h1>
        </div>
        <div className="nav-links">
          <NavLink to="/" end>
            Home
          </NavLink>
          <NavLink to="/log-analysis">Log Analysis</NavLink>
        </div>
      </nav>

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/log-analysis" element={<LogAnalysis />} />
      </Routes>
    </main>
  );
}

export default App;
