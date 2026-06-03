import { useState } from "react";
import { uploadLogCsv } from "../api";

function LogAnalysis() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();

    if (!file) {
      setError("Please choose a CSV file before starting analysis.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await uploadLogCsv(file);
      setResult(data);
      if (!data.success && data.error) {
        setError(data.error);
      }
    } catch (requestError) {
      const details = requestError.data?.error || requestError.data?.message || requestError.message;
      setError(details);
      setResult(requestError.data || null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="analysis-page">
      <div className="page-intro">
        <h2>AWS CloudWatch Log Analysis</h2>
        <p>
          Upload an exported CloudWatch CSV file to analyze real API Gateway logs using the
          trained ML anomaly detection model.
        </p>
      </div>

      <form className="analysis-form" onSubmit={handleSubmit}>
        <label>
          CloudWatch CSV file
          <input
            accept=".csv,text/csv"
            type="file"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
          />
        </label>
        <button type="submit" disabled={loading || !file}>
          {loading ? "Analyzing logs..." : "Start Analysis"}
        </button>
      </form>

      {error && (
        <div className="error-box">
          <strong>Analysis failed</strong>
          <p>{error}</p>
        </div>
      )}

      <section className="result-panel">
        <h2>Analysis Result</h2>
        {loading ? (
          <pre className="response">Analyzing logs...</pre>
        ) : result?.output ? (
          <pre className="response analysis-output">{result.output}</pre>
        ) : (
          <pre className="response muted">No analysis run yet.</pre>
        )}
      </section>
    </section>
  );
}

export default LogAnalysis;
