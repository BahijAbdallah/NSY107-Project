export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json")
    ? response.json()
    : response.text().then((message) => ({ message }));
}

function createApiError(response, data) {
  const error = new Error(data.message || data.error || "Request failed");
  error.status = response.status;
  error.data = data;
  return error;
}

export async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  const data = await parseResponse(response);

  if (!response.ok) {
    throw createApiError(response, data);
  }

  return data;
}

export async function uploadLogCsv(file) {
  const formData = new FormData();
  formData.append("csvFile", file);

  const response = await fetch(`${API_BASE_URL}/analyze-logs`, {
    method: "POST",
    body: formData,
  });

  const data = await parseResponse(response);

  if (!response.ok) {
    throw createApiError(response, data);
  }

  return data;
}
