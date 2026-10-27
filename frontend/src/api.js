import { mockResponse } from "./mock/sampleResponse";


const API_URL =
  process.env.REACT_APP_API_URL || "http://localhost:8000";

const USE_MOCK = process.env.REACT_APP_USE_MOCK === "true";

export async function analyzeImage(file, { signal } = {}) {
  if (USE_MOCK) {
  await new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Request canceled", "AbortError"));
      return;
    }

    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", handleAbort);
      resolve();
    }, 1500);

    function handleAbort() {
      clearTimeout(timer);
      signal?.removeEventListener("abort", handleAbort);
      reject(new DOMException("Request canceled", "AbortError"));
    }

    signal?.addEventListener("abort", handleAbort, { once: true });
  });

  return mockResponse(file.name);
}

  const formData = new FormData();
  formData.append("file", file);

  let response;

 try {
  response = await fetch(`${API_URL}/predict`, {
  method: "POST",
  body: formData,
  signal,
});
  } catch (error) {
    if (error.name === "AbortError") {
      throw error;
    }

    throw new Error(
      "Could not reach the analysis server. Is the backend running?"
    );
  }

  if (!response.ok) {
    throw new Error(
      `The analysis failed (server returned ${response.status}).`
    );
  }

  return response.json();
}