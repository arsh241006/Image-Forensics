import { useEffect, useRef, useState } from "react";
import { analyzeImage } from "./api";
import UploadZone from "./components/UploadZone";
import AnalysisProgress from "./components/AnalysisProgress";

function App() {
  const [phase, setPhase] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");

  const abortRef = useRef(null);

  // Release the preview URL when it changes or the component unmounts.
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // Cancel an active request if the component unmounts.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  async function handleFile(file) {
    setError("");
    setResult(null);
    setPreviewUrl(URL.createObjectURL(file));
    setPhase("analyzing");

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const data = await analyzeImage(file, {
        signal: controller.signal,
      });

      if (controller.signal.aborted) return;

      setResult(data);
      setPhase("done");
    } catch (err) {
      if (controller.signal.aborted || err.name === "AbortError") {
        return;
      }

      setError(err.message || "Something went wrong.");
      setPhase("error");
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
    }
  }

  function cancel() {
    abortRef.current?.abort();
    abortRef.current = null;
    setPreviewUrl("");
    setPhase("idle");
  }

  function reset() {
    setResult(null);
    setError("");
    setPreviewUrl("");
    setPhase("idle");
  }

  return (
    <main className="page">
      <h1>Image Forensics</h1>

      <p className="muted">
        Upload a photo to check for possible signs of tampering.
      </p>

      {phase === "idle" && <UploadZone onFile={handleFile} />}

      {phase === "analyzing" && (
        <AnalysisProgress
          previewUrl={previewUrl}
          onCancel={cancel}
        />
      )}

      {phase === "error" && (
        <div className="banner banner-error" role="alert">
          <strong>Something went wrong.</strong> {error}

          <div className="banner-action">
            <button
              type="button"
              className="btn-secondary"
              onClick={reset}
            >
              Try another image
            </button>
          </div>
        </div>
      )}

      {phase === "done" && (
        <section className="card" aria-live="polite">
          <h2>Analysis result</h2>

          <p className="muted">
            This is a model output, not a certain determination.
          </p>

          <pre className="result-preview">
            {JSON.stringify(result, null, 2)}
          </pre>

          <button
            type="button"
            className="btn-secondary"
            onClick={reset}
          >
            Analyze another image
          </button>
        </section>
      )}
    </main>
  );
}

export default App;