import { useEffect, useState } from "react";

export default function AnalysisProgress({ previewUrl, onCancel }) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setSeconds((previous) => previous + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return (
    <section className="card progress">
      {previewUrl && (
        <img
          className="progress-thumb"
          src={previewUrl}
          alt="Image being analyzed"
        />
      )}

      <div>
        <p className="progress-title" role="status">
          <span className="spinner" aria-hidden="true" />
          Analyzing your image
        </p>

        <p className="muted">Elapsed time: {seconds} seconds</p>

        <button
          type="button"
          className="btn-secondary"
          onClick={onCancel}
        >
          Cancel
        </button>
      </div>
    </section>
  );
}