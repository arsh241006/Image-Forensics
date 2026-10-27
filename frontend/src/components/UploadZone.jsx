import { useRef, useState } from "react";

const MAX_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/png"];

export default function UploadZone({ onFile }) {
  const inputRef = useRef(null);

  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  function acceptFile(file) {
    if (!file) return;

    if (!ALLOWED_TYPES.includes(file.type)) {
      setError("Please choose a JPG or PNG image.");
      return;
    }

    if (file.size > MAX_BYTES) {
      setError("The image must be 10 MB or smaller.");
      return;
    }

    setError("");
    onFile(file);
  }

  function openFilePicker() {
    inputRef.current?.click();
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFilePicker();
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files[0]);
  }

  return (
    <div>
      <div
        className={`dropzone ${dragging ? "dragging" : ""}`}
        role="button"
        tabIndex={0}
        aria-label="Choose or drop an image to analyze"
        onClick={openFilePicker}
        onKeyDown={handleKeyDown}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <p className="dropzone-title">
          Drop an image here, or click to choose
        </p>

        <p className="dropzone-hint">
          JPG or PNG, up to 10 MB. JPEG is recommended.
        </p>

        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          hidden
          onChange={(event) => {
            acceptFile(event.target.files[0]);
            event.target.value = "";
          }}
        />
      </div>

      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}