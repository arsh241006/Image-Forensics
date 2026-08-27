import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to analyze image.");
      }

      const data = await response.json();

      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#f5f7fa",
        padding: "40px 20px",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <div
        style={{
          maxWidth: "900px",
          margin: "0 auto",
          backgroundColor: "#fff",
          padding: "30px",
          borderRadius: "12px",
          boxShadow: "0 4px 15px rgba(0,0,0,0.1)",
        }}
      >
        <h1 style={{ textAlign: "center", marginBottom: "10px" }}>
          Image Forensics
        </h1>

        <p style={{ textAlign: "center", color: "#666" }}>
          Upload an image to detect possible image tampering.
        </p>

        {/* Upload */}
        <div
          style={{
            border: "2px dashed #bbb",
            borderRadius: "10px",
            padding: "30px",
            textAlign: "center",
            marginTop: "30px",
          }}
        >
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />

          {preview && (
            <div style={{ marginTop: "25px" }}>
              <img
                src={preview}
                alt="Preview"
                style={{
                  maxWidth: "100%",
                  maxHeight: "400px",
                  borderRadius: "8px",
                  objectFit: "contain",
                }}
              />
            </div>
          )}
        </div>

        {/* Analyze Button */}
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          style={{
            display: "block",
            width: "100%",
            marginTop: "20px",
            padding: "14px",
            border: "none",
            borderRadius: "8px",
            backgroundColor: !file || loading ? "#aaa" : "#2563eb",
            color: "white",
            fontSize: "16px",
            fontWeight: "bold",
            cursor: !file || loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Analyzing..." : "Analyze Image"}
        </button>

        {/* Error */}
        {error && (
          <div
            style={{
              marginTop: "20px",
              padding: "15px",
              backgroundColor: "#fee2e2",
              color: "#b91c1c",
              borderRadius: "8px",
            }}
          >
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div
            style={{
              marginTop: "30px",
              padding: "20px",
              borderRadius: "10px",
              backgroundColor: "#f8fafc",
              border: "1px solid #ddd",
            }}
          >
            <h2>Analysis Result</h2>

            <div style={{ marginTop: "15px" }}>
              {Object.entries(result).map(([key, value]) => (
                <div
                  key={key}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "12px 0",
                    borderBottom: "1px solid #eee",
                  }}
                >
                  <strong>{key}</strong>
                  <span>
                    {typeof value === "object"
                      ? JSON.stringify(value)
                      : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;