export default function ImageCompare({ originalUrl, heatmapUrl }) {
  if (!originalUrl) return null;

  return (
    <section className="image-card">
      <h2>Uploaded image</h2>

      <img
        className="result-image"
        src={originalUrl}
        alt="The image uploaded for analysis"
      />

      {heatmapUrl ? (
        <>
          <h3>Approximate regions influencing the prediction</h3>
          <img
            className="result-image"
            src={heatmapUrl}
            alt="Heatmap showing approximate regions that influenced the prediction"
          />
          <p>
            The heatmap is approximate and does not mark exact edit boundaries.
          </p>
        </>
      ) : (
        <p className="image-note">Heatmap is not available yet.</p>
      )}
    </section>
  );
}