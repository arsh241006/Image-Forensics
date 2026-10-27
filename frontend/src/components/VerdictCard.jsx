export default function VerdictCard({ label, probability }) {
  const isTampered = String(label).toLowerCase() === "tampered";

  const hasScore =
    typeof probability === "number" &&
    Number.isFinite(probability) &&
    probability >= 0 &&
    probability <= 1;

  const percentage = hasScore ? Math.round(probability * 100) : null;

  return (
    <section className="verdict-card" aria-live="polite">
      <h2>Analysis result</h2>

      <p className="verdict-label">
        {isTampered
          ? "⚠️ Flagged as potentially tampered"
          : "✓ Looks authentic to the model"}
      </p>

      {hasScore ? (
        <p>
          Model score for tampering: <strong>{percentage}%</strong>
        </p>
      ) : (
        <p>Model score is unavailable.</p>
      )}

      <p className="verdict-note">
        This is a model prediction, not a certainty.
      </p>
    </section>
  );
}