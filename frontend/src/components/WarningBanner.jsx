export default function WarningBanner({ warnings }) {
  if (!Array.isArray(warnings) || warnings.length === 0) {
    return null;
  }

  return (
    <section className="warning-banner" role="status">
      <strong>Things to keep in mind</strong>
      <ul>
        {warnings.map((warning, index) => (
          <li key={index}>{String(warning)}</li>
        ))}
      </ul>
    </section>
  );
}