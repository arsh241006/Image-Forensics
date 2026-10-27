export function mockResponse(filename) {
  return {
    filename: filename,
    label: "tampered",
    tampered_probability: 0.84,
    manipulation_type: "copy-move",
    manipulation_details: {
      consistent_matches: 12,
    },
    heatmap: null,
    frequency_view: null,
    warnings: [
      "This is a mock result used for frontend development.",
    ],
  };
}