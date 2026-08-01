import React from "react";
import ReactDOM from "react-dom/client";
import RecordingOverlay from "./RecordingOverlay";
import SelectionTranslationOverlay from "./SelectionTranslationOverlay";
import "@/i18n";

const isSelectionTranslationOverlay =
  new URLSearchParams(window.location.search).get("mode") ===
  "selection-translation";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {isSelectionTranslationOverlay ? (
      <SelectionTranslationOverlay />
    ) : (
      <RecordingOverlay />
    )}
  </React.StrictMode>,
);
