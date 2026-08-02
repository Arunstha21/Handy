import React, { useEffect, useState } from "react";
import { emit, listen } from "@tauri-apps/api/event";
import { useTranslation } from "react-i18next";
import "./SelectionTranslationOverlay.css";

type SelectionTranslationState = "loading" | "success" | "error";

type SelectionTranslationPresentation = {
  state: SelectionTranslationState;
  targetLanguage?: string;
  text?: string;
};

const SelectionTranslationOverlay: React.FC = () => {
  const { t } = useTranslation();
  const [isVisible, setIsVisible] = useState(false);
  const [presentation, setPresentation] =
    useState<SelectionTranslationPresentation>({ state: "loading" });

  useEffect(() => {
    let cancelled = false;
    let unlistenShow: (() => void) | undefined;
    let unlistenHide: (() => void) | undefined;

    const setupListeners = async () => {
      const stopShow = await listen<SelectionTranslationPresentation>(
        "show-selection-translation",
        (event) => {
          if (cancelled) return;
          setPresentation(event.payload);
          setIsVisible(true);
        },
      );
      if (cancelled) {
        stopShow();
        return;
      }
      unlistenShow = stopShow;

      const stopHide = await listen("hide-selection-translation", () => {
        if (!cancelled) setIsVisible(false);
      });
      if (cancelled) {
        stopHide();
        return;
      }
      unlistenHide = stopHide;

      if (!cancelled) {
        await emit("selection-translation-ready");
      }
    };

    void setupListeners();
    return () => {
      cancelled = true;
      unlistenShow?.();
      unlistenHide?.();
    };
  }, []);

  const targetLanguage =
    presentation.targetLanguage || t("selectionTranslation.targetLanguage");
  const heading =
    presentation.state === "loading"
      ? t("selectionTranslation.translating", { language: targetLanguage })
      : presentation.state === "success"
        ? t("selectionTranslation.translatedTo", { language: targetLanguage })
        : t("selectionTranslation.failed");

  return (
    <div className={`selection-translation-stage ${isVisible ? "show" : ""}`}>
      <section
        className={`selection-translation-card ${presentation.state}`}
        aria-live="polite"
        aria-atomic="true"
        role="status"
      >
        <div className="selection-translation-header">
          {presentation.state === "loading" ? (
            <span
              className="selection-translation-spinner"
              aria-hidden="true"
            />
          ) : (
            <span
              className="selection-translation-status-dot"
              aria-hidden="true"
            />
          )}
          <span className="selection-translation-heading">{heading}</span>
        </div>
        {presentation.state !== "loading" && presentation.text && (
          <p className="selection-translation-text">{presentation.text}</p>
        )}
      </section>
    </div>
  );
};

export default SelectionTranslationOverlay;
