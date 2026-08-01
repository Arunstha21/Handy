import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSettings } from "../../hooks/useSettings";
import { SettingContainer } from "../ui/SettingContainer";
import { Textarea } from "../ui/Textarea";

const MAX_CONTEXT_CHARS = 2_000;

interface TranscriptionContextProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/**
 * Provides stable domain vocabulary to Whisper-family decoders without adding
 * another model or sending the context to a remote service.
 */
export const TranscriptionContext: React.FC<TranscriptionContextProps> = ({
  descriptionMode = "tooltip",
  grouped = false,
}) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const savedContext = getSetting("transcription_context") || "";
  const [draftContext, setDraftContext] = useState(savedContext);

  useEffect(() => {
    setDraftContext(savedContext);
  }, [savedContext]);

  const saveContext = () => {
    const normalized = draftContext.trim();
    if (normalized !== savedContext) {
      void updateSetting("transcription_context", normalized);
    }
  };

  return (
    <SettingContainer
      title={t("settings.advanced.transcriptionContext.title")}
      description={t("settings.advanced.transcriptionContext.description")}
      descriptionMode={descriptionMode}
      grouped={grouped}
      layout="stacked"
    >
      <Textarea
        value={draftContext}
        onChange={(event) => setDraftContext(event.target.value)}
        onBlur={saveContext}
        onKeyDown={(event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            event.preventDefault();
            saveContext();
          }
        }}
        maxLength={MAX_CONTEXT_CHARS}
        rows={3}
        placeholder={t("settings.advanced.transcriptionContext.placeholder")}
        disabled={isUpdating("transcription_context")}
        aria-label={t("settings.advanced.transcriptionContext.title")}
      />
      <div className="mt-1 text-right text-xs text-mid-gray">
        {draftContext.length}/{MAX_CONTEXT_CHARS}
      </div>
    </SettingContainer>
  );
};
