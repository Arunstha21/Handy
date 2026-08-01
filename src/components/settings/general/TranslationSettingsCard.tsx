import React from "react";
import { useTranslation } from "react-i18next";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { ShortcutInput } from "../ShortcutInput";
import { TranslateToEnglish } from "../TranslateToEnglish";
import { useModelStore } from "../../../stores/modelStore";

/**
 * Keeps the controls for the dedicated translation action together. Translation
 * can use any downloaded ASR model in the balanced profile, so it should not
 * disappear just because that model has no built-in speech-translation mode.
 */
export const TranslationSettingsCard: React.FC = () => {
  const { t } = useTranslation();
  const { currentModel, models } = useModelStore();
  const currentModelInfo = models.find((model) => model.id === currentModel);

  if (!currentModel || !currentModelInfo) {
    return null;
  }

  const supportsDirectTranslation = currentModelInfo.supports_translation;
  const translationTargetLanguages =
    currentModelInfo.engine_type === "TranscribeCpp"
      ? ["en"]
      : currentModelInfo.engine_type === "Canary"
        ? currentModelInfo.supported_languages
        : [];

  return (
    <SettingsGroup title={t("settings.translation.title")}>
      <TranslateToEnglish
        descriptionMode="tooltip"
        grouped={true}
        supportedLanguages={currentModelInfo.supported_languages}
        translationTargetLanguages={translationTargetLanguages}
        supportsDirectTranslation={supportsDirectTranslation}
      />
      <ShortcutInput
        shortcutId="transcribe_with_translation"
        descriptionMode="tooltip"
        grouped={true}
      />
    </SettingsGroup>
  );
};
