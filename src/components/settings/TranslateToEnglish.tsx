import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { Dropdown } from "../ui/Dropdown";
import { SettingContainer } from "../ui/SettingContainer";
import { useSettings } from "../../hooks/useSettings";
import {
  getLanguageLabel,
  MODEL_CAPABILITY_LANGUAGES,
  supportsLanguageCode,
} from "../../lib/constants/languages";

interface TranslateToEnglishProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
  supportedLanguages?: string[];
}

export const TranslateToEnglish: React.FC<TranslateToEnglishProps> = React.memo(
  ({
    descriptionMode = "tooltip",
    grouped = false,
    supportedLanguages = [],
  }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const translationEnabled = getSetting("translation_enabled") || false;
    const targetLanguage = getSetting("translation_target_language") || "en";
    const targetOptions = MODEL_CAPABILITY_LANGUAGES.filter(
      (language) =>
        supportedLanguages.length === 0 ||
        supportsLanguageCode(supportedLanguages, language.value),
    ).map((language) => ({
      value: language.value,
      label: getLanguageLabel(language.value) || language.label,
    }));

    return (
      <>
        <ToggleSwitch
          checked={translationEnabled}
          onChange={(enabled) => updateSetting("translation_enabled", enabled)}
          isUpdating={isUpdating("translation_enabled")}
          label={t("settings.advanced.translation.label")}
          description={t("settings.advanced.translation.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
        />
        {translationEnabled && (
          <SettingContainer
            title={t("settings.advanced.translation.targetLabel")}
            description={t("settings.advanced.translation.targetDescription")}
            descriptionMode={descriptionMode}
            grouped={grouped}
          >
            <Dropdown
              options={targetOptions}
              selectedValue={targetLanguage}
              onSelect={(value) =>
                updateSetting("translation_target_language", value)
              }
              disabled={
                targetOptions.length === 0 ||
                isUpdating("translation_target_language")
              }
            />
          </SettingContainer>
        )}
      </>
    );
  },
);
