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
  translationTargetLanguages?: string[];
  supportsDirectTranslation?: boolean;
}

export const TranslateToEnglish: React.FC<TranslateToEnglishProps> = React.memo(
  ({
    descriptionMode = "tooltip",
    grouped = false,
    supportedLanguages = [],
    translationTargetLanguages = [],
    supportsDirectTranslation = false,
  }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const translationEnabled = getSetting("translation_enabled") || false;
    const translationMode = getSetting("translation_mode") || "balanced";
    const effectiveTranslationMode =
      supportsDirectTranslation || translationMode === "balanced"
        ? translationMode
        : "balanced";
    const targetLanguage = getSetting("translation_target_language") || "en";
    const targetOptions = MODEL_CAPABILITY_LANGUAGES.filter((language) =>
      effectiveTranslationMode === "balanced"
        ? true
        : translationTargetLanguages.length === 0
          ? supportedLanguages.length === 0 ||
            supportsLanguageCode(supportedLanguages, language.value)
          : translationTargetLanguages.some(
              (target) =>
                target === language.value ||
                target.split("-")[0] === language.value.split("-")[0],
            ),
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
          <>
            <SettingContainer
              title={t("settings.advanced.translation.modeLabel")}
              description={t("settings.advanced.translation.modeDescription")}
              descriptionMode={descriptionMode}
              grouped={grouped}
            >
              <Dropdown
                options={[
                  {
                    value: "balanced",
                    label: t("settings.advanced.translation.modes.balanced"),
                  },
                  ...(supportsDirectTranslation
                    ? [
                        {
                          value: "direct",
                          label: t(
                            "settings.advanced.translation.modes.direct",
                          ),
                        },
                      ]
                    : []),
                ]}
                selectedValue={effectiveTranslationMode}
                onSelect={(value) =>
                  updateSetting(
                    "translation_mode",
                    value as "direct" | "balanced",
                  )
                }
                disabled={isUpdating("translation_mode")}
              />
            </SettingContainer>
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
          </>
        )}
      </>
    );
  },
);
