import React from "react";
import { useTranslation } from "react-i18next";
import { useSettings } from "../../hooks/useSettings";
import {
  getLanguageLabel,
  MODEL_CAPABILITY_LANGUAGES,
} from "../../lib/constants/languages";
import { Dropdown } from "../ui/Dropdown";
import { SettingContainer } from "../ui/SettingContainer";

interface SelectedTextTranslationTargetProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/** Target language used only by the selected-text translation shortcut. */
export const SelectedTextTranslationTarget: React.FC<SelectedTextTranslationTargetProps> =
  React.memo(({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();
    const selectedValue =
      getSetting("selected_text_translation_target_language") || "en";
    const options = MODEL_CAPABILITY_LANGUAGES.map((language) => ({
      value: language.value,
      label: getLanguageLabel(language.value) || language.label,
    }));

    return (
      <SettingContainer
        title={`${t("settings.general.shortcut.bindings.translate_selected_text.name")} · ${t("settings.advanced.translation.targetLabel")}`}
        description={t("settings.advanced.translation.targetDescription")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      >
        <Dropdown
          options={options}
          selectedValue={selectedValue}
          onSelect={(value) =>
            updateSetting("selected_text_translation_target_language", value)
          }
          disabled={isUpdating("selected_text_translation_target_language")}
        />
      </SettingContainer>
    );
  });

SelectedTextTranslationTarget.displayName = "SelectedTextTranslationTarget";
