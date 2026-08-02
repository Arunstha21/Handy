import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { ModelInfo } from "@/bindings";
import { useSettings } from "@/hooks/useSettings";
import { Dropdown } from "@/components/ui/Dropdown";
import { SettingContainer } from "@/components/ui/SettingContainer";
import { ToggleSwitch } from "@/components/ui/ToggleSwitch";

interface DualModelSettingsProps {
  currentModel: string;
  models: ModelInfo[];
}

export const DualModelSettings: React.FC<DualModelSettingsProps> = ({
  currentModel,
  models,
}) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const enabled = getSetting("dual_model_enabled") || false;
  const secondaryModelId = getSetting("secondary_model_id") || "";
  const candidates = useMemo(
    () =>
      models
        .filter(
          (model) =>
            model.is_downloaded &&
            model.id !== currentModel &&
            !model.is_custom,
        )
        .map((model) => ({ value: model.id, label: model.name })),
    [currentModel, models],
  );

  if (candidates.length === 0 && !enabled) return null;

  return (
    <>
      <ToggleSwitch
        checked={enabled}
        onChange={(value) => updateSetting("dual_model_enabled", value)}
        isUpdating={isUpdating("dual_model_enabled")}
        disabled={candidates.length === 0}
        label={t("settings.advanced.dualModel.label")}
        description={t("settings.advanced.dualModel.description")}
        descriptionMode="tooltip"
        grouped
      />
      {enabled && (
        <SettingContainer
          title={t("settings.advanced.dualModel.secondaryLabel")}
          description={t("settings.advanced.dualModel.secondaryDescription")}
          descriptionMode="tooltip"
          grouped
        >
          <Dropdown
            options={candidates}
            selectedValue={secondaryModelId}
            onSelect={(value) => updateSetting("secondary_model_id", value)}
            disabled={
              candidates.length === 0 || isUpdating("secondary_model_id")
            }
            placeholder={t("settings.advanced.dualModel.selectPlaceholder")}
          />
        </SettingContainer>
      )}
    </>
  );
};
