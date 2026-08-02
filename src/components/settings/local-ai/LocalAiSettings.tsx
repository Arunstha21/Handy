import React, { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { ask } from "@tauri-apps/plugin-dialog";
import { Cpu, Database, Download, HardDrive, MemoryStick, Trash2 } from "lucide-react";
import {
  commands,
  type LocalTextModelInfo,
  type LocalTextModelKind,
  type LocalTextUsageSnapshot,
} from "@/bindings";
import { Button } from "../../ui/Button";
import { Input } from "../../ui/Input";
import { SettingsGroup } from "../../ui/SettingsGroup";

const formatBytes = (bytes: number | null | undefined): string => {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
};

const initialUsage: LocalTextUsageSnapshot = {
  loaded_model_id: null,
  total_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  disk_bytes: 0,
  process_memory_bytes: 0,
  process_cpu_percent: 0,
  gpu_offload_supported: false,
  gpu_layers: 0,
  estimated_device_memory_bytes: null,
  gpu_utilization_percent: null,
  models: [],
};

export const LocalAiSettings: React.FC = () => {
  const { t } = useTranslation();
  const [models, setModels] = useState<LocalTextModelInfo[]>([]);
  const [usage, setUsage] = useState(initialUsage);
  const [name, setName] = useState("TranslateGemma 4B");
  const [sourceUrl, setSourceUrl] = useState("");
  const [sha256, setSha256] = useState("");
  const [kind, setKind] = useState<LocalTextModelKind>("translate_gemma");
  const [busyModelId, setBusyModelId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [modelsResult, usageResult] = await Promise.all([
      commands.getLocalTextModels(),
      commands.getLocalTextUsage(),
    ]);
    if (modelsResult.status === "ok") setModels(modelsResult.data);
    if (usageResult.status === "ok") setUsage(usageResult.data);
    if (modelsResult.status === "error") setError(modelsResult.error);
    if (usageResult.status === "error") setError(usageResult.error);
  }, []);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(interval);
  }, [refresh]);

  const handleAdd = async () => {
    setError(null);
    if (!sourceUrl.trim()) {
      setError(t("settings.localAi.errors.urlRequired"));
      return;
    }
    const result = await commands.addLocalTextModel(
      name.trim(),
      sourceUrl.trim(),
      kind,
      sha256.trim() || null,
    );
    if (result.status === "error") {
      setError(result.error);
      return;
    }
    setSourceUrl("");
    setSha256("");
    await refresh();
  };

  const handleModelAction = async (
    model: LocalTextModelInfo,
    action: "download" | "load" | "unload" | "delete",
  ) => {
    setError(null);
    setBusyModelId(model.id);
    try {
      if (action === "delete") {
        const confirmed = await ask(
          t("settings.localAi.models.deleteConfirm", { name: model.name }),
          { title: t("settings.localAi.models.deleteTitle"), kind: "warning" },
        );
        if (!confirmed) return;
      }
      const result =
        action === "download"
          ? await commands.downloadLocalTextModel(model.id)
          : action === "load"
            ? await commands.loadLocalTextModel(model.id)
            : action === "unload"
              ? await commands.unloadLocalTextModel()
              : await commands.deleteLocalTextModel(model.id);
      if (result.status === "error") setError(result.error);
      await refresh();
    } finally {
      setBusyModelId(null);
    }
  };

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <SettingsGroup title={t("settings.localAi.title")}>
        <div className="px-4 py-3 text-sm text-mid-gray border-b border-mid-gray/20">
          {t("settings.localAi.description")}
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 px-4 py-3">
          <MetricCard
            icon={<Cpu className="w-4 h-4" />}
            label={t("settings.localAi.usage.cpu")}
            value={`${usage.process_cpu_percent.toFixed(1)}%`}
          />
          <MetricCard
            icon={<MemoryStick className="w-4 h-4" />}
            label={t("settings.localAi.usage.memory")}
            value={formatBytes(usage.process_memory_bytes)}
          />
          <MetricCard
            icon={<HardDrive className="w-4 h-4" />}
            label={t("settings.localAi.usage.disk")}
            value={formatBytes(usage.disk_bytes)}
          />
          <MetricCard
            icon={<Database className="w-4 h-4" />}
            label={t("settings.localAi.usage.requests")}
            value={usage.total_requests.toString()}
          />
        </div>

        <div className="px-4 pb-3 text-xs text-mid-gray">
          {usage.gpu_offload_supported && usage.gpu_layers !== 0
            ? t("settings.localAi.usage.gpuOffload", {
                layers: usage.gpu_layers < 0 ? "all" : usage.gpu_layers,
                memory: formatBytes(usage.estimated_device_memory_bytes),
              })
            : t("settings.localAi.usage.gpuUnavailable")}
        </div>

        <div className="px-4 pb-4 text-xs text-mid-gray">
          {t("settings.localAi.usage.note")}
        </div>
      </SettingsGroup>

      <SettingsGroup title={t("settings.localAi.add.title")}>
        <div className="space-y-3 px-4 py-3">
          <p className="text-sm text-mid-gray">{t("settings.localAi.add.description")}</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder={t("settings.localAi.add.namePlaceholder")}
              aria-label={t("settings.localAi.add.nameLabel")}
            />
            <Input
              value={sourceUrl}
              onChange={(event) => setSourceUrl(event.target.value)}
              placeholder={t("settings.localAi.add.urlPlaceholder")}
              aria-label={t("settings.localAi.add.urlLabel")}
              type="url"
            />
            <Input
              value={sha256}
              onChange={(event) => setSha256(event.target.value)}
              placeholder={t("settings.localAi.add.shaPlaceholder")}
              aria-label={t("settings.localAi.add.shaLabel")}
              className="font-mono"
            />
            <select
              value={kind}
              onChange={(event) => setKind(event.target.value as LocalTextModelKind)}
              className="px-3 py-2 text-sm bg-mid-gray/10 border border-mid-gray/80 rounded-md"
              aria-label={t("settings.localAi.add.kindLabel")}
            >
              <option value="translate_gemma">{t("settings.localAi.add.translateGemma")}</option>
              <option value="generic">{t("settings.localAi.add.generic")}</option>
            </select>
          </div>
          <Button onClick={handleAdd} disabled={!sourceUrl.trim()}>
            {t("settings.localAi.add.button")}
          </Button>
          {error && <p className="text-sm text-red-400">{error}</p>}
        </div>
      </SettingsGroup>

      <SettingsGroup title={t("settings.localAi.models.title")}>
        {models.length === 0 ? (
          <p className="px-4 py-4 text-sm text-mid-gray">{t("settings.localAi.models.empty")}</p>
        ) : (
          <div className="divide-y divide-mid-gray/20">
            {models.map((model) => {
              const usageForModel = usage.models.find((item) => item.model_id === model.id);
              const percentage = model.download_total_bytes
                ? Math.min(100, (model.downloaded_bytes / model.download_total_bytes) * 100)
                : 0;
              const busy = busyModelId === model.id;
              return (
                <div key={model.id} className="px-4 py-3 space-y-2">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-semibold truncate">{model.name}</p>
                        {model.is_loaded && (
                          <span className="text-xs text-logo-primary">{t("settings.localAi.models.loaded")}</span>
                        )}
                      </div>
                      <p className="text-xs text-mid-gray truncate">{model.filename}</p>
                      <p className="text-xs text-mid-gray">
                        {model.is_downloaded ? formatBytes(model.size_bytes) : t("settings.localAi.models.notDownloaded")}
                        {usageForModel && ` · ${usageForModel.request_count} ${t("settings.localAi.models.requests")}`}
                      </p>
                    </div>
                    <div className="flex shrink-0 gap-2">
                      {!model.is_downloaded && (
                        <Button
                          size="sm"
                          variant="primary-soft"
                          disabled={busy || model.is_downloading}
                          onClick={() => void handleModelAction(model, "download")}
                        >
                          <span className="flex items-center gap-1"><Download className="w-3 h-3" />{t("settings.localAi.models.download")}</span>
                        </Button>
                      )}
                      {model.is_downloaded && (
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={busy}
                          onClick={() => void handleModelAction(model, model.is_loaded ? "unload" : "load")}
                        >
                          {model.is_loaded ? t("settings.localAi.models.unload") : t("settings.localAi.models.load")}
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="danger-ghost"
                        disabled={busy || model.is_downloading}
                        onClick={() => void handleModelAction(model, "delete")}
                        aria-label={t("settings.localAi.models.delete")}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                  {model.is_downloading && (
                    <div className="space-y-1">
                      <div className="h-1.5 rounded-full bg-mid-gray/20 overflow-hidden">
                        <div className="h-full bg-logo-primary" style={{ width: `${percentage}%` }} />
                      </div>
                      <p className="text-xs text-mid-gray">{formatBytes(model.downloaded_bytes)}</p>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </SettingsGroup>
    </div>
  );
};

const MetricCard: React.FC<{ icon: React.ReactNode; label: string; value: string }> = ({
  icon,
  label,
  value,
}) => (
  <div className="rounded-lg border border-mid-gray/20 bg-mid-gray/5 p-2">
    <div className="flex items-center gap-1 text-xs text-mid-gray">{icon}{label}</div>
    <p className="text-sm font-semibold mt-1">{value}</p>
  </div>
);
