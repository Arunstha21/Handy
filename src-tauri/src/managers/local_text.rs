//! Embedded local text-model runtime and model lifecycle.
//!
//! Speech models intentionally remain owned by [`super::model::ModelManager`].
//! This manager owns GGUF text models only, so users can install, load, inspect,
//! and remove a translation/post-processing model without touching ASR models.

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;
use sysinfo::{Pid, ProcessesToUpdate, System};
use tauri::{AppHandle, Emitter};
use tokio::io::AsyncWriteExt;

const MODELS_FILE: &str = "local-text-models.json";
const USAGE_FILE: &str = "local-text-usage.json";
const MAX_OUTPUT_TOKENS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "snake_case")]
pub enum LocalTextModelKind {
    Generic,
    TranslateGemma,
}

#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
pub struct LocalTextModelInfo {
    pub id: String,
    pub name: String,
    pub filename: String,
    pub source_url: String,
    pub expected_sha256: Option<String>,
    pub kind: LocalTextModelKind,
    pub size_bytes: Option<u64>,
    pub is_downloaded: bool,
    pub is_downloading: bool,
    pub downloaded_bytes: u64,
    pub download_total_bytes: Option<u64>,
    pub is_loaded: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, specta::Type)]
pub struct LocalTextModelUsage {
    pub model_id: String,
    pub request_count: u64,
    pub successful_request_count: u64,
    pub failed_request_count: u64,
    pub input_characters: u64,
    pub output_characters: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_latency_ms: u64,
    pub last_latency_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
pub struct LocalTextUsageSnapshot {
    pub loaded_model_id: Option<String>,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub disk_bytes: u64,
    pub process_memory_bytes: u64,
    pub process_cpu_percent: f32,
    pub gpu_offload_supported: bool,
    pub gpu_layers: i32,
    pub estimated_device_memory_bytes: Option<u64>,
    /// Exact live GPU utilisation is not exposed consistently by macOS,
    /// Windows, and Linux without platform-specific privileged APIs. `None`
    /// is intentional; the dashboard shows offload state and device estimates.
    pub gpu_utilization_percent: Option<f32>,
    pub models: Vec<LocalTextModelUsage>,
}

#[derive(Debug, Clone, Serialize)]
struct LocalTextDownloadProgress {
    model_id: String,
    downloaded: u64,
    total: Option<u64>,
}

struct LoadedLocalTextModel {
    model_id: String,
    model: LlamaModel,
    gpu_layers: i32,
}

struct UsageUpdate {
    successful: bool,
    input_characters: u64,
    output_characters: u64,
    input_tokens: u64,
    output_tokens: u64,
    latency_ms: u64,
}

pub struct LocalTextModelManager {
    app_handle: AppHandle,
    models_dir: PathBuf,
    models_path: PathBuf,
    usage_path: PathBuf,
    models: Mutex<HashMap<String, LocalTextModelInfo>>,
    usage: Mutex<HashMap<String, LocalTextModelUsage>>,
    backend: Mutex<Option<LlamaBackend>>,
    loaded_model: Mutex<Option<LoadedLocalTextModel>>,
    system: Mutex<System>,
}

impl LocalTextModelManager {
    pub fn new(app_handle: &AppHandle) -> Result<Self> {
        let data_dir = crate::portable::app_data_dir(app_handle)
            .map_err(|error| anyhow!("Failed to get app data directory: {error}"))?;
        let models_dir = data_dir.join("text-models");
        fs::create_dir_all(&models_dir)
            .with_context(|| format!("Failed to create {}", models_dir.display()))?;

        let models_path = data_dir.join(MODELS_FILE);
        let usage_path = data_dir.join(USAGE_FILE);
        let mut models = read_json_map::<LocalTextModelInfo>(&models_path).unwrap_or_default();
        for model in models.values_mut() {
            model.is_downloading = false;
            model.download_total_bytes = None;
        }
        let usage = read_json_map::<LocalTextModelUsage>(&usage_path).unwrap_or_default();

        Ok(Self {
            app_handle: app_handle.clone(),
            models_dir,
            models_path,
            usage_path,
            models: Mutex::new(models),
            usage: Mutex::new(usage),
            backend: Mutex::new(None),
            loaded_model: Mutex::new(None),
            system: Mutex::new(System::new()),
        })
    }

    pub fn get_models(&self) -> Vec<LocalTextModelInfo> {
        let loaded_id = self
            .loaded_model
            .lock()
            .ok()
            .and_then(|loaded| loaded.as_ref().map(|model| model.model_id.clone()));
        let models = self
            .models
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let mut result = models
            .values()
            .cloned()
            .map(|mut model| {
                let path = self.models_dir.join(&model.filename);
                model.is_downloaded = path.is_file();
                model.size_bytes = path.metadata().ok().map(|metadata| metadata.len());
                model.is_loaded = loaded_id.as_deref() == Some(model.id.as_str());
                model
            })
            .collect::<Vec<_>>();
        result.sort_by(|left, right| left.name.to_lowercase().cmp(&right.name.to_lowercase()));
        result
    }

    pub fn add_model(
        &self,
        name: String,
        source_url: String,
        kind: LocalTextModelKind,
        expected_sha256: Option<String>,
    ) -> Result<LocalTextModelInfo> {
        let source_url = normalize_model_url(source_url.trim());
        if !matches!(source_url.split(':').next(), Some("https")) {
            return Err(anyhow!("Text model links must use HTTPS"));
        }

        let filename = filename_from_url(&source_url)
            .ok_or_else(|| anyhow!("The model link must point to a .gguf file"))?;
        let id = model_id_for_url(&source_url);
        let name = if name.trim().is_empty() {
            if kind == LocalTextModelKind::TranslateGemma {
                "TranslateGemma 4B".to_string()
            } else {
                filename.trim_end_matches(".gguf").to_string()
            }
        } else {
            name.trim().to_string()
        };
        let expected_sha256 = expected_sha256
            .map(|hash| hash.trim().to_ascii_lowercase())
            .filter(|hash| !hash.is_empty());
        if let Some(hash) = expected_sha256.as_deref() {
            if hash.len() != 64 || !hash.chars().all(|character| character.is_ascii_hexdigit()) {
                return Err(anyhow!("SHA-256 must be a 64-character hexadecimal value"));
            }
        }

        let info = LocalTextModelInfo {
            id: id.clone(),
            name,
            filename,
            source_url,
            expected_sha256,
            kind,
            size_bytes: None,
            is_downloaded: false,
            is_downloading: false,
            downloaded_bytes: 0,
            download_total_bytes: None,
            is_loaded: false,
        };
        self.models
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(id, info.clone());
        self.persist_models()?;
        Ok(info)
    }

    pub async fn download_model(&self, model_id: &str) -> Result<()> {
        let model = self
            .models
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(model_id)
            .cloned()
            .ok_or_else(|| anyhow!("Local text model not found: {model_id}"))?;
        let destination = self.models_dir.join(&model.filename);
        let partial = self.models_dir.join(format!("{}.partial", &model.filename));
        let initial_size = fs::metadata(&partial)
            .map(|metadata| metadata.len())
            .unwrap_or(0);

        self.update_download_state(model_id, true, initial_size, None);
        self.emit_download(model_id, initial_size, None);

        let result = self
            .download_model_inner(&model, &destination, &partial, initial_size)
            .await;

        match result {
            Ok(()) => {
                self.update_download_state(
                    model_id,
                    false,
                    destination
                        .metadata()
                        .map(|metadata| metadata.len())
                        .unwrap_or(0),
                    None,
                );
                self.persist_models()?;
                self.emit_download(
                    model_id,
                    destination
                        .metadata()
                        .map(|metadata| metadata.len())
                        .unwrap_or(0),
                    destination.metadata().ok().map(|metadata| metadata.len()),
                );
                Ok(())
            }
            Err(error) => {
                self.update_download_state(model_id, false, initial_size, None);
                let _ = self.persist_models();
                let _ = self.app_handle.emit(
                    "local-text-model-download-failed",
                    serde_json::json!({ "model_id": model_id, "error": error.to_string() }),
                );
                Err(error)
            }
        }
    }

    async fn download_model_inner(
        &self,
        model: &LocalTextModelInfo,
        destination: &Path,
        partial: &Path,
        initial_size: u64,
    ) -> Result<()> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(20))
            .build()?;
        let mut request = client.get(&model.source_url);
        if initial_size > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={initial_size}-"));
        }
        let response = request.send().await?;
        let append = initial_size > 0 && response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        if !response.status().is_success() {
            return Err(anyhow!(
                "Model download failed with HTTP {}",
                response.status()
            ));
        }

        let starting_size = if append { initial_size } else { 0 };
        let total = response
            .content_length()
            .map(|length| length + starting_size);
        let mut file = if append {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(partial)
                .await?
        } else {
            tokio::fs::File::create(partial).await?
        };
        let mut downloaded = starting_size;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            self.update_download_state(model.id.as_str(), true, downloaded, total);
            self.emit_download(model.id.as_str(), downloaded, total);
        }
        file.flush().await?;
        drop(file);

        if let Some(expected_size) = total {
            let actual_size = tokio::fs::metadata(partial).await?.len();
            if actual_size != expected_size {
                return Err(anyhow!(
                    "Model download ended early: expected {expected_size} bytes, got {actual_size}"
                ));
            }
        }
        if let Some(expected_sha256) = model.expected_sha256.as_deref() {
            let path = partial.to_path_buf();
            let expected_sha256 = expected_sha256.to_string();
            tokio::task::spawn_blocking(move || verify_sha256(&path, &expected_sha256))
                .await
                .map_err(|error| anyhow!("SHA256 verification task failed: {error}"))??;
        }
        tokio::fs::rename(partial, destination).await?;
        Ok(())
    }

    pub fn delete_model(&self, model_id: &str) -> Result<()> {
        let loaded_id = self
            .loaded_model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .map(|model| model.model_id.clone());
        if loaded_id.as_deref() == Some(model_id) {
            self.unload_model()?;
        }

        let model = self
            .models
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(model_id)
            .ok_or_else(|| anyhow!("Local text model not found: {model_id}"))?;
        let path = self.models_dir.join(&model.filename);
        let partial = self.models_dir.join(format!("{}.partial", &model.filename));
        if path.is_dir() {
            fs::remove_dir_all(path)?;
        } else if path.exists() {
            fs::remove_file(path)?;
        }
        if partial.exists() {
            fs::remove_file(partial)?;
        }
        self.persist_models()
    }

    pub fn load_model(&self, model_id: &str) -> Result<()> {
        let model = self
            .models
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(model_id)
            .cloned()
            .ok_or_else(|| anyhow!("Local text model not found: {model_id}"))?;
        let path = self.models_dir.join(&model.filename);
        if !path.is_file() {
            return Err(anyhow!(
                "Local text model is not downloaded: {}",
                model.name
            ));
        }

        let current_model_id = self
            .loaded_model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .map(|loaded| loaded.model_id.clone());
        if current_model_id.as_deref() == Some(model_id) {
            return Ok(());
        }
        // Drop the previous model before loading another one so switching does
        // not briefly hold both multi-gigabyte weight sets in memory.
        if current_model_id.is_some() {
            self.unload_model()?;
        }

        let mut backend = self
            .backend
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if backend.is_none() {
            *backend =
                Some(LlamaBackend::init().map_err(|error| {
                    anyhow!("Failed to initialize embedded LLM runtime: {error}")
                })?);
        }
        let backend_ref = backend.as_ref().expect("backend initialized above");
        let mut params = LlamaModelParams::default();
        if !backend_ref.supports_gpu_offload() {
            params = params.with_n_gpu_layers(0);
        }
        let gpu_layers = params.n_gpu_layers();
        let loaded = LlamaModel::load_from_file(backend_ref, &path, &params)
            .map_err(|error| anyhow!("Failed to load {}: {error}", model.name))?;
        *self
            .loaded_model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(LoadedLocalTextModel {
            model_id: model_id.to_string(),
            model: loaded,
            gpu_layers,
        });
        Ok(())
    }

    pub fn unload_model(&self) -> Result<()> {
        self.loaded_model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        Ok(())
    }

    pub fn complete(&self, model_id: &str, prompt: &str) -> Result<String> {
        let loaded_model_id = self
            .loaded_model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .map(|model| model.model_id.clone());
        if loaded_model_id.as_deref() != Some(model_id) {
            self.load_model(model_id)?;
        }
        let started = Instant::now();
        let result = self.complete_inner(model_id, prompt);
        let latency_ms = started.elapsed().as_millis() as u64;
        match &result {
            Ok((output, input_tokens, output_tokens)) => self.record_usage(
                model_id,
                UsageUpdate {
                    successful: true,
                    input_characters: prompt.chars().count() as u64,
                    output_characters: output.chars().count() as u64,
                    input_tokens: *input_tokens as u64,
                    output_tokens: *output_tokens as u64,
                    latency_ms,
                },
            ),
            Err(_) => self.record_usage(
                model_id,
                UsageUpdate {
                    successful: false,
                    input_characters: prompt.chars().count() as u64,
                    output_characters: 0,
                    input_tokens: 0,
                    output_tokens: 0,
                    latency_ms,
                },
            ),
        }
        result.map(|(output, _, _)| output)
    }

    fn complete_inner(&self, model_id: &str, prompt: &str) -> Result<(String, usize, usize)> {
        let model_info = self
            .models
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(model_id)
            .cloned()
            .ok_or_else(|| anyhow!("Local text model not found: {model_id}"))?;
        let backend = self
            .backend
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let backend = backend
            .as_ref()
            .ok_or_else(|| anyhow!("Embedded LLM runtime is not initialized"))?;
        let loaded = self
            .loaded_model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let loaded = loaded
            .as_ref()
            .filter(|loaded| loaded.model_id == model_id)
            .ok_or_else(|| anyhow!("Load the local text model before using it"))?;

        let formatted_prompt = format_prompt(&loaded.model, &model_info, prompt)?;
        let prompt_tokens = loaded
            .model
            .str_to_token(&formatted_prompt, AddBos::Never)
            .map_err(|error| anyhow!("Failed to tokenize local text prompt: {error}"))?;
        if prompt_tokens.len() >= 512 {
            return Err(anyhow!(
                "Local text prompt is too long for the embedded context ({} tokens)",
                prompt_tokens.len()
            ));
        }
        let max_output_tokens = MAX_OUTPUT_TOKENS.min(511 - prompt_tokens.len());
        let mut context = loaded
            .model
            .new_context(backend, LlamaContextParams::default())
            .map_err(|error| anyhow!("Failed to create local text context: {error}"))?;
        let mut batch = LlamaBatch::new(prompt_tokens.len() + max_output_tokens, 1);
        batch
            .add_sequence(&prompt_tokens, 0, false)
            .map_err(|error| anyhow!("Failed to prepare local text prompt: {error}"))?;
        context
            .decode(&mut batch)
            .map_err(|error| anyhow!("Local text prompt evaluation failed: {error}"))?;

        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::temp(0.1), LlamaSampler::greedy()]);
        let mut output = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        for generated in 0..max_output_tokens {
            let token = sampler.sample(&context, -1);
            sampler.accept(token);
            if loaded.model.is_eog_token(token) {
                break;
            }
            output.push_str(
                &loaded
                    .model
                    .token_to_piece(token, &mut decoder, false, None)
                    .map_err(|error| anyhow!("Failed to decode local text output: {error}"))?,
            );
            batch.clear();
            batch
                .add(token, (prompt_tokens.len() + generated) as i32, &[0], true)
                .map_err(|error| anyhow!("Failed to append local text output: {error}"))?;
            context
                .decode(&mut batch)
                .map_err(|error| anyhow!("Local text generation failed: {error}"))?;
        }

        Ok((
            output.trim().to_string(),
            prompt_tokens.len(),
            output.chars().count(),
        ))
    }

    pub fn get_usage(&self) -> LocalTextUsageSnapshot {
        let loaded_model = self
            .loaded_model
            .lock()
            .ok()
            .and_then(|loaded| loaded.as_ref().map(|model| model.model_id.clone()));
        let gpu_layers = self
            .loaded_model
            .lock()
            .ok()
            .and_then(|loaded| loaded.as_ref().map(|model| model.gpu_layers))
            .unwrap_or(0);
        let gpu_offload_supported = self
            .backend
            .lock()
            .ok()
            .and_then(|backend| backend.as_ref().map(LlamaBackend::supports_gpu_offload))
            .unwrap_or(false);
        let managed_models = self.get_models();
        let disk_bytes = managed_models
            .iter()
            .filter_map(|model| model.size_bytes)
            .sum();
        let estimated_device_memory_bytes = if gpu_layers != 0 {
            loaded_model.as_deref().and_then(|loaded_id| {
                managed_models
                    .iter()
                    .find(|model| model.id == loaded_id)
                    .and_then(|model| model.size_bytes)
            })
        } else {
            None
        };
        let (process_memory_bytes, process_cpu_percent) = self.process_usage();
        let models = self
            .usage
            .lock()
            .map(|usage| usage.values().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let total_requests = models.iter().map(|usage| usage.request_count).sum();
        let successful_requests = models
            .iter()
            .map(|usage| usage.successful_request_count)
            .sum();
        let failed_requests = models.iter().map(|usage| usage.failed_request_count).sum();

        LocalTextUsageSnapshot {
            loaded_model_id: loaded_model,
            total_requests,
            successful_requests,
            failed_requests,
            disk_bytes,
            process_memory_bytes,
            process_cpu_percent,
            gpu_offload_supported,
            gpu_layers,
            estimated_device_memory_bytes,
            gpu_utilization_percent: None,
            models,
        }
    }

    fn process_usage(&self) -> (u64, f32) {
        let mut system = self
            .system
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let pid = Pid::from_u32(std::process::id());
        system.refresh_processes(ProcessesToUpdate::Some(&[pid]), false);
        system
            .process(pid)
            .map(|process| (process.memory(), process.cpu_usage()))
            .unwrap_or((0, 0.0))
    }

    fn record_usage(&self, model_id: &str, update: UsageUpdate) {
        let mut usage = self
            .usage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let record = usage
            .entry(model_id.to_string())
            .or_insert_with(|| LocalTextModelUsage {
                model_id: model_id.to_string(),
                ..Default::default()
            });
        record.request_count += 1;
        if update.successful {
            record.successful_request_count += 1;
            record.input_characters += update.input_characters;
            record.output_characters += update.output_characters;
            record.input_tokens += update.input_tokens;
            record.output_tokens += update.output_tokens;
        } else {
            record.failed_request_count += 1;
        }
        record.total_latency_ms += update.latency_ms;
        record.last_latency_ms = Some(update.latency_ms);
        if let Err(error) = write_json(&self.usage_path, &*usage) {
            log::warn!("Failed to persist local text usage: {error}");
        }
    }

    fn update_download_state(
        &self,
        model_id: &str,
        is_downloading: bool,
        downloaded_bytes: u64,
        total: Option<u64>,
    ) {
        if let Ok(mut models) = self.models.lock() {
            if let Some(model) = models.get_mut(model_id) {
                model.is_downloading = is_downloading;
                model.downloaded_bytes = downloaded_bytes;
                model.download_total_bytes = total;
            }
        }
    }

    fn emit_download(&self, model_id: &str, downloaded: u64, total: Option<u64>) {
        let _ = self.app_handle.emit(
            "local-text-model-download-progress",
            LocalTextDownloadProgress {
                model_id: model_id.to_string(),
                downloaded,
                total,
            },
        );
    }

    fn persist_models(&self) -> Result<()> {
        let models = self
            .models
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        write_json(&self.models_path, &*models)
    }
}

fn format_prompt(
    model: &LlamaModel,
    model_info: &LocalTextModelInfo,
    prompt: &str,
) -> Result<String> {
    let message = LlamaChatMessage::new("user".to_string(), prompt.to_string())
        .map_err(|error| anyhow!("Invalid local text prompt: {error}"))?;
    match model.chat_template(None) {
        Ok(template) => match model.apply_chat_template(&template, &[message], true) {
            Ok(formatted) => Ok(formatted),
            Err(error) => {
                log::warn!(
                    "Falling back to a plain prompt for {} because its chat template failed: {}",
                    model_info.name,
                    error
                );
                Ok(prompt.to_string())
            }
        },
        Err(_) => Ok(prompt.to_string()),
    }
}

fn filename_from_url(source_url: &str) -> Option<String> {
    let path = source_url.split('?').next()?.trim_end_matches('/');
    let filename = path.rsplit('/').next()?.trim();
    if !filename.to_ascii_lowercase().ends_with(".gguf") {
        return None;
    }
    let sanitized = filename
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    (!sanitized.is_empty() && sanitized != ".gguf").then_some(sanitized)
}

fn normalize_model_url(source_url: &str) -> String {
    if let Some((prefix, suffix)) = source_url.split_once("/blob/") {
        if prefix.contains("huggingface.co/") {
            return format!("{prefix}/resolve/{suffix}");
        }
    }
    source_url.to_string()
}

fn model_id_for_url(source_url: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(source_url.as_bytes());
    let hex = format!("{:x}", digest.finalize());
    format!("local-{}", &hex[..16])
}

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != expected {
        let _ = fs::remove_file(path);
        return Err(anyhow!("SHA256 verification failed for local text model"));
    }
    Ok(())
}

fn read_json_map<T>(path: &Path) -> Option<HashMap<String, T>>
where
    T: for<'de> Deserialize<'de>,
{
    fs::read_to_string(path)
        .ok()
        .and_then(|contents| serde_json::from_str(&contents).ok())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let serialized = serde_json::to_string_pretty(value)?;
    fs::write(path, serialized)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_gguf_urls_are_accepted() {
        assert_eq!(
            filename_from_url("https://example.com/model.gguf"),
            Some("model.gguf".to_string())
        );
        assert_eq!(
            filename_from_url("https://example.com/model.gguf?download=1"),
            Some("model.gguf".to_string())
        );
        assert!(filename_from_url("https://example.com/model.safetensors").is_none());
    }

    #[test]
    fn hugging_face_file_pages_are_normalized_to_download_urls() {
        assert_eq!(
            normalize_model_url("https://huggingface.co/acme/model/blob/main/model.gguf"),
            "https://huggingface.co/acme/model/resolve/main/model.gguf"
        );
        assert_eq!(
            normalize_model_url("https://example.com/model.gguf"),
            "https://example.com/model.gguf"
        );
    }

    #[test]
    fn model_ids_are_stable_and_not_the_full_url() {
        let first = model_id_for_url("https://example.com/model.gguf");
        assert_eq!(first, model_id_for_url("https://example.com/model.gguf"));
        assert!(first.starts_with("local-"));
        assert!(first.len() < 32);
    }
}
