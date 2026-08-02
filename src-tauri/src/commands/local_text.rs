use crate::managers::local_text::{
    LocalTextModelInfo, LocalTextModelKind, LocalTextModelManager, LocalTextUsageSnapshot,
};
use std::sync::Arc;
use tauri::State;

#[tauri::command]
#[specta::specta]
pub async fn get_local_text_models(
    manager: State<'_, Arc<LocalTextModelManager>>,
) -> Result<Vec<LocalTextModelInfo>, String> {
    Ok(manager.get_models())
}

#[tauri::command]
#[specta::specta]
pub async fn add_local_text_model(
    manager: State<'_, Arc<LocalTextModelManager>>,
    name: String,
    source_url: String,
    kind: LocalTextModelKind,
    expected_sha256: Option<String>,
) -> Result<LocalTextModelInfo, String> {
    manager
        .add_model(name, source_url, kind, expected_sha256)
        .map_err(|error| error.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn download_local_text_model(
    manager: State<'_, Arc<LocalTextModelManager>>,
    model_id: String,
) -> Result<(), String> {
    manager
        .download_model(&model_id)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn delete_local_text_model(
    manager: State<'_, Arc<LocalTextModelManager>>,
    model_id: String,
) -> Result<(), String> {
    manager
        .delete_model(&model_id)
        .map_err(|error| error.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn load_local_text_model(
    manager: State<'_, Arc<LocalTextModelManager>>,
    model_id: String,
) -> Result<(), String> {
    let manager = manager.inner().clone();
    tokio::task::spawn_blocking(move || manager.load_model(&model_id))
        .await
        .map_err(|error| format!("Local text model load task panicked: {error}"))?
        .map_err(|error| error.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn unload_local_text_model(
    manager: State<'_, Arc<LocalTextModelManager>>,
) -> Result<(), String> {
    manager.unload_model().map_err(|error| error.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn get_local_text_usage(
    manager: State<'_, Arc<LocalTextModelManager>>,
) -> Result<LocalTextUsageSnapshot, String> {
    Ok(manager.get_usage())
}
