use crate::input;
use crate::settings;
use crate::settings::{OverlayPosition, OverlayStyle};
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Listener, Manager, PhysicalPosition, PhysicalSize};

use log::{debug, error};

#[cfg(not(target_os = "macos"))]
use tauri::WebviewWindowBuilder;

#[cfg(target_os = "macos")]
use tauri::WebviewUrl;

#[cfg(target_os = "macos")]
use tauri_nspanel::{tauri_panel, CollectionBehavior, PanelBuilder, PanelLevel, StyleMask};

#[cfg(target_os = "macos")]
use objc2::MainThreadMarker as ObjcMainThreadMarker;
#[cfg(target_os = "macos")]
use objc2_app_kit::NSScreen;

#[cfg(target_os = "linux")]
use gtk_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};

#[cfg(target_os = "linux")]
use std::env;

#[cfg(target_os = "macos")]
tauri_panel! {
    panel!(RecordingOverlayPanel {
        config: {
            can_become_key_window: false,
            is_floating_panel: true
        }
    })
    panel!(SelectionTranslationOverlayPanel {
        config: {
            can_become_key_window: false,
            is_floating_panel: true
        }
    })
}

// Native overlay window sizes (logical points). One window is reused for every
// state and resized in `show_overlay_state`; each size need only be at least as
// large as the card it hosts (the `--ov-*` vars in RecordingOverlay.css). The
// card is CSS-anchored flush to the screen edge, so window height doesn't move
// where the card sits — only OVERLAY_TOP_OFFSET / OVERLAY_BOTTOM_OFFSET do. Keep
// these in sync with the CSS card geometry.
//
// Compact overlay (Minimal / transcribing / processing): the 40h pill animates
// width from 172 (--ov-rest-w) to 216 (--ov-work-w) and expands from center, so
// the window must fit the widest state plus a little slack.
const OVERLAY_WIDTH: f64 = 256.0;
const OVERLAY_HEIGHT: f64 = 46.0;

// Actual is 394x118, just a little extra
const OVERLAY_STREAM_WIDTH: f64 = 400.0;
const OVERLAY_STREAM_HEIGHT: f64 = 120.0;

// A separate, non-activating card positioned beneath the cursor. It is not
// coupled to the recording-overlay preference because selected-text translation
// must remain visible even when speech overlays are disabled.
const SELECTION_TRANSLATION_OVERLAY_WIDTH: f64 = 440.0;
const SELECTION_TRANSLATION_OVERLAY_HEIGHT: f64 = 148.0;
const SELECTION_TRANSLATION_CURSOR_GAP: f64 = 18.0;
const SELECTION_TRANSLATION_EDGE_GAP: f64 = 10.0;

// On a notched MacBook the webview reserves the largest footprint each form can
// animate into. The visible island is still sized from measured housing geometry
// in CSS, but the native window must not shrink with the resting state or the
// working/open morph gets clipped at the webview boundary.
// Webview reserves room for the widest/tallest island state so width morphs
// (rest → work → open) are not clipped. Visible size is driven by CSS.
// Compact notch states can grow a status shelf beneath the camera-safe row.
const OVERLAY_NOTCH_WIDTH: f64 = 480.0;
const OVERLAY_NOTCH_HEIGHT: f64 = 80.0;
const OVERLAY_NOTCH_STREAM_WIDTH: f64 = 520.0;
const OVERLAY_NOTCH_STREAM_HEIGHT: f64 = 148.0;

const DEFAULT_NOTCH_INSET: f64 = 32.0;
/// Approximate MacBook Dynamic Island / camera-housing width in logical points.
const DEFAULT_NOTCH_HOUSING_WIDTH: f64 = 126.0;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct NotchPresentation {
    safe_area_top: f64,
    housing_width: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct OverlayPresentation {
    state: String,
    placement: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    notch: Option<NotchPresentation>,
}

fn normalized_notch_presentation(
    safe_area_top: f64,
    housing_width: Option<f64>,
) -> NotchPresentation {
    NotchPresentation {
        // Current MacBook notch depths are around the low 30s. Keep corrupted or
        // future API values from pushing the visible body outside our window.
        safe_area_top: safe_area_top.clamp(20.0, 42.0),
        housing_width: housing_width
            .unwrap_or(DEFAULT_NOTCH_HOUSING_WIDTH)
            .clamp(110.0, 200.0),
    }
}

/// Effective overlay placement for the current display and user preference.
/// Frontend applies notch styling only for [`EffectivePlacement::NotchAttached`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectivePlacement {
    NotchAttached,
    TopFallback,
    Top,
    Bottom,
}

impl EffectivePlacement {
    pub fn as_str(self) -> &'static str {
        match self {
            EffectivePlacement::NotchAttached => "notch_attached",
            EffectivePlacement::TopFallback => "top_fallback",
            EffectivePlacement::Top => "top",
            EffectivePlacement::Bottom => "bottom",
        }
    }

    pub fn uses_notch_geometry(self) -> bool {
        matches!(self, EffectivePlacement::NotchAttached)
    }
}

/// Resolve placement from settings + whether the cursor monitor actually has a notch.
fn resolve_effective_placement(app_handle: &AppHandle) -> EffectivePlacement {
    let settings = settings::get_settings(app_handle);
    match settings.overlay_position {
        OverlayPosition::Bottom => EffectivePlacement::Bottom,
        OverlayPosition::Top => EffectivePlacement::Top,
        OverlayPosition::Notch => {
            #[cfg(target_os = "macos")]
            {
                if let Some(monitor) = get_monitor_with_cursor(app_handle) {
                    if macos_notch_geometry(&monitor).is_some() {
                        return EffectivePlacement::NotchAttached;
                    }
                }
                EffectivePlacement::TopFallback
            }
            #[cfg(not(target_os = "macos"))]
            {
                EffectivePlacement::TopFallback
            }
        }
    }
}

/// Overlay window size (logical) for a given UI state.
fn overlay_dimensions(app_handle: &AppHandle, state: &str) -> (f64, f64) {
    let placement = resolve_effective_placement(app_handle);
    let is_notch = placement.uses_notch_geometry();
    let streaming = state == "streaming";

    if is_notch {
        return if streaming {
            (OVERLAY_NOTCH_STREAM_WIDTH, OVERLAY_NOTCH_STREAM_HEIGHT)
        } else {
            (OVERLAY_NOTCH_WIDTH, OVERLAY_NOTCH_HEIGHT)
        };
    }

    if streaming {
        (OVERLAY_STREAM_WIDTH, OVERLAY_STREAM_HEIGHT)
    } else {
        (OVERLAY_WIDTH, OVERLAY_HEIGHT)
    }
}

static LAST_MIC_LEVEL_EMIT: AtomicU64 = AtomicU64::new(0);
static SELECTION_TRANSLATION_GENERATION: AtomicU64 = AtomicU64::new(0);
const EMIT_THROTTLE_MS: u64 = 33; // ~30 FPS

#[cfg(target_os = "macos")]
const OVERLAY_TOP_OFFSET: f64 = 46.0;
#[cfg(any(target_os = "windows", target_os = "linux"))]
const OVERLAY_TOP_OFFSET: f64 = 4.0;

#[cfg(target_os = "macos")]
const OVERLAY_BOTTOM_OFFSET: f64 = 15.0;

#[cfg(any(target_os = "windows", target_os = "linux"))]
const OVERLAY_BOTTOM_OFFSET: f64 = 40.0;

#[cfg(target_os = "linux")]
fn update_gtk_layer_shell_anchors(overlay_window: &tauri::webview::WebviewWindow) {
    let window_clone = overlay_window.clone();
    let _ = overlay_window.run_on_main_thread(move || {
        // Try to get the GTK window from the Tauri webview
        if let Ok(gtk_window) = window_clone.gtk_window() {
            let settings = settings::get_settings(window_clone.app_handle());
            match settings.overlay_position {
                OverlayPosition::Top => {
                    gtk_window.set_anchor(Edge::Top, true);
                    gtk_window.set_anchor(Edge::Bottom, false);
                }
                OverlayPosition::Bottom => {
                    gtk_window.set_anchor(Edge::Bottom, true);
                    gtk_window.set_anchor(Edge::Top, false);
                }
                OverlayPosition::Notch => {
                    // Layer-shell has no portable camera-housing primitive.
                    // Keep the preference, but place it at the top edge on
                    // Linux rather than silently disabling the overlay.
                    gtk_window.set_anchor(Edge::Top, true);
                    gtk_window.set_anchor(Edge::Bottom, false);
                }
            }
        }
    });
}

/// Returns true when the environment variable is set to a truthy value
/// (e.g. "1", "true", "yes", "on").
/// "0", "false", "no", "off" and empty string are treated as falsy (case-insensitive).
/// Returns false when the variable is not set.
#[cfg(target_os = "linux")]
fn env_flag_enabled(name: &str) -> bool {
    match env::var(name) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        Err(_) => false,
    }
}

/// Initializes GTK layer shell for Linux overlay window
/// Returns true if layer shell was successfully initialized, false otherwise
#[cfg(target_os = "linux")]
fn init_gtk_layer_shell(overlay_window: &tauri::webview::WebviewWindow) -> bool {
    if env_flag_enabled("HANDY_NO_GTK_LAYER_SHELL") {
        debug!("Skipping GTK layer shell init (HANDY_NO_GTK_LAYER_SHELL is enabled)");
        return false;
    }

    if !gtk_layer_shell::is_supported() {
        return false;
    }

    // Try to get the GTK window from the Tauri webview
    if let Ok(gtk_window) = overlay_window.gtk_window() {
        // Initialize layer shell
        gtk_window.init_layer_shell();
        gtk_window.set_layer(Layer::Overlay);
        gtk_window.set_keyboard_mode(KeyboardMode::None);
        gtk_window.set_exclusive_zone(0);

        update_gtk_layer_shell_anchors(overlay_window);

        return true;
    }
    false
}

/// Forces a window to be topmost using Win32 API (Windows only)
/// This is more reliable than Tauri's set_always_on_top which can be overridden
#[cfg(target_os = "windows")]
fn force_overlay_topmost(overlay_window: &tauri::webview::WebviewWindow) {
    use windows::Win32::UI::WindowsAndMessaging::{
        SetWindowPos, HWND_TOPMOST, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SWP_SHOWWINDOW,
    };

    // Clone because run_on_main_thread takes 'static
    let overlay_clone = overlay_window.clone();

    // Make sure the Win32 call happens on the UI thread
    let _ = overlay_clone.clone().run_on_main_thread(move || {
        if let Ok(hwnd) = overlay_clone.hwnd() {
            unsafe {
                // Force Z-order: make this window topmost without changing size/pos or stealing focus
                let _ = SetWindowPos(
                    hwnd,
                    Some(HWND_TOPMOST),
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
                );
            }
        }
    });
}

fn get_monitor_with_cursor(app_handle: &AppHandle) -> Option<tauri::Monitor> {
    if let Some(mouse_location) = input::get_cursor_position(app_handle) {
        if let Ok(monitors) = app_handle.available_monitors() {
            for monitor in monitors {
                // On Windows both the cursor (enigo -> GetCursorPos) and the
                // monitor bounds are physical pixels, so compare them directly.
                #[cfg(target_os = "windows")]
                if is_mouse_within_monitor(mouse_location, monitor.position(), monitor.size()) {
                    return Some(monitor);
                }

                // macOS/Linux: enigo returns logical coords, so scale the bounds down.
                #[cfg(not(target_os = "windows"))]
                {
                    let scale = monitor.scale_factor();
                    let pos = PhysicalPosition::new(
                        (monitor.position().x as f64 / scale) as i32,
                        (monitor.position().y as f64 / scale) as i32,
                    );
                    let size = PhysicalSize::new(
                        (monitor.size().width as f64 / scale) as u32,
                        (monitor.size().height as f64 / scale) as u32,
                    );
                    if is_mouse_within_monitor(mouse_location, &pos, &size) {
                        return Some(monitor);
                    }
                }
            }
        }
    }

    app_handle.primary_monitor().ok().flatten()
}

fn is_mouse_within_monitor(
    mouse_pos: (i32, i32),
    monitor_pos: &PhysicalPosition<i32>,
    monitor_size: &PhysicalSize<u32>,
) -> bool {
    let (mouse_x, mouse_y) = mouse_pos;
    let PhysicalPosition {
        x: monitor_x,
        y: monitor_y,
    } = *monitor_pos;
    let PhysicalSize {
        width: monitor_width,
        height: monitor_height,
    } = *monitor_size;

    mouse_x >= monitor_x
        && mouse_x < (monitor_x + monitor_width as i32)
        && mouse_y >= monitor_y
        && mouse_y < (monitor_y + monitor_height as i32)
}

/// Position the selected-text translation card directly beneath the cursor,
/// clamped to its active monitor. The cursor is normally inside the text range
/// the user just selected, so this keeps the result adjacent without querying
/// app-specific accessibility selection bounds.
fn calculate_selection_translation_overlay_position(
    app_handle: &AppHandle,
    width: f64,
    height: f64,
) -> Option<(f64, f64)> {
    let monitor = get_monitor_with_cursor(app_handle)?;
    let scale = monitor.scale_factor();
    let monitor_x = monitor.position().x as f64 / scale;
    let monitor_y = monitor.position().y as f64 / scale;
    let monitor_width = monitor.size().width as f64 / scale;
    let monitor_height = monitor.size().height as f64 / scale;

    let cursor = input::get_cursor_position(app_handle).map(|(x, y)| {
        #[cfg(target_os = "windows")]
        {
            (x as f64 / scale, y as f64 / scale)
        }
        #[cfg(not(target_os = "windows"))]
        {
            (x as f64, y as f64)
        }
    });
    let (cursor_x, cursor_y) = cursor.unwrap_or((
        monitor_x + monitor_width / 2.0,
        monitor_y + monitor_height / 2.0,
    ));

    let min_x = monitor_x + SELECTION_TRANSLATION_EDGE_GAP;
    let min_y = monitor_y + SELECTION_TRANSLATION_EDGE_GAP;
    let max_x = (monitor_x + monitor_width - width - SELECTION_TRANSLATION_EDGE_GAP).max(min_x);
    let max_y = (monitor_y + monitor_height - height - SELECTION_TRANSLATION_EDGE_GAP).max(min_y);

    Some((
        (cursor_x - width / 2.0).clamp(min_x, max_x),
        (cursor_y + SELECTION_TRANSLATION_CURSOR_GAP).clamp(min_y, max_y),
    ))
}

/// Measured macOS camera-housing geometry for the monitor under the cursor.
///
/// Combines `safeAreaInsets.top` (notch depth) with the gap between the left
/// and right auxiliary top areas when AppKit exposes them. The island width is
/// derived from that gap so it tracks the physical housing instead of a fixed
/// slab size for every MacBook.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
struct NotchGeometry {
    /// Top safe-area inset in logical points (notch depth).
    inset: f64,
    /// Estimated camera-housing width in logical points, when measurable.
    housing_width: Option<f64>,
}

/// Returns notch geometry for the monitor, or `None` when the display has no
/// camera housing (external monitors, older Macs).
#[cfg(target_os = "macos")]
fn macos_notch_geometry(monitor: &tauri::Monitor) -> Option<NotchGeometry> {
    let marker = ObjcMainThreadMarker::new()?;
    let screens = NSScreen::screens(marker);
    let monitor_size = monitor.size();

    let mut best: Option<(f64, NotchGeometry)> = None;
    for index in 0..screens.count() {
        let screen = screens.objectAtIndex(index);
        let scale = screen.backingScaleFactor();
        let frame = screen.frame();
        let width = (frame.size.width * scale).round();
        let height = (frame.size.height * scale).round();
        let dw = (width - monitor_size.width as f64).abs();
        let dh = (height - monitor_size.height as f64).abs();
        let distance = dw + dh;
        let inset = screen.safeAreaInsets().top;

        // Estimate housing width from the gap between auxiliary top regions.
        // auxiliaryTopLeftArea / auxiliaryTopRightArea describe the usable menu-
        // bar strips on either side of the camera housing on notched Macs.
        let housing_width = {
            let left = screen.auxiliaryTopLeftArea();
            let right = screen.auxiliaryTopRightArea();
            let left_end = left.origin.x + left.size.width;
            let right_start = right.origin.x;
            let gap = right_start - left_end;
            // Only trust a positive gap that's smaller than half the screen.
            let frame_w = frame.size.width;
            if gap > 8.0 && gap < frame_w * 0.5 {
                Some(gap)
            } else {
                None
            }
        };

        if best.is_none_or(|(best_distance, _)| distance < best_distance) {
            best = Some((
                distance,
                NotchGeometry {
                    inset,
                    housing_width,
                },
            ));
        }
    }

    let (distance, geometry) = best?;
    // A loose tolerance accommodates AppKit/Tauri rounding at scaled
    // resolutions, while preventing an external display from inheriting the
    // MacBook's notch inset.
    let tolerance = 8.0 * monitor.scale_factor();
    (distance <= tolerance && geometry.inset > 0.0).then_some(geometry)
}

/// Returns overlay position in logical coordinates (points on macOS).
///
/// The Bottom anchor uses the macOS work area (visibleFrame) so the overlay
/// tracks the Dock — above it when shown, at the screen edge when hidden.
/// This relies on tauri 2.11's work_area.position.y fix (#14655), the same
/// bug that led PR #969 to abandon work_area for full monitor bounds. Top and
/// the other platforms keep full monitor bounds plus the fixed offsets
/// (work_area is unreliable on Wayland; Windows' offset clears the taskbar).
///
/// We must use LogicalPosition (not PhysicalPosition) because Tauri/tao
/// converts PhysicalPosition using the scale factor of the monitor the window
/// is *currently* on, which is wrong when moving cross-monitor. Windows uses
/// `place_windows_overlay` instead (no single logical space across mixed DPI).
fn calculate_overlay_position(
    app_handle: &AppHandle,
    width: f64,
    height: f64,
) -> Option<(f64, f64)> {
    let monitor = get_monitor_with_cursor(app_handle)?;
    let scale = monitor.scale_factor();
    let monitor_x = monitor.position().x as f64 / scale;
    let monitor_y = monitor.position().y as f64 / scale;
    let monitor_width = monitor.size().width as f64 / scale;

    let settings = settings::get_settings(app_handle);

    let x = monitor_x + (monitor_width - width) / 2.0;
    let y = match settings.overlay_position {
        OverlayPosition::Top => monitor_y + OVERLAY_TOP_OFFSET,
        OverlayPosition::Notch => {
            #[cfg(target_os = "macos")]
            {
                macos_notch_geometry(&monitor)
                    // Start at the physical display edge, not below the safe
                    // area: the black card is the visual continuation of the
                    // camera housing, like a Dynamic Island.
                    .map(|_| monitor_y)
                    .unwrap_or(monitor_y + OVERLAY_TOP_OFFSET)
            }
            #[cfg(not(target_os = "macos"))]
            {
                monitor_y + OVERLAY_TOP_OFFSET
            }
        }
        OverlayPosition::Bottom => {
            // work_area.position shares monitor.position's global coordinate
            // space, so no monitor offset is added.
            #[cfg(target_os = "macos")]
            let bottom = {
                let wa = monitor.work_area();
                (wa.position.y as f64 + wa.size.height as f64) / scale
            };
            #[cfg(not(target_os = "macos"))]
            let bottom = monitor_y + monitor.size().height as f64 / scale;

            bottom - height - OVERLAY_BOTTOM_OFFSET
        }
    };

    Some((x, y))
}

/// Current overlay window size in logical units (points), for repositioning
/// without assuming a fixed size (compact vs. streaming).
#[cfg(not(target_os = "windows"))]
fn current_overlay_logical_size(window: &tauri::webview::WebviewWindow) -> Option<(f64, f64)> {
    let size = window.inner_size().ok()?;
    let scale = window.scale_factor().ok()?;
    Some((size.width as f64 / scale, size.height as f64 / scale))
}

#[cfg(target_os = "windows")]
static WINDOWS_OVERLAY_IS_STREAMING: AtomicBool = AtomicBool::new(false);

/// Overlay rectangle in the destination monitor's physical pixels, so nothing
/// is converted through the window's previous-monitor DPI.
#[cfg(target_os = "windows")]
fn windows_overlay_bounds(
    monitor_position: PhysicalPosition<i32>,
    monitor_size: PhysicalSize<u32>,
    scale: f64,
    logical_width: f64,
    logical_height: f64,
    overlay_position: OverlayPosition,
) -> (i32, i32, i32, i32) {
    let width = (logical_width * scale).round().max(1.0) as i32;
    let height = (logical_height * scale).round().max(1.0) as i32;
    let x = (monitor_position.x as f64 + (monitor_size.width as f64 - width as f64) / 2.0).round()
        as i32;
    let y = match overlay_position {
        OverlayPosition::Top | OverlayPosition::Notch => {
            (monitor_position.y as f64 + OVERLAY_TOP_OFFSET * scale).round() as i32
        }
        OverlayPosition::Bottom => (monitor_position.y as f64 + monitor_size.height as f64
            - height as f64
            - OVERLAY_BOTTOM_OFFSET * scale)
            .round() as i32,
    };

    (x, y, width, height)
}

/// Moves and sizes the overlay in one native SetWindowPos, bypassing tao's
/// current-DPI logical conversion that mislands cross-monitor moves.
#[cfg(target_os = "windows")]
fn place_windows_overlay(
    app_handle: &AppHandle,
    overlay_window: &tauri::webview::WebviewWindow,
    logical_width: f64,
    logical_height: f64,
) -> Result<(), String> {
    use windows::Win32::UI::WindowsAndMessaging::{SetWindowPos, SWP_NOACTIVATE, SWP_NOZORDER};

    let monitor = get_monitor_with_cursor(app_handle)
        .ok_or_else(|| "failed to determine the monitor containing the cursor".to_string())?;
    let (x, y, width, height) = windows_overlay_bounds(
        *monitor.position(),
        *monitor.size(),
        monitor.scale_factor(),
        logical_width,
        logical_height,
        settings::get_settings(app_handle).overlay_position,
    );
    let hwnd = overlay_window
        .hwnd()
        .map_err(|error| format!("failed to get overlay window handle: {error}"))?;

    unsafe {
        SetWindowPos(
            hwnd,
            None,
            x,
            y,
            width,
            height,
            SWP_NOACTIVATE | SWP_NOZORDER,
        )
        .map_err(|error| format!("failed to set overlay bounds: {error}"))?;
    }

    log::debug!(
        "windows overlay bounds: x={} y={} width={} height={} scale={}",
        x,
        y,
        width,
        height,
        monitor.scale_factor()
    );
    Ok(())
}

/// Creates the recording overlay window and keeps it hidden by default
#[cfg(not(target_os = "macos"))]
pub fn create_recording_overlay(app_handle: &AppHandle) {
    let (width, height) = overlay_dimensions(app_handle, "recording");

    // On Linux (Wayland), monitor detection often fails, but we don't need exact coordinates
    // for Layer Shell as we use anchors. On other platforms, we require a monitor.
    #[cfg(not(target_os = "linux"))]
    {
        let position = calculate_overlay_position(app_handle, width, height);
        if position.is_none() {
            debug!("Failed to determine overlay position, not creating overlay window");
            return;
        }
    }

    // Position starts unset — update_overlay_position() sets the correct
    // LogicalPosition before the overlay is shown.
    let mut builder = WebviewWindowBuilder::new(
        app_handle,
        "recording_overlay",
        tauri::WebviewUrl::App("src/overlay/index.html".into()),
    )
    .title("Recording")
    .resizable(false)
    .inner_size(width, height)
    .shadow(false)
    .maximizable(false)
    .minimizable(false)
    .closable(false)
    .accept_first_mouse(true)
    .decorations(false)
    .always_on_top(true)
    .skip_taskbar(true)
    .transparent(true)
    .focusable(false)
    .focused(false)
    .visible(false);

    if let Some(data_dir) = crate::portable::data_dir() {
        builder = builder.data_directory(data_dir.join("webview"));
    }

    #[allow(unused_variables)]
    match builder.build() {
        Ok(window) => {
            #[cfg(target_os = "linux")]
            {
                // Try to initialize GTK layer shell, ignore errors if compositor doesn't support it
                if init_gtk_layer_shell(&window) {
                    debug!("GTK layer shell initialized for overlay window");
                } else {
                    debug!("GTK layer shell not available, falling back to regular window");
                }
            }

            debug!("Recording overlay window created successfully (hidden)");
        }
        Err(e) => {
            debug!("Failed to create recording overlay window: {}", e);
        }
    }
}

/// Creates the non-activating selected-text translation overlay (non-macOS).
#[cfg(not(target_os = "macos"))]
pub fn create_selection_translation_overlay(app_handle: &AppHandle) {
    register_selection_translation_overlay_ready_listener(app_handle);
    let (width, height) = (
        SELECTION_TRANSLATION_OVERLAY_WIDTH,
        SELECTION_TRANSLATION_OVERLAY_HEIGHT,
    );
    let position = calculate_selection_translation_overlay_position(app_handle, width, height)
        .unwrap_or((0.0, 0.0));
    let mut builder = WebviewWindowBuilder::new(
        app_handle,
        "selection_translation_overlay",
        tauri::WebviewUrl::App("src/overlay/index.html?mode=selection-translation".into()),
    )
    .title("Selected Text Translation")
    .resizable(false)
    .inner_size(width, height)
    .position(position.0, position.1)
    .shadow(false)
    .maximizable(false)
    .minimizable(false)
    .closable(false)
    .decorations(false)
    .always_on_top(true)
    .skip_taskbar(true)
    .transparent(true)
    .focusable(false)
    .focused(false)
    .visible(false);

    if let Some(data_dir) = crate::portable::data_dir() {
        builder = builder.data_directory(data_dir.join("webview"));
    }

    match builder.build() {
        Ok(window) => {
            // The card is informational only. Let clicks pass through to the
            // app whose selection is being translated, and reassert the
            // non-focusable flag after WebView creation on Windows where the
            // native child can otherwise briefly become the active window.
            let _ = window.set_ignore_cursor_events(true);
            let _ = window.set_focusable(false);
        }
        Err(error) => {
            log::error!("Failed to create selected-text translation overlay: {error}");
        }
    }
}

/// Creates the non-activating selected-text translation panel (macOS).
#[cfg(target_os = "macos")]
pub fn create_selection_translation_overlay(app_handle: &AppHandle) {
    register_selection_translation_overlay_ready_listener(app_handle);
    let width = SELECTION_TRANSLATION_OVERLAY_WIDTH;
    let height = SELECTION_TRANSLATION_OVERLAY_HEIGHT;
    let (x, y) = calculate_selection_translation_overlay_position(app_handle, width, height)
        .unwrap_or((0.0, 0.0));
    match PanelBuilder::<_, SelectionTranslationOverlayPanel>::new(
        app_handle,
        "selection_translation_overlay",
    )
    .url(WebviewUrl::App(
        "src/overlay/index.html?mode=selection-translation".into(),
    ))
    .title("Selected Text Translation")
    .position(tauri::Position::Logical(tauri::LogicalPosition { x, y }))
    .level(PanelLevel::Status)
    .size(tauri::Size::Logical(tauri::LogicalSize { width, height }))
    .has_shadow(false)
    .transparent(true)
    .no_activate(true)
    .corner_radius(0.0)
    .style_mask(StyleMask::empty().borderless().nonactivating_panel())
    .with_window(|window| window.decorations(false).transparent(true).focusable(false))
    .collection_behavior(
        CollectionBehavior::new()
            .can_join_all_spaces()
            .full_screen_auxiliary(),
    )
    .build()
    {
        Ok(panel) => {
            panel.hide();
            debug!("Selected-text translation overlay panel created (hidden)");
        }
        Err(error) => log::error!("Failed to create selected-text translation panel: {error}"),
    }
}

/// Creates the recording overlay panel and keeps it hidden by default (macOS)
#[cfg(target_os = "macos")]
pub fn create_recording_overlay(app_handle: &AppHandle) {
    let (width, height) = overlay_dimensions(app_handle, "recording");
    if let Some((x, y)) = calculate_overlay_position(app_handle, width, height) {
        // PanelBuilder creates a Tauri window then converts it to NSPanel.
        // The window remains registered, so get_webview_window() still works.
        match PanelBuilder::<_, RecordingOverlayPanel>::new(app_handle, "recording_overlay")
            .url(WebviewUrl::App("src/overlay/index.html".into()))
            .title("Recording")
            .position(tauri::Position::Logical(tauri::LogicalPosition { x, y }))
            .level(PanelLevel::Status)
            .size(tauri::Size::Logical(tauri::LogicalSize { width, height }))
            .has_shadow(false)
            .transparent(true)
            .no_activate(true)
            .corner_radius(0.0)
            .style_mask(StyleMask::empty().borderless().nonactivating_panel())
            .with_window(|w| w.decorations(false).transparent(true).focusable(false))
            .collection_behavior(
                CollectionBehavior::new()
                    .can_join_all_spaces()
                    .full_screen_auxiliary(),
            )
            .build()
        {
            Ok(panel) => {
                panel.hide();
            }
            Err(e) => {
                log::error!("Failed to create recording overlay panel: {}", e);
            }
        }
    }
}

fn show_overlay_state(app_handle: &AppHandle, state: &str) {
    // Whether the overlay shows at all is governed by overlay_style; position
    // only chooses Top vs Bottom placement. Checked here (off the main thread)
    // so the common overlay-disabled case never pays for a main-thread hop.
    let settings = settings::get_settings(app_handle);
    if settings.overlay_style == OverlayStyle::None {
        return;
    }

    // The rest queries monitors and the cursor and mutates window geometry. On
    // Linux the monitor/cursor lookups hit GDK/Xlib on the process's shared X11
    // connection, which is only safe from the GTK main thread — running them on
    // a background thread corrupts the connection and hard-crashes the app
    // (issue #227). Hop to the main thread on every platform to keep the
    // geometry path uniform (a no-op cost on Windows, and it also keeps macOS's
    // NSScreen access main-thread-correct). run_on_main_thread runs the closure
    // inline when already on the main thread, so this never deadlocks.
    let handle = app_handle.clone();
    let state = state.to_string();
    let _ = app_handle.run_on_main_thread(move || show_overlay_state_on_main(&handle, &state));
}

fn show_overlay_state_on_main(app_handle: &AppHandle, state: &str) {
    // Size the overlay for this state (compact vs. streaming), then position it.
    let (width, height) = overlay_dimensions(app_handle, state);
    if let Some(overlay_window) = app_handle.get_webview_window("recording_overlay") {
        #[cfg(target_os = "linux")]
        update_gtk_layer_shell_anchors(&overlay_window);

        let size_started = std::time::Instant::now();
        #[cfg(not(target_os = "windows"))]
        let _ = overlay_window.set_size(tauri::Size::Logical(tauri::LogicalSize { width, height }));
        #[cfg(target_os = "windows")]
        WINDOWS_OVERLAY_IS_STREAMING.store(state == "streaming", Ordering::Relaxed);
        let size_elapsed = size_started.elapsed();

        let pos_started = std::time::Instant::now();
        #[cfg(not(target_os = "windows"))]
        let set_pos_elapsed =
            if let Some((x, y)) = calculate_overlay_position(app_handle, width, height) {
                let set_pos_started = std::time::Instant::now();
                let _ = overlay_window
                    .set_position(tauri::Position::Logical(tauri::LogicalPosition { x, y }));
                set_pos_started.elapsed()
            } else {
                std::time::Duration::ZERO
            };
        #[cfg(target_os = "windows")]
        let set_pos_elapsed = {
            let set_pos_started = std::time::Instant::now();
            if let Err(error) = place_windows_overlay(app_handle, &overlay_window, width, height) {
                log::error!("Failed to place recording overlay: {error}");
            }
            set_pos_started.elapsed()
        };
        let pos_calc_elapsed = pos_started.elapsed() - set_pos_elapsed;

        let placement = resolve_effective_placement(app_handle);
        #[cfg(target_os = "macos")]
        let notch = if placement == EffectivePlacement::NotchAttached {
            get_monitor_with_cursor(app_handle)
                .and_then(|monitor| macos_notch_geometry(&monitor))
                .map(|geometry| {
                    normalized_notch_presentation(geometry.inset, geometry.housing_width)
                })
                .or_else(|| Some(normalized_notch_presentation(DEFAULT_NOTCH_INSET, None)))
        } else {
            None
        };
        #[cfg(not(target_os = "macos"))]
        let notch = None;

        // Send placement and measured notch geometry as one payload before the
        // native window becomes visible. The overlay starts transparent, so the
        // first painted frame has the correct attached/fallback silhouette.
        let presentation = OverlayPresentation {
            state: state.to_string(),
            placement: placement.as_str().to_string(),
            notch,
        };
        let _ = overlay_window.emit("show-overlay", presentation);
        // Retain the placement-only event for older overlay webviews during a
        // development hot reload.
        let _ = overlay_window.emit("overlay-placement", placement.as_str());

        let show_started = std::time::Instant::now();
        let _ = overlay_window.show();
        let show_elapsed = show_started.elapsed();

        // On Windows, aggressively re-assert "topmost" in the native Z-order after showing
        #[cfg(target_os = "windows")]
        force_overlay_topmost(&overlay_window);

        // Re-assert bounds after show(): the pre-show move crosses the DPI
        // boundary, and tao's WM_DPICHANGED reflow clobbers the first placement.
        #[cfg(target_os = "windows")]
        if let Err(error) = place_windows_overlay(app_handle, &overlay_window, width, height) {
            log::error!("Failed to re-assert recording overlay position: {error}");
        }

        log::debug!(
            "overlay '{}' placement={}: set_size={:?} pos_calc={:?} set_pos={:?} show={:?}",
            state,
            placement.as_str(),
            size_elapsed,
            pos_calc_elapsed,
            set_pos_elapsed,
            show_elapsed
        );
    }
}

/// Shows the recording overlay window with fade-in animation
pub fn show_recording_overlay(app_handle: &AppHandle) {
    show_overlay_state(app_handle, "recording");
}

/// Shows the larger streaming overlay that displays live transcription text
pub fn show_streaming_overlay(app_handle: &AppHandle) {
    show_overlay_state(app_handle, "streaming");
}

/// Shows the transcribing overlay window
pub fn show_transcribing_overlay(app_handle: &AppHandle) {
    show_overlay_state(app_handle, "transcribing");
}

/// Shows the processing overlay window
pub fn show_processing_overlay(app_handle: &AppHandle) {
    show_overlay_state(app_handle, "processing");
}

/// Shows the translating overlay (dedicated translation action).
pub fn show_translating_overlay(app_handle: &AppHandle) {
    show_overlay_state(app_handle, "translating");
}

/// Shows the dual-model verification overlay.
pub fn show_verifying_overlay(app_handle: &AppHandle) {
    show_overlay_state(app_handle, "verifying");
}

/// Updates the overlay window position based on current settings
pub fn update_overlay_position(app_handle: &AppHandle) {
    // Positioning queries monitors/cursor (GDK/Xlib on Linux) and moves the
    // window, so it must run on the main thread — see show_overlay_state.
    let handle = app_handle.clone();
    let _ = app_handle.run_on_main_thread(move || update_overlay_position_on_main(&handle));
}

fn update_overlay_position_on_main(app_handle: &AppHandle) {
    if let Some(overlay_window) = app_handle.get_webview_window("recording_overlay") {
        #[cfg(target_os = "linux")]
        {
            update_gtk_layer_shell_anchors(&overlay_window);
        }

        #[cfg(target_os = "windows")]
        {
            let state = if WINDOWS_OVERLAY_IS_STREAMING.load(Ordering::Relaxed) {
                "streaming"
            } else {
                "recording"
            };
            let (width, height) = overlay_dimensions(app_handle, state);
            if let Err(error) = place_windows_overlay(app_handle, &overlay_window, width, height) {
                log::error!("Failed to update recording overlay position: {error}");
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            // Use the window's current size so centering stays correct whether the
            // overlay is in compact or streaming layout.
            let (width, height) = current_overlay_logical_size(&overlay_window)
                .unwrap_or_else(|| overlay_dimensions(app_handle, "recording"));
            if let Some((x, y)) = calculate_overlay_position(app_handle, width, height) {
                let _ = overlay_window
                    .set_position(tauri::Position::Logical(tauri::LogicalPosition { x, y }));
            }
        }
    }
}

/// Hides the recording overlay window with fade-out animation
pub fn hide_recording_overlay(app_handle: &AppHandle) {
    // Always hide the overlay regardless of settings - if setting was changed while recording,
    // we still want to hide it properly
    if let Some(overlay_window) = app_handle.get_webview_window("recording_overlay") {
        // Emit event to trigger fade-out animation
        let _ = overlay_window.emit("hide-overlay", ());
        // Hide the window after a short delay to allow animation to complete
        let window_clone = overlay_window.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(300));
            let _ = window_clone.hide();
        });
    }
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct SelectionTranslationPresentation {
    state: String,
    target_language: Option<String>,
    text: Option<String>,
}

#[derive(Clone)]
struct PendingSelectionTranslationPresentation {
    presentation: SelectionTranslationPresentation,
    reposition: bool,
    generation: u64,
}

// A hidden WebView may not have attached its frontend event listeners when the
// first shortcut is pressed. Keep the newest state until that WebView explicitly
// reports readiness, then replay it. This also recovers if the overlay WebView
// reloads while a translation is in flight.
static SELECTION_TRANSLATION_READY: AtomicBool = AtomicBool::new(false);
static PENDING_SELECTION_TRANSLATION_PRESENTATION: Lazy<
    Mutex<Option<PendingSelectionTranslationPresentation>>,
> = Lazy::new(|| Mutex::new(None));

fn remember_selection_translation_presentation(
    presentation: PendingSelectionTranslationPresentation,
) {
    match PENDING_SELECTION_TRANSLATION_PRESENTATION.lock() {
        Ok(mut pending) => *pending = Some(presentation),
        Err(lock_error) => {
            error!("Could not retain selected-text translation presentation: {lock_error}")
        }
    }
}

fn pending_selection_translation_presentation() -> Option<PendingSelectionTranslationPresentation> {
    match PENDING_SELECTION_TRANSLATION_PRESENTATION.lock() {
        Ok(pending) => pending.clone(),
        Err(lock_error) => {
            error!("Could not read selected-text translation presentation: {lock_error}");
            None
        }
    }
}

fn clear_selection_translation_presentation(generation: u64) {
    match PENDING_SELECTION_TRANSLATION_PRESENTATION.lock() {
        Ok(mut pending) => {
            if pending
                .as_ref()
                .is_some_and(|presentation| presentation.generation == generation)
            {
                *pending = None;
            }
        }
        Err(lock_error) => {
            error!("Could not clear selected-text translation presentation: {lock_error}")
        }
    }
}

fn register_selection_translation_overlay_ready_listener(app_handle: &AppHandle) {
    let replay_handle = app_handle.clone();
    app_handle.listen("selection-translation-ready", move |_| {
        SELECTION_TRANSLATION_READY.store(true, Ordering::Release);
        debug!("Selected-text translation overlay frontend is ready");
        if let Some(presentation) = pending_selection_translation_presentation() {
            present_selection_translation(&replay_handle, presentation);
        }
    });
}

fn present_selection_translation(
    app_handle: &AppHandle,
    pending: PendingSelectionTranslationPresentation,
) {
    let handle = app_handle.clone();
    if let Err(schedule_error) = app_handle.run_on_main_thread(move || {
        let Some(window) = handle.get_webview_window("selection_translation_overlay") else {
            error!("Selected-text translation overlay window is unavailable");
            return;
        };

        if pending.reposition {
            if let Some((x, y)) = calculate_selection_translation_overlay_position(
                &handle,
                SELECTION_TRANSLATION_OVERLAY_WIDTH,
                SELECTION_TRANSLATION_OVERLAY_HEIGHT,
            ) {
                if let Err(position_error) =
                    window.set_position(tauri::Position::Logical(tauri::LogicalPosition { x, y }))
                {
                    error!(
                        "Could not position selected-text translation overlay: {position_error}"
                    );
                }
            }
        }

        // Show first so an initially hidden WebView has a chance to finish
        // loading. The ready callback below replays this state if the event
        // listener was not attached yet.
        if let Err(show_error) = window.show() {
            error!("Could not show selected-text translation overlay: {show_error}");
            return;
        }

        // `focusable(false)` is set at construction time, but WebView2 can
        // recreate its child window while the hidden overlay is first shown.
        // Reapply the native non-activating/topmost state after every show so
        // the source application's selection remains the foreground target
        // for the synthetic Ctrl+C below.
        let _ = window.set_focusable(false);
        #[cfg(target_os = "windows")]
        force_overlay_topmost(&window);

        if SELECTION_TRANSLATION_READY.load(Ordering::Acquire) {
            if let Err(emit_error) = window.emit("show-selection-translation", pending.presentation)
            {
                error!("Could not update selected-text translation overlay: {emit_error}");
            }
        } else {
            debug!("Selected-text translation overlay is waiting for frontend readiness");
        }
    }) {
        error!("Could not schedule selected-text translation overlay: {schedule_error}");
    }
}

fn dismiss_selection_translation_after(
    app_handle: &AppHandle,
    generation: u64,
    delay: std::time::Duration,
) {
    let dismiss_handle = app_handle.clone();
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        if SELECTION_TRANSLATION_GENERATION.load(Ordering::Acquire) != generation {
            return;
        }

        clear_selection_translation_presentation(generation);
        let hide_event_handle = dismiss_handle.clone();
        if let Err(schedule_error) = dismiss_handle.run_on_main_thread(move || {
            let Some(window) =
                hide_event_handle.get_webview_window("selection_translation_overlay")
            else {
                error!("Selected-text translation overlay window is unavailable during dismissal");
                return;
            };
            if let Err(emit_error) = window.emit("hide-selection-translation", ()) {
                error!("Could not hide selected-text translation overlay content: {emit_error}");
            }
        }) {
            error!("Could not schedule selected-text translation dismissal: {schedule_error}");
            return;
        }

        std::thread::sleep(std::time::Duration::from_millis(180));
        let hide_window_handle = dismiss_handle.clone();
        if let Err(schedule_error) = dismiss_handle.run_on_main_thread(move || {
            let Some(window) =
                hide_window_handle.get_webview_window("selection_translation_overlay")
            else {
                error!("Selected-text translation overlay window is unavailable while hiding");
                return;
            };
            if let Err(hide_error) = window.hide() {
                error!("Could not hide selected-text translation overlay: {hide_error}");
            }
        }) {
            error!("Could not schedule selected-text translation hide: {schedule_error}");
        }
    });
}

fn show_selected_text_translation(
    app_handle: &AppHandle,
    state: &str,
    target_language: Option<String>,
    text: Option<String>,
    reposition: bool,
    dismiss_after: Option<std::time::Duration>,
) {
    let presentation = SelectionTranslationPresentation {
        state: state.to_string(),
        target_language,
        text,
    };
    // Every presentation invalidates a previous auto-dismiss timer. Without
    // this, a completed earlier translation could hide a newer loading/result
    // card while it is still active.
    let generation = SELECTION_TRANSLATION_GENERATION.fetch_add(1, Ordering::AcqRel) + 1;
    let pending = PendingSelectionTranslationPresentation {
        presentation,
        reposition,
        generation,
    };
    remember_selection_translation_presentation(pending.clone());
    present_selection_translation(app_handle, pending);

    if let Some(delay) = dismiss_after {
        dismiss_selection_translation_after(app_handle, generation, delay);
    }
}

/// Shows immediate feedback while Handy safely copies and translates the selection.
pub fn show_selected_text_translation_loading(app_handle: &AppHandle, target_language: &str) {
    show_selected_text_translation(
        app_handle,
        "loading",
        Some(target_language.to_string()),
        None,
        true,
        None,
    );
}

/// Shows the translated text near the original selection, without modifying it.
pub fn show_selected_text_translation_result(
    app_handle: &AppHandle,
    target_language: &str,
    text: String,
) {
    show_selected_text_translation(
        app_handle,
        "success",
        Some(target_language.to_string()),
        Some(text),
        false,
        Some(std::time::Duration::from_secs(7)),
    );
}

/// Shows a short, recoverable error in the same non-activating overlay.
pub fn show_selected_text_translation_error(app_handle: &AppHandle, message: String) {
    show_selected_text_translation(
        app_handle,
        "error",
        None,
        Some(message),
        true,
        Some(std::time::Duration::from_secs(5)),
    );
}

// Cached "overlay is enabled" flag, kept in sync with overlay_style. Avoids
// reading the Tauri store on every audio callback (~24 Hz during recording).
// Defaults to false so the audio path doesn't emit until lib.rs::setup
// populates the cache from initial settings.
static OVERLAY_ENABLED: AtomicBool = AtomicBool::new(false);

/// Update the cached overlay-enabled flag. Called from `lib.rs` at
/// startup after settings load, and from `change_overlay_style_setting`
/// whenever the user changes whether the overlay is shown.
pub fn update_overlay_enabled_cache(enabled: bool) {
    OVERLAY_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn emit_levels(app_handle: &AppHandle, levels: &[f32]) {
    // Skip emission when the overlay is disabled. The recording_overlay
    // window is created at boot regardless of overlay_style, so without this
    // guard a hidden overlay's WebKit subprocess still
    // processes every event. Each event drives some kind of WebKit
    // C++ allocation that accumulates without bound (mechanism not
    // directly characterized; see issue #1279 for the investigation).
    // For users with `overlay_style: none` (the Linux default) this skip
    // eliminates the upstream driver of that accumulation.
    if !OVERLAY_ENABLED.load(Ordering::Relaxed) {
        return;
    }

    // Throttle to ~30 FPS. Even with the overlay enabled, the raw audio
    // callback fires far faster than the UI needs; capping emission rate
    // cuts the per-frame `eval_script`/IPC volume that drives the wry
    // memory growth in issue #1279 (upstream tauri-apps/wry#1489).
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let last = LAST_MIC_LEVEL_EMIT.load(Ordering::Relaxed);
    if now.saturating_sub(last) < EMIT_THROTTLE_MS {
        return;
    }
    LAST_MIC_LEVEL_EMIT.store(now, Ordering::Relaxed);

    // Target only the overlay window. In Tauri 2 both `AppHandle::emit`
    // and `WebviewWindow::emit` broadcast to all webviews; Tauri's
    // listener filter then skips webviews with no registered listener
    // for the event, so the settings webview never received `mic-level`.
    // But the previous dual-call pattern still produced two `eval_script`
    // calls to the overlay per audio callback (one from each .emit()).
    // `emit_to` with the overlay's window label produces a single
    // eval_script call per callback, cutting the per-callback WebKit
    // dispatch work in half.
    let _ = app_handle.emit_to("recording_overlay", "mic-level", levels);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notch_presentation_clamps_untrusted_appkit_geometry() {
        let too_small = normalized_notch_presentation(4.0, Some(80.0));
        assert_eq!(too_small.safe_area_top, 20.0);
        assert_eq!(too_small.housing_width, 110.0);

        let too_large = normalized_notch_presentation(90.0, Some(500.0));
        assert_eq!(too_large.safe_area_top, 42.0);
        assert_eq!(too_large.housing_width, 200.0);
    }

    #[test]
    fn notch_presentation_uses_hardware_sized_default() {
        let geometry = normalized_notch_presentation(DEFAULT_NOTCH_INSET, None);
        assert_eq!(geometry.safe_area_top, DEFAULT_NOTCH_INSET);
        assert_eq!(geometry.housing_width, DEFAULT_NOTCH_HOUSING_WIDTH);
    }

    #[test]
    fn monitor_hit_test_uses_half_open_physical_bounds() {
        let position = PhysicalPosition::new(-2560, -200);
        let size = PhysicalSize::new(2560, 1440);

        assert!(is_mouse_within_monitor((-2560, -200), &position, &size));
        assert!(is_mouse_within_monitor((-1, 1239), &position, &size));
        assert!(!is_mouse_within_monitor((0, 0), &position, &size));
        assert!(!is_mouse_within_monitor((-1, 1240), &position, &size));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_cursor_hit_test_does_not_scale_physical_monitor_bounds() {
        let position = PhysicalPosition::new(1920, 0);
        let size = PhysicalSize::new(3840, 2160);
        let cursor = (5000, 1000);

        assert!(is_mouse_within_monitor(cursor, &position, &size));

        // This is the old mixed-coordinate comparison. It excludes a cursor
        // that is visibly inside a secondary display running at 150%.
        let scale = 1.5;
        let logical_position = PhysicalPosition::new(
            (position.x as f64 / scale) as i32,
            (position.y as f64 / scale) as i32,
        );
        let logical_size = PhysicalSize::new(
            (size.width as f64 / scale) as u32,
            (size.height as f64 / scale) as u32,
        );
        assert!(!is_mouse_within_monitor(
            cursor,
            &logical_position,
            &logical_size
        ));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_overlay_bounds_use_destination_monitor_scale() {
        let monitor_position = PhysicalPosition::new(1920, 0);
        let monitor_size = PhysicalSize::new(3840, 2160);

        assert_eq!(
            windows_overlay_bounds(
                monitor_position,
                monitor_size,
                1.5,
                OVERLAY_WIDTH,
                OVERLAY_HEIGHT,
                OverlayPosition::Bottom,
            ),
            (3648, 2031, 384, 69)
        );
        assert_eq!(
            windows_overlay_bounds(
                monitor_position,
                monitor_size,
                1.5,
                OVERLAY_WIDTH,
                OVERLAY_HEIGHT,
                OverlayPosition::Top,
            ),
            (3648, 6, 384, 69)
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_overlay_bounds_support_negative_monitor_origins() {
        assert_eq!(
            windows_overlay_bounds(
                PhysicalPosition::new(-2560, -200),
                PhysicalSize::new(2560, 1440),
                1.25,
                OVERLAY_STREAM_WIDTH,
                OVERLAY_STREAM_HEIGHT,
                OverlayPosition::Bottom,
            ),
            (-1530, 1040, 500, 150)
        );
    }
}
