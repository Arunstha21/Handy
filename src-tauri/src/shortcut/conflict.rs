//! Semantic shortcut conflict detection.
//!
//! Parse validation alone is not enough: a modifier-only binding such as
//! `rightcommand` fires as soon as that key is pressed, so a longer combo like
//! `rightcommand+space` can never start its action while the pipeline is busy.
//! This module detects exact duplicates and prefix conflicts among transcription
//! shortcuts so the UI can refuse the save and offer the recommended default.

use serde::Serialize;
use specta::Type;
use std::collections::BTreeSet;

/// Transcription action IDs that share the recording pipeline.
pub const TRANSCRIBE_BINDING_IDS: &[&str] = &[
    "transcribe",
    "transcribe_with_post_process",
    "transcribe_with_translation",
];

/// Global bindings that must not share an exact or modifier-prefix shortcut.
/// Selected-text translation does not touch the recording coordinator, but it
/// is still a global action and therefore must not shadow a speech action.
pub const CONFLICT_CHECKED_BINDING_IDS: &[&str] = &[
    "transcribe",
    "transcribe_with_post_process",
    "transcribe_with_translation",
    "translate_selected_text",
];

const MODIFIERS: &[&str] = &[
    "ctrl",
    "control",
    "leftctrl",
    "rightctrl",
    "leftcontrol",
    "rightcontrol",
    "shift",
    "leftshift",
    "rightshift",
    "alt",
    "option",
    "leftalt",
    "rightalt",
    "leftoption",
    "rightoption",
    "meta",
    "command",
    "cmd",
    "super",
    "win",
    "windows",
    "leftmeta",
    "rightmeta",
    "leftcommand",
    "rightcommand",
    "leftcmd",
    "rightcmd",
    "leftsuper",
    "rightsuper",
    "leftwin",
    "rightwin",
    "fn",
    "function",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ConflictKind {
    /// Both bindings resolve to the same normalized chord.
    Exact,
    /// One binding's keys are a proper prefix of the other (modifier-only vs longer combo).
    Prefix,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Type)]
pub struct ShortcutConflict {
    pub action_id: String,
    pub other_action_id: String,
    pub binding: String,
    pub other_binding: String,
    pub kind: ConflictKind,
    /// Recommended non-conflicting binding for `action_id`, when one is known.
    pub recommended_binding: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NormalizedShortcut {
    /// Canonical modifier tokens, sorted for set comparison.
    modifiers: BTreeSet<String>,
    /// Main key, if any (modifier-only shortcuts have `None`).
    key: Option<String>,
}

impl NormalizedShortcut {
    fn parse(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err("Shortcut cannot be empty".to_string());
        }

        let parts: Vec<String> = trimmed
            .split('+')
            .map(|p| p.trim().to_lowercase())
            .filter(|p| !p.is_empty())
            .collect();

        if parts.is_empty() {
            return Err("Shortcut cannot be empty".to_string());
        }

        let mut modifiers = BTreeSet::new();
        let mut key: Option<String> = None;

        for part in parts {
            let canonical = canonicalize_token(&part);
            if is_modifier(&canonical) {
                modifiers.insert(canonical);
            } else if key.is_some() {
                return Err(format!(
                    "Shortcut '{}' has more than one non-modifier key",
                    raw
                ));
            } else {
                key = Some(canonical);
            }
        }

        if modifiers.is_empty() && key.is_none() {
            return Err("Shortcut cannot be empty".to_string());
        }

        Ok(Self { modifiers, key })
    }

    fn is_modifier_only(&self) -> bool {
        self.key.is_none()
    }

    /// True when `self` fires as soon as its keys are held, before `other` can complete.
    ///
    /// A modifier-only shortcut is a prefix of any longer shortcut that includes
    /// all of those modifiers (with or without an additional main key). Exact
    /// matches are handled separately.
    fn is_prefix_of(&self, other: &Self) -> bool {
        if self == other {
            return false;
        }
        if !self.is_modifier_only() {
            return false;
        }
        self.modifiers.is_subset(&other.modifiers)
    }
}

fn is_modifier(token: &str) -> bool {
    MODIFIERS.contains(&token)
}

/// Collapse platform aliases so `cmd` and `command` compare equal, while still
/// preserving left/right distinctions used by HandyKeys.
fn canonicalize_token(token: &str) -> String {
    match token {
        "control" => "ctrl".to_string(),
        "leftcontrol" => "leftctrl".to_string(),
        "rightcontrol" => "rightctrl".to_string(),
        "option" => "alt".to_string(),
        "leftoption" => "leftalt".to_string(),
        "rightoption" => "rightalt".to_string(),
        "command" | "cmd" | "meta" | "super" | "win" | "windows" => "command".to_string(),
        "leftcommand" | "leftcmd" | "leftmeta" | "leftsuper" | "leftwin" => {
            "leftcommand".to_string()
        }
        "rightcommand" | "rightcmd" | "rightmeta" | "rightsuper" | "rightwin" => {
            "rightcommand".to_string()
        }
        "function" => "fn".to_string(),
        other => other.to_string(),
    }
}

/// Human-readable action names for conflict messages.
pub fn action_display_name(id: &str) -> String {
    match id {
        "transcribe" => "Transcribe".to_string(),
        "transcribe_with_post_process" => "Post-process".to_string(),
        "transcribe_with_translation" => "Translate".to_string(),
        "translate_selected_text" => "Translate selected text".to_string(),
        "cancel" => "Cancel".to_string(),
        other => other.to_string(),
    }
}

/// Default binding for an action on this platform (from settings defaults).
pub fn recommended_binding_for(action_id: &str) -> Option<String> {
    crate::settings::get_default_settings()
        .bindings
        .get(action_id)
        .map(|b| b.default_binding.clone())
}

/// Detect a conflict between `candidate` (for `action_id`) and an existing binding.
pub fn detect_pair_conflict(
    action_id: &str,
    candidate_raw: &str,
    other_action_id: &str,
    other_raw: &str,
) -> Option<ShortcutConflict> {
    if action_id == other_action_id {
        return None;
    }

    let candidate = NormalizedShortcut::parse(candidate_raw).ok()?;
    let other = NormalizedShortcut::parse(other_raw).ok()?;

    let kind = if candidate == other {
        ConflictKind::Exact
    } else if candidate.is_prefix_of(&other) || other.is_prefix_of(&candidate) {
        ConflictKind::Prefix
    } else {
        return None;
    };

    Some(ShortcutConflict {
        action_id: action_id.to_string(),
        other_action_id: other_action_id.to_string(),
        binding: candidate_raw.to_string(),
        other_binding: other_raw.to_string(),
        kind,
        recommended_binding: recommended_binding_for(action_id),
    })
}

/// Whether this binding participates in the transcription pipeline and should
/// be checked for semantic conflicts with other transcribe bindings.
pub fn is_transcribe_action(id: &str) -> bool {
    TRANSCRIBE_BINDING_IDS.contains(&id)
}

/// Find the first conflict between `action_id`/`candidate` and the other active
/// transcription bindings in `bindings` (id → current binding string).
///
/// `active_ids` limits which other bindings are considered (e.g. skip post-process
/// when that feature is disabled).
pub fn find_conflict_among<'a, I>(
    action_id: &str,
    candidate: &str,
    others: I,
) -> Option<ShortcutConflict>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    if !is_transcribe_action(action_id) {
        return None;
    }

    for (other_id, other_binding) in others {
        if !is_transcribe_action(other_id) || other_id == action_id {
            continue;
        }
        if let Some(conflict) = detect_pair_conflict(action_id, candidate, other_id, other_binding)
        {
            return Some(conflict);
        }
    }
    None
}

/// Format a conflict for command errors and toasts.
pub fn format_conflict_message(conflict: &ShortcutConflict) -> String {
    let a = action_display_name(&conflict.action_id);
    let b = action_display_name(&conflict.other_action_id);
    match conflict.kind {
        ConflictKind::Exact => format!(
            "Shortcut conflict: {a} (`{}`) is identical to {b} (`{}`). Choose a different combination.",
            conflict.binding, conflict.other_binding
        ),
        ConflictKind::Prefix => format!(
            "Shortcut conflict: {a} (`{}`) conflicts with {b} (`{}`). \
A shorter modifier-only shortcut starts recording before the longer combination can fire. \
Use distinct combinations (recommended for Translate: {}).",
            conflict.binding,
            conflict.other_binding,
            conflict
                .recommended_binding
                .as_deref()
                .or(recommended_binding_for("transcribe_with_translation").as_deref())
                .unwrap_or("option+control+space")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_duplicate_is_conflict() {
        let c = detect_pair_conflict(
            "transcribe",
            "option+space",
            "transcribe_with_translation",
            "option+space",
        )
        .expect("exact match should conflict");
        assert_eq!(c.kind, ConflictKind::Exact);
    }

    #[test]
    fn selected_text_translation_conflicts_with_speech_translation() {
        let conflict = detect_pair_conflict(
            "translate_selected_text",
            "option+command+t",
            "transcribe_with_translation",
            "option+command+t",
        )
        .expect("matching global bindings must conflict");

        assert_eq!(conflict.kind, ConflictKind::Exact);
        assert_eq!(conflict.other_action_id, "transcribe_with_translation");
    }

    #[test]
    fn right_command_prefix_of_right_command_space() {
        let c = detect_pair_conflict(
            "transcribe",
            "rightcommand",
            "transcribe_with_translation",
            "rightcommand+space",
        )
        .expect("prefix conflict");
        assert_eq!(c.kind, ConflictKind::Prefix);
    }

    #[test]
    fn right_command_space_prefix_of_right_command() {
        // Either direction must be detected so validation works no matter which
        // binding the user edits last.
        let c = detect_pair_conflict(
            "transcribe_with_translation",
            "rightcommand+space",
            "transcribe",
            "rightcommand",
        )
        .expect("prefix conflict either direction");
        assert_eq!(c.kind, ConflictKind::Prefix);
    }

    #[test]
    fn non_conflicting_defaults_are_ok() {
        assert!(detect_pair_conflict(
            "transcribe",
            "option+space",
            "transcribe_with_translation",
            "option+control+space",
        )
        .is_none());
        assert!(detect_pair_conflict(
            "transcribe",
            "option+space",
            "transcribe_with_post_process",
            "option+shift+space",
        )
        .is_none());
    }

    #[test]
    fn different_keys_with_same_modifiers_ok() {
        assert!(detect_pair_conflict(
            "transcribe",
            "command+a",
            "transcribe_with_translation",
            "command+b",
        )
        .is_none());
    }

    #[test]
    fn side_specific_and_generic_command_aliases_normalize() {
        // "command" is compound (either side); side-specific rightcommand is a
        // distinct binding in HandyKeys, so they are not an exact match. Prefix
        // rules only apply to modifier-only chords, so command+space vs
        // rightcommand is not a prefix either.
        assert!(detect_pair_conflict(
            "transcribe",
            "command",
            "transcribe_with_translation",
            "rightcommand+space",
        )
        .is_none());
    }

    #[test]
    fn cmd_alias_matches_command_exact() {
        let c = detect_pair_conflict(
            "transcribe",
            "cmd+space",
            "transcribe_with_translation",
            "command+space",
        )
        .expect("aliases should normalize to exact conflict");
        assert_eq!(c.kind, ConflictKind::Exact);
    }

    #[test]
    fn find_conflict_among_skips_non_transcribe() {
        let others = [
            ("cancel", "escape"),
            ("transcribe_with_translation", "rightcommand+space"),
        ];
        let c = find_conflict_among("transcribe", "rightcommand", others).expect("should find");
        assert_eq!(c.other_action_id, "transcribe_with_translation");
    }

    #[test]
    fn cancel_bindings_are_not_checked() {
        assert!(find_conflict_among("cancel", "escape", [("transcribe", "escape")],).is_none());
    }
}
