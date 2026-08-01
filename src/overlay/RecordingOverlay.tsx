import { listen } from "@tauri-apps/api/event";
import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import "./RecordingOverlay.css";
import { commands, events } from "@/bindings";
import type {
  StreamPhase,
  StreamPhaseEvent,
  StreamTextEvent,
  StreamWorkKind,
} from "@/bindings";
import i18n, { syncLanguageFromSettings } from "@/i18n";
import { getLanguageDirection } from "@/lib/utils/rtl";

type OverlayState =
  | "recording"
  | "streaming"
  | "transcribing"
  | "processing"
  | "translating"
  | "verifying";

/** Effective placement from the backend — notch styling only for NotchAttached. */
type OverlayPlacement = "notch_attached" | "top_fallback" | "top" | "bottom";

type NotchPresentation = {
  safeAreaTop: number;
  housingWidth: number;
};

type OverlayPresentation = {
  state: OverlayState;
  placement?: OverlayPlacement;
  notch?: NotchPresentation;
};

// Number of reactive bars in the waveform (the simple, smoothed style shared by
// every overlay form). Mic levels arrive as 16 FFT buckets; we take the first N.
const WAVE_BARS = 9;

function workLabelFromPhase(
  t: (key: string, opts?: Record<string, unknown>) => string,
  kind: StreamWorkKind,
  detail: StreamPhaseEvent | null,
): string {
  switch (kind) {
    case "translating": {
      const language =
        detail?.target_language_name || detail?.target_language || undefined;
      return language
        ? t("overlay.translatingTo", { language })
        : t("overlay.translating");
    }
    case "verifying": {
      if (detail?.step != null && detail?.step_total != null) {
        return t("overlay.verifyingStep", {
          step: detail.step,
          total: detail.step_total,
        });
      }
      return t("overlay.verifying");
    }
    case "post_processing":
    case "polishing":
      return t("overlay.postProcessing");
    case "transcribing":
    default:
      return t("overlay.transcribing");
  }
}

function workLabelFromState(
  t: (key: string, opts?: Record<string, unknown>) => string,
  state: OverlayState,
): string {
  switch (state) {
    case "processing":
      return t("overlay.postProcessing");
    case "translating":
      return t("overlay.translating");
    case "verifying":
      return t("overlay.verifying");
    case "transcribing":
    default:
      return t("overlay.transcribing");
  }
}

/** Map settings preference + optional backend placement into CSS stage class. */
function stageClass(placement: OverlayPlacement): string {
  switch (placement) {
    case "notch_attached":
      return "notch";
    case "top_fallback":
    case "top":
      return "top";
    case "bottom":
    default:
      return "bottom";
  }
}

const RecordingOverlay: React.FC = () => {
  const { t } = useTranslation();
  const [isVisible, setIsVisible] = useState(false);
  const [state, setState] = useState<OverlayState>("recording");
  const [levels, setLevels] = useState<number[]>(Array(WAVE_BARS).fill(0));
  const [streamText, setStreamText] = useState<StreamTextEvent>({
    committed: "",
    tentative: "",
  });
  const [phase, setPhase] = useState<StreamPhase>("listening");
  const [workKind, setWorkKind] = useState<StreamWorkKind>("transcribing");
  const [phaseDetail, setPhaseDetail] = useState<StreamPhaseEvent | null>(null);
  const [elapsed, setElapsed] = useState(0);
  // Bumped on each new streaming session so the Live card remounts fresh (replays
  // the pop-in, and never animates in from the previous panel's open size).
  const [session, setSession] = useState(0);
  // Effective placement from the last show-overlay / placement event.
  const [placement, setPlacement] = useState<OverlayPlacement>("bottom");
  const [notch, setNotch] = useState<NotchPresentation | null>(null);
  // True once live text overflows the cap. A top overlay fades its top edge only
  // while overflowing, so the resting first line stays crisp flush under the pill.
  const [overflowing, setOverflowing] = useState(false);

  const smoothedLevelsRef = useRef<number[]>(Array(16).fill(0));
  // Live-text scroll-back: the text region "sticks" to the newest line while the
  // user is at the bottom; if they scroll up to read history, auto-follow pauses
  // until they scroll back down.
  const capRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const direction = getLanguageDirection(i18n.language);

  useEffect(() => {
    const setupEventListeners = async () => {
      const unlistenShow = await listen("show-overlay", async (event) => {
        // Language synchronization does not need to delay placement. Applying
        // the combined backend payload immediately prevents a one-frame top-pill
        // flash before the attached notch shape is painted.
        void syncLanguageFromSettings();
        // Prefer backend effective placement when present (object payload).
        // Older path sends a plain state string.
        const payload = event.payload as OverlayState | OverlayPresentation;

        let overlayState: OverlayState;
        if (typeof payload === "string") {
          overlayState = payload;
          // Fall back to settings preference; notch is only applied when the
          // backend later confirms notch_attached via overlay-placement.
          try {
            const settings = await commands.getAppSettings();
            if (settings.status === "ok") {
              const configured = settings.data.overlay_position;
              if (configured === "notch") {
                // Do not force notch styling until geometry confirms attachment.
                // Default to top until placement event arrives.
                setPlacement((prev) =>
                  prev === "notch_attached" ? prev : "top_fallback",
                );
              } else if (configured === "top") {
                setPlacement("top");
              } else {
                setPlacement("bottom");
              }
            }
          } catch {
            // Keep the previous/default placement if settings can't be read.
          }
        } else {
          overlayState = payload.state;
          if (payload.placement) {
            setPlacement(payload.placement);
          }
          setNotch(payload.notch ?? null);
        }

        setState(overlayState);
        if (overlayState === "recording" || overlayState === "streaming") {
          setStreamText({ committed: "", tentative: "" });
        }
        if (overlayState === "streaming") {
          setPhase("listening");
          setWorkKind("transcribing");
          setPhaseDetail(null);
          setElapsed(0);
          setSession((s) => s + 1); // remount the card fresh for this session
        }
        setIsVisible(true);
      });

      const unlistenPlacement = await listen<OverlayPlacement>(
        "overlay-placement",
        (event) => {
          setPlacement(event.payload);
          if (event.payload !== "notch_attached") {
            setNotch(null);
          }
        },
      );

      const unlistenHide = await listen("hide-overlay", () => {
        setIsVisible(false);
      });

      const unlistenLevel = await listen<number[]>("mic-level", (event) => {
        const newLevels = event.payload as number[];
        // Exponential smoothing across the 16 buckets, then take the first N
        // bars for the shared waveform.
        const smoothed = smoothedLevelsRef.current.map((prev, i) => {
          const target = newLevels[i] || 0;
          return prev * 0.7 + target * 0.3;
        });
        smoothedLevelsRef.current = smoothed;
        setLevels(smoothed.slice(0, WAVE_BARS));
      });

      const unlistenStream = await events.streamTextEvent.listen((event) => {
        setStreamText(event.payload);
      });

      const unlistenPhase = await events.streamPhaseEvent.listen((event) => {
        const payload: StreamPhaseEvent = event.payload;
        setPhase(payload.phase);
        setPhaseDetail(payload);
        if (payload.kind) setWorkKind(payload.kind);
      });

      return () => {
        unlistenShow();
        unlistenPlacement();
        unlistenHide();
        unlistenLevel();
        unlistenStream();
        unlistenPhase();
      };
    };

    setupEventListeners();
  }, []);

  // Elapsed timer while the Live overlay is visible.
  useEffect(() => {
    if (state !== "streaming" || !isVisible) return;
    const id = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(id);
  }, [state, isVisible]);

  // Stick to the bottom as text streams in — but only while pinned, so a user who
  // has scrolled up to read history isn't yanked back down by the next chunk.
  useLayoutEffect(() => {
    const el = capRef.current;
    if (!el) return;
    // Fade the top edge only once text actually overflows the cap.
    setOverflowing(el.scrollHeight > el.clientHeight + 1);
    if (pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [streamText]);

  // Each fresh streaming session starts pinned to the bottom, fade cleared.
  useEffect(() => {
    pinnedRef.current = true;
    setOverflowing(false);
  }, [session]);

  // Re-pin when the user is within ~a line of the bottom; unpin otherwise.
  const handleStreamScroll = () => {
    const el = capRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight <= 16;
  };

  const fmtTime = (s: number) =>
    `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

  const stage = stageClass(placement);
  const isNotch = stage === "notch";
  const stageStyle =
    isNotch && notch
      ? ({
          "--notch-safe-top": `${notch.safeAreaTop}px`,
          "--notch-housing-w": `${notch.housingWidth}px`,
        } as React.CSSProperties)
      : undefined;
  // Housing-width black fill flush with the top of the display so the island
  // attaches to the camera cutout instead of floating below it.
  const notchBridge = isNotch ? (
    <span className="snotch-bridge" aria-hidden="true" />
  ) : null;

  // ---- Shared building blocks (one visual language for every overlay form) ----
  // Dynamic Island uses fewer, taller right-side rails (reference); other
  // placements keep the full waveform in the center.
  const waveBarCount = isNotch ? 5 : WAVE_BARS;
  const waveMax = isNotch ? 16 : 18;
  const waveform = (
    <div className="swave" aria-hidden="true">
      {levels.slice(0, waveBarCount).map((v, i) => (
        <i
          key={i}
          style={{
            height: `${Math.max(3, Math.min(waveMax, 3 + Math.pow(v, 0.7) * (waveMax - 3)))}px`,
          }}
        />
      ))}
    </div>
  );

  const cancelBtn = (
    <button
      className="sx"
      aria-label={t("overlay.cancel")}
      onClick={() => commands.cancelOperation()}
    >
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path
          d="M4 4 L12 12 M12 4 L4 12"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </svg>
    </button>
  );

  // Standard pill: dot (left) | waveform (center) | timer + cancel (right).
  // Dynamic Island: activity chip (left) | empty stage (center) | waveform + cancel
  // (right) — matching the system island's three-rail silhouette.
  const listeningRow = (showTimer: boolean, showCancel: boolean) =>
    isNotch ? (
      <div className="sbase">
        <div className="sbase-l">
          <span className="sactivity">
            <span className="sdot" />
          </span>
        </div>
        <div className="sstage" />
        <div className="sbase-r">
          {showTimer && <span className="stimer">{fmtTime(elapsed)}</span>}
          {waveform}
          {showCancel && cancelBtn}
        </div>
      </div>
    ) : (
      <div className="sbase">
        <div className="sbase-l">
          <span className="sdot" />
        </div>
        {waveform}
        <div className="sbase-r">
          {showTimer && <span className="stimer">{fmtTime(elapsed)}</span>}
          {showCancel && cancelBtn}
        </div>
      </div>
    );

  // Working: spinner/activity (left) | label (center) | cancel (right).
  const workingRow = (label: string, showCancel: boolean) =>
    isNotch ? (
      <div className="sbase">
        <div className="sbase-l">
          <span className="sactivity">
            <span className="sspinner" />
          </span>
        </div>
        <div className="sstage">
          <span className="swork-label">{label}</span>
        </div>
        <div className="sbase-r">{showCancel && cancelBtn}</div>
      </div>
    ) : (
      <div className="sbase">
        <div className="sbase-l">
          <span className="sspinner" />
        </div>
        <span className="swork-label">{label}</span>
        <div className="sbase-r">{showCancel && cancelBtn}</div>
      </div>
    );

  // ---- Live overlay: a pill that sculpts open into a panel ----
  if (state === "streaming") {
    const hasText =
      streamText.committed.length > 0 || streamText.tentative.length > 0;
    const working = phase === "working";
    // Keep the panel open whenever there's text — even while finalizing — so the
    // transcript stays put under a working spinner instead of collapsing and
    // squishing the text mid-stream. Only fall back to the small working pill
    // when there was no text to preserve.
    const open = hasText;
    const collapsed = working && !hasText;

    // Control rail + live text. Top/notch place the rail above the transcript
    // (column-reverse + this order, or column with rail first for notch). Bottom
    // keeps the original text-then-rail DOM so the pill sits under the text.
    const controlRail = working
      ? workingRow(workLabelFromPhase(t, workKind, phaseDetail), true)
      : listeningRow(open, true);
    const textRegion = (
      <div className="stext">
        <div className="stext-clip">
          <div
            className={`stext-cap ${overflowing ? "overflowing" : ""}`}
            ref={capRef}
            onScroll={handleStreamScroll}
          >
            <p>
              <span className="committed">
                {streamText.committed ? streamText.committed + " " : ""}
              </span>
              <span className="tentative">{streamText.tentative}</span>
              {/* Drop the blinking caret once finalizing — it's no longer
                  capturing, and a static spinner conveys the work. */}
              {!working && <span className="scaret" />}
            </p>
          </div>
        </div>
      </div>
    );

    return (
      <div dir={direction} className={`ov-stage ${stage}`} style={stageStyle}>
        {notchBridge}
        <div
          key={session}
          className={`scard ${open ? "open" : ""} ${collapsed ? "working" : ""} ${
            isVisible ? "" : "leaving"
          }`}
        >
          {isNotch ? (
            <>
              {controlRail}
              {textRegion}
            </>
          ) : (
            <>
              {textRegion}
              {controlRail}
            </>
          )}
        </div>
      </div>
    );
  }

  // ---- Minimal overlay: exactly one row at a time — waveform (recording), or a
  // spinner + label (transcribing / processing / translating / verifying).
  const working =
    state === "transcribing" ||
    state === "processing" ||
    state === "translating" ||
    state === "verifying";
  const workLabel = workLabelFromState(t, state);

  return (
    <div
      dir={direction}
      className={`ov-stage ${stage} ov-fade ${isVisible ? "show" : ""}`}
      style={stageStyle}
    >
      {notchBridge}
      <div
        className={`scard compact ${working && isVisible ? "cworking" : ""}`}
      >
        {working ? workingRow(workLabel, true) : listeningRow(false, true)}
      </div>
    </div>
  );
};

export default RecordingOverlay;
