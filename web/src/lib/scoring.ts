// Cross-session high scores for the four lifetime plant metrics the engine
// tracks per plant (see Plant::max_height_reached/max_leaves_at_once/
// leaves_produced_total/alive_duration in engine/src/sim/plant.rs) — kept
// separate from EngineCanvas.tsx so the merge math is unit-testable without
// a wasm module, canvas, or real localStorage, same reasoning as
// formatStats.ts/healthTiers.ts.

export interface PlantMetrics {
  maxHeightReached: number;
  maxLeavesAtOnce: number;
  leavesProducedTotal: number;
  aliveDays: number;
}

export type MetricKey = keyof PlantMetrics;

const HIGH_SCORE_STORAGE_KEY = "plant-game-high-scores";

const ZERO_METRICS: PlantMetrics = {
  maxHeightReached: 0,
  maxLeavesAtOnce: 0,
  leavesProducedTotal: 0,
  aliveDays: 0,
};

// `localStorage` doesn't exist during a build-time prerender — every
// storage-touching function here goes through this rather than referencing
// `window`/`localStorage` directly, so importers don't each need their own
// SSR guard.
export function safeLocalStorage(): Storage | undefined {
  return typeof window === "undefined" ? undefined : window.localStorage;
}

// The best-ever value for each metric across every plant this browser has
// ever grown, across sessions/restarts — distinct from the engine's own
// per-plant high-water marks, which reset the moment that one plant is
// replaced. Falls back to all-zero scores if storage is unavailable or
// holds something unparseable, rather than throwing.
export function loadHighScores(storage: Storage | undefined = safeLocalStorage()): PlantMetrics {
  if (!storage) return { ...ZERO_METRICS };
  try {
    const raw = storage.getItem(HIGH_SCORE_STORAGE_KEY);
    if (!raw) return { ...ZERO_METRICS };
    const parsed = JSON.parse(raw) as Partial<Record<MetricKey, unknown>>;
    return {
      maxHeightReached: Number(parsed.maxHeightReached) || 0,
      maxLeavesAtOnce: Number(parsed.maxLeavesAtOnce) || 0,
      leavesProducedTotal: Number(parsed.leavesProducedTotal) || 0,
      aliveDays: Number(parsed.aliveDays) || 0,
    };
  } catch {
    return { ...ZERO_METRICS };
  }
}

export function saveHighScores(scores: PlantMetrics, storage: Storage | undefined = safeLocalStorage()): void {
  storage?.setItem(HIGH_SCORE_STORAGE_KEY, JSON.stringify(scores));
}

// Merges a fresh reading against the stored record, one metric at a time —
// a plant that's this browser's tallest ever but not its longest-lived
// still gets credit for the height record without erasing some other
// plant's still-standing leaf-count record. Pure (no storage I/O) so it's
// trivial to test in isolation.
export function mergeHighScores(
  previous: PlantMetrics,
  current: PlantMetrics,
): { scores: PlantMetrics; improved: MetricKey[] } {
  const improved: MetricKey[] = [];
  const scores = { ...previous };
  (Object.keys(ZERO_METRICS) as MetricKey[]).forEach((key) => {
    if (current[key] > previous[key]) {
      scores[key] = current[key];
      improved.push(key);
    }
  });
  return { scores, improved };
}
