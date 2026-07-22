// Milestone achievements built on top of the high-score record in
// scoring.ts — kept as plain, pure data + functions (no React, no storage
// I/O beyond the two load/save helpers) so the completion logic is
// unit-testable on its own.

import { safeLocalStorage, type MetricKey, type PlantMetrics } from "./scoring";

export interface Challenge {
  id: string;
  metric: MetricKey;
  threshold: number;
  label: string;
  icon: string;
}

// Three tiers per metric. Height thresholds are tuned to Dracaena's own
// scale (engine/src/sim/config.rs's realistic_max_height: 5.5) — Peace Lily
// tops out far shorter and Pothos far taller, so which height tiers a given
// species can realistically reach varies, the same way a real houseplant's
// mature size does.
export const CHALLENGES: Challenge[] = [
  { id: "height_1", metric: "maxHeightReached", threshold: 1, label: "Reach 1.0 tall", icon: "📏" },
  { id: "height_2", metric: "maxHeightReached", threshold: 3, label: "Reach 3.0 tall", icon: "📏" },
  { id: "height_3", metric: "maxHeightReached", threshold: 8, label: "Reach 8.0 tall", icon: "📏" },
  { id: "leaves_at_once_1", metric: "maxLeavesAtOnce", threshold: 5, label: "5 leaves at once", icon: "🍃" },
  { id: "leaves_at_once_2", metric: "maxLeavesAtOnce", threshold: 15, label: "15 leaves at once", icon: "🍃" },
  { id: "leaves_at_once_3", metric: "maxLeavesAtOnce", threshold: 30, label: "30 leaves at once", icon: "🍃" },
  { id: "leaves_total_1", metric: "leavesProducedTotal", threshold: 10, label: "10 leaves grown", icon: "🌿" },
  { id: "leaves_total_2", metric: "leavesProducedTotal", threshold: 50, label: "50 leaves grown", icon: "🌿" },
  { id: "leaves_total_3", metric: "leavesProducedTotal", threshold: 150, label: "150 leaves grown", icon: "🌿" },
  { id: "alive_1", metric: "aliveDays", threshold: 1, label: "Survive 1 day", icon: "⏳" },
  { id: "alive_2", metric: "aliveDays", threshold: 7, label: "Survive 7 days", icon: "⏳" },
  { id: "alive_3", metric: "aliveDays", threshold: 30, label: "Survive 30 days", icon: "⏳" },
];

const COMPLETED_STORAGE_KEY = "plant-game-challenges-completed";

export function isChallengeMet(challenge: Challenge, metrics: PlantMetrics): boolean {
  return metrics[challenge.metric] >= challenge.threshold;
}

// Which challenge ids `metrics` satisfies that aren't already in
// `alreadyCompleted` — only ever adds ids, never removes them, since
// completion is meant to persist even once the plant that earned it is
// gone (replaced, or dead).
export function newlyCompletedChallenges(metrics: PlantMetrics, alreadyCompleted: ReadonlySet<string>): string[] {
  return CHALLENGES.filter((c) => !alreadyCompleted.has(c.id) && isChallengeMet(c, metrics)).map((c) => c.id);
}

export function loadCompletedChallenges(storage: Storage | undefined = safeLocalStorage()): Set<string> {
  if (!storage) return new Set();
  try {
    const raw = storage.getItem(COMPLETED_STORAGE_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? new Set(parsed.filter((id): id is string => typeof id === "string")) : new Set();
  } catch {
    return new Set();
  }
}

export function saveCompletedChallenges(completed: ReadonlySet<string>, storage: Storage | undefined = safeLocalStorage()): void {
  storage?.setItem(COMPLETED_STORAGE_KEY, JSON.stringify(Array.from(completed)));
}
