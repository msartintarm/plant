import { describe, expect, it } from "vitest";
import { loadHighScores, mergeHighScores, saveHighScores, type PlantMetrics } from "./scoring";

// A minimal in-memory Storage stand-in — avoids depending on jsdom's own
// localStorage just to exercise the load/save round-trip.
function fakeStorage(): Storage {
  const data = new Map<string, string>();
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, value),
    removeItem: (key: string) => void data.delete(key),
    clear: () => data.clear(),
    key: () => null,
    get length() {
      return data.size;
    },
  };
}

const metrics = (overrides: Partial<PlantMetrics> = {}): PlantMetrics => ({
  maxHeightReached: 0,
  maxLeavesAtOnce: 0,
  leavesProducedTotal: 0,
  aliveDays: 0,
  ...overrides,
});

describe("loadHighScores", () => {
  it("returns all zeros when storage is undefined (e.g. during SSR)", () => {
    expect(loadHighScores(undefined)).toEqual(metrics());
  });

  it("returns all zeros when nothing has been saved yet", () => {
    expect(loadHighScores(fakeStorage())).toEqual(metrics());
  });

  it("round-trips whatever saveHighScores wrote", () => {
    const storage = fakeStorage();
    const saved = metrics({ maxHeightReached: 4.2, maxLeavesAtOnce: 12, leavesProducedTotal: 30, aliveDays: 5.5 });
    saveHighScores(saved, storage);
    expect(loadHighScores(storage)).toEqual(saved);
  });

  it("falls back to zeros for corrupted JSON instead of throwing", () => {
    const storage = fakeStorage();
    storage.setItem("plant-game-high-scores", "{not json");
    expect(loadHighScores(storage)).toEqual(metrics());
  });
});

describe("mergeHighScores", () => {
  it("keeps the previous record when nothing in the fresh reading beats it", () => {
    const previous = metrics({ maxHeightReached: 5 });
    const { scores, improved } = mergeHighScores(previous, metrics({ maxHeightReached: 3 }));
    expect(scores).toEqual(previous);
    expect(improved).toEqual([]);
  });

  it("updates only the metrics that actually improved, leaving siblings alone", () => {
    const previous = metrics({ maxHeightReached: 5, aliveDays: 10 });
    const { scores, improved } = mergeHighScores(previous, metrics({ maxHeightReached: 8, aliveDays: 2 }));
    expect(scores.maxHeightReached).toBe(8);
    // The lower fresh reading shouldn't erase a still-standing record.
    expect(scores.aliveDays).toBe(10);
    expect(improved).toEqual(["maxHeightReached"]);
  });

  it("can improve more than one metric in the same reading", () => {
    const previous = metrics();
    const current = metrics({ maxLeavesAtOnce: 7, leavesProducedTotal: 20 });
    const { improved } = mergeHighScores(previous, current);
    expect(improved.sort()).toEqual(["leavesProducedTotal", "maxLeavesAtOnce"]);
  });
});
