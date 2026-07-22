import { describe, expect, it } from "vitest";
import {
  CHALLENGES,
  isChallengeMet,
  loadCompletedChallenges,
  newlyCompletedChallenges,
  saveCompletedChallenges,
} from "./challenges";
import type { PlantMetrics } from "./scoring";

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

describe("CHALLENGES", () => {
  it("has no duplicate ids", () => {
    const ids = CHALLENGES.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe("isChallengeMet", () => {
  it("is met once the metric reaches the threshold, not just past it", () => {
    const challenge = CHALLENGES.find((c) => c.id === "leaves_at_once_1")!;
    expect(isChallengeMet(challenge, metrics({ maxLeavesAtOnce: 5 }))).toBe(true);
    expect(isChallengeMet(challenge, metrics({ maxLeavesAtOnce: 4 }))).toBe(false);
  });
});

describe("newlyCompletedChallenges", () => {
  it("only returns challenges not already in the completed set", () => {
    const current = metrics({ aliveDays: 40 });
    const already = new Set(["alive_1", "alive_2"]);
    const fresh = newlyCompletedChallenges(current, already);
    expect(fresh).toEqual(["alive_3"]);
  });

  it("returns nothing once every satisfied challenge is already completed", () => {
    const current = metrics({ aliveDays: 40 });
    const already = new Set(["alive_1", "alive_2", "alive_3"]);
    expect(newlyCompletedChallenges(current, already)).toEqual([]);
  });

  it("returns an empty list for a metrics reading that meets nothing", () => {
    expect(newlyCompletedChallenges(metrics(), new Set())).toEqual([]);
  });
});

describe("loadCompletedChallenges/saveCompletedChallenges", () => {
  it("round-trips a saved set", () => {
    const storage = fakeStorage();
    saveCompletedChallenges(new Set(["height_1", "alive_1"]), storage);
    expect(loadCompletedChallenges(storage)).toEqual(new Set(["height_1", "alive_1"]));
  });

  it("returns an empty set when storage is undefined or empty", () => {
    expect(loadCompletedChallenges(undefined)).toEqual(new Set());
    expect(loadCompletedChallenges(fakeStorage())).toEqual(new Set());
  });

  it("falls back to an empty set for corrupted JSON instead of throwing", () => {
    const storage = fakeStorage();
    storage.setItem("plant-game-challenges-completed", "not json");
    expect(loadCompletedChallenges(storage)).toEqual(new Set());
  });
});
