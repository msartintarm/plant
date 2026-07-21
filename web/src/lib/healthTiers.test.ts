import { describe, expect, it } from "vitest";
import {
  humidityTier,
  nutrientTier,
  pestTier,
  rootHealthTier,
  temperatureTier,
  waterTier,
} from "./healthTiers";

describe("waterTier", () => {
  it("is good in the middle of the range", () => {
    expect(waterTier(0.6)).toBe("good");
  });

  it("is bad once bone dry", () => {
    expect(waterTier(0.1)).toBe("bad");
  });

  it("is caution approaching dry, not yet bad", () => {
    expect(waterTier(0.3)).toBe("caution");
  });

  it("is bad once waterlogged past the engine's own threshold", () => {
    expect(waterTier(0.98)).toBe("bad");
  });

  it("is caution approaching waterlogged, not yet bad", () => {
    expect(waterTier(0.93)).toBe("caution");
  });
});

describe("nutrientTier", () => {
  it("is good in the middle of the range", () => {
    expect(nutrientTier(0.5)).toBe("good");
  });

  it("is bad once starved", () => {
    expect(nutrientTier(0.02)).toBe("bad");
  });

  it("is bad once overfed past the engine's own threshold", () => {
    expect(nutrientTier(1.5)).toBe("bad");
  });

  it("is caution approaching overfed, not yet bad", () => {
    expect(nutrientTier(1.3)).toBe("caution");
  });
});

describe("humidityTier", () => {
  it("is good above the engine's safe_humidity threshold", () => {
    expect(humidityTier(0.7)).toBe("good");
  });

  it("is bad once dry enough for pests to thrive", () => {
    expect(humidityTier(0.2)).toBe("bad");
  });

  it("is caution just below safe_humidity, not yet bad", () => {
    expect(humidityTier(0.45)).toBe("caution");
  });
});

describe("rootHealthTier", () => {
  it("is good near full health", () => {
    expect(rootHealthTier(0.95)).toBe("good");
  });

  it("is bad below the engine's own warning threshold", () => {
    expect(rootHealthTier(0.3)).toBe("bad");
  });

  it("is caution in the middle band", () => {
    expect(rootHealthTier(0.55)).toBe("caution");
  });
});

describe("pestTier", () => {
  it("is good with no infestation", () => {
    expect(pestTier(0.0)).toBe("good");
  });

  it("is bad past the engine's own warning threshold", () => {
    expect(pestTier(0.6)).toBe("bad");
  });

  it("is caution in the middle band", () => {
    expect(pestTier(0.25)).toBe("caution");
  });
});

describe("temperatureTier", () => {
  it("is good at the optimal temperature", () => {
    expect(temperatureTier(24)).toBe("good");
  });

  it("is bad at or below the engine's own cold_stress_threshold_c", () => {
    expect(temperatureTier(10)).toBe("bad");
  });

  it("is bad symmetrically far above optimal on the hot side", () => {
    expect(temperatureTier(38)).toBe("bad");
  });

  it("is caution moderately off from optimal, not yet bad", () => {
    expect(temperatureTier(17)).toBe("caution");
  });
});
