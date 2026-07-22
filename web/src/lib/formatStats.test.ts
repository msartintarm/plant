import { describe, expect, it } from "vitest";
import { formatDays, formatHeight, formatNutrient, formatPercent, formatTemperature, formatTimeOfDay } from "./formatStats";

describe("formatTimeOfDay", () => {
  it("maps 0 to midnight", () => {
    expect(formatTimeOfDay(0)).toBe("12:00 AM");
  });

  it("maps 0.5 to noon", () => {
    expect(formatTimeOfDay(0.5)).toBe("12:00 PM");
  });

  it("maps 0.25 (sunrise) to 6am", () => {
    expect(formatTimeOfDay(0.25)).toBe("6:00 AM");
  });

  it("maps 0.75 (sunset) to 6pm", () => {
    expect(formatTimeOfDay(0.75)).toBe("6:00 PM");
  });

  it("wraps values past 1.0 the same way the engine's day_progress does", () => {
    expect(formatTimeOfDay(1.5)).toBe(formatTimeOfDay(0.5));
  });

  it("wraps negative values instead of producing a negative-minutes bug", () => {
    expect(formatTimeOfDay(-0.25)).toBe(formatTimeOfDay(0.75));
  });
});

describe("formatPercent", () => {
  it("formats a plain fraction", () => {
    expect(formatPercent(0.5)).toBe("50%");
  });

  it("clamps slightly-over-1 float rounding instead of showing >100%", () => {
    expect(formatPercent(1.0000001)).toBe("100%");
  });

  it("clamps slightly-under-0 instead of showing a negative percentage", () => {
    expect(formatPercent(-0.0000001)).toBe("0%");
  });
});

describe("formatHeight", () => {
  it("shows one decimal place", () => {
    expect(formatHeight(4.3219)).toBe("4.3");
  });

  it("still shows one decimal place for a whole number", () => {
    expect(formatHeight(0)).toBe("0.0");
  });
});

describe("formatNutrient", () => {
  it("formats a plain fraction like formatPercent", () => {
    expect(formatNutrient(0.5)).toBe("50%");
  });

  it("does not clamp above 100%, unlike formatPercent", () => {
    expect(formatNutrient(1.4)).toBe("140%");
  });

  it("still floors at 0% for slightly-negative float rounding", () => {
    expect(formatNutrient(-0.0000001)).toBe("0%");
  });
});

describe("formatTemperature", () => {
  it("rounds to the nearest whole degree", () => {
    expect(formatTemperature(21.4)).toBe("21°C");
    expect(formatTemperature(21.6)).toBe("22°C");
  });

  it("handles negative values", () => {
    expect(formatTemperature(-2.3)).toBe("-2°C");
  });
});

describe("formatDays", () => {
  it("shows one decimal place", () => {
    expect(formatDays(3)).toBe("3.0d");
    expect(formatDays(0.5)).toBe("0.5d");
  });

  it("floors slightly-negative float rounding at zero", () => {
    expect(formatDays(-0.0000001)).toBe("0.0d");
  });
});
