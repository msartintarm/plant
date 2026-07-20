import { describe, expect, it } from "vitest";
import { formatHeight, formatPercent, formatTimeOfDay } from "./formatStats";

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
