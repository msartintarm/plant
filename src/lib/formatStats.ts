// Pure formatting helpers for the HUD — kept separate from EngineCanvas.tsx
// so they're unit-testable without a wasm module or a canvas.

/// Maps the engine's `day_progress` (0..1, wraps at midnight — see
/// `sim::sun`) onto a familiar 12-hour clock. 0 is midnight, 0.5 is noon,
/// matching the engine's own sunrise=0.25/sunset=0.75 convention.
export function formatTimeOfDay(dayProgress: number): string {
  const wrapped = ((dayProgress % 1) + 1) % 1;
  const totalMinutes = Math.round(wrapped * 24 * 60) % (24 * 60);
  const hours24 = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  const period = hours24 < 12 ? "AM" : "PM";
  const hours12 = hours24 % 12 === 0 ? 12 : hours24 % 12;
  return `${hours12}:${minutes.toString().padStart(2, "0")} ${period}`;
}

/// 0..1 fraction to a whole-number percentage string, clamped so a
/// slightly-out-of-range float (e.g. 1.0000001 from float rounding) doesn't
/// show as "101%".
export function formatPercent(fraction: number): string {
  const clamped = Math.min(1, Math.max(0, fraction));
  return `${Math.round(clamped * 100)}%`;
}

/// One decimal place — height/radius are dimensionless sim units, not a
/// real-world measurement, so more precision than this isn't meaningful to
/// show a player.
export function formatHeight(height: number): string {
  return height.toFixed(1);
}
