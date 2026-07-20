"use client";

import { useEffect, useRef, useState } from "react";
import { basePath } from "@/lib/basePath";
import { formatHeight, formatPercent, formatTimeOfDay } from "@/lib/formatStats";
import styles from "./EngineCanvas.module.css";

interface EngineStats {
  day_progress: number;
  is_daytime: boolean;
  stage: string;
  height: number;
  leaf_count: number;
  branch_count: number;
  water_level: number;
}

interface EngineSimulation {
  start(): void;
  stop(): void;
  resize(width: number, height: number): void;
  water(amount: number): void;
  set_time_scale(multiplier: number): void;
  set_auto_water(enabled: boolean): void;
  set_species(species: string): void;
  stats(): EngineStats;
  free(): void;
}

interface EngineModule {
  default: (input?: unknown) => Promise<unknown>;
  Simulation: {
    create(canvas: HTMLCanvasElement): Promise<EngineSimulation>;
  };
}

type Status = "loading" | "ready" | "error";

// How often the HUD re-reads engine state — far coarser than the render
// loop's own per-frame pace, since nothing here needs to be read that
// often for a human to watch it change.
const STATS_POLL_MS = 250;
const WATER_DOSE = 0.5;
const DEFAULT_TIME_SCALE = 1.0;
const DEFAULT_SPECIES = "dracaena";
const SPECIES_OPTIONS: { value: string; label: string }[] = [
  { value: "dracaena", label: "Dracaena (branching)" },
  { value: "peace_lily", label: "Peace Lily (rosette)" },
];

// Sizes the canvas's backing pixel buffer to match how large it's actually
// rendered (times devicePixelRatio, for crisp output on hi-DPI displays) —
// distinct from its CSS width/height, which the .canvas class already
// handles. wgpu reads these attributes (not the CSS size) when configuring
// the surface.
function syncCanvasBackingSize(canvas: HTMLCanvasElement): { width: number; height: number } {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  canvas.width = width;
  canvas.height = height;
  return { width, height };
}

export default function EngineCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<EngineSimulation | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [stats, setStats] = useState<EngineStats | null>(null);
  const [timeScale, setTimeScale] = useState(DEFAULT_TIME_SCALE);
  const [autoWater, setAutoWater] = useState(false);
  const [species, setSpecies] = useState(DEFAULT_SPECIES);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const canvas = canvasRef.current;
      if (!canvas) return;
      try {
        // Loaded from /public at runtime, not bundled — it's a wasm-pack
        // "web" target build, not an ES module webpack should process.
        const mod = (await import(
          /* webpackIgnore: true */ `${basePath()}/wasm-pkg/engine.js`
        )) as EngineModule;
        await mod.default();
        if (cancelled) return;

        syncCanvasBackingSize(canvas);
        const created = await mod.Simulation.create(canvas);
        if (cancelled) {
          // React StrictMode can invoke this effect twice in dev — if the
          // cleanup below already fired before create() resolved, stop
          // immediately instead of ever starting a second frame loop on the
          // same canvas.
          created.stop();
          created.free();
          return;
        }
        simRef.current = created;
        created.start();
        setStatus("ready");
      } catch (err) {
        if (cancelled) return;
        // WebGPU/WebGL2 adapter/device acquisition failing is the most
        // likely real-world cause of this branch (an unsupported browser),
        // so lead with that explanation — the raw error is still shown
        // underneath for anyone who needs to actually debug it.
        setErrorMessage(err instanceof Error ? err.message : String(err));
        setStatus("error");
      }
    }

    function handleResize() {
      const canvas = canvasRef.current;
      if (!canvas || !simRef.current) return;
      const { width, height } = syncCanvasBackingSize(canvas);
      simRef.current.resize(width, height);
    }

    load();
    window.addEventListener("resize", handleResize);
    return () => {
      cancelled = true;
      window.removeEventListener("resize", handleResize);
      simRef.current?.stop();
      simRef.current?.free();
      simRef.current = null;
    };
  }, []);

  // A separate effect (rather than folding this into the load effect above)
  // since it only needs `status`, not anything from the load/cleanup
  // lifecycle — starts once the sim is actually ready, stops on unmount.
  useEffect(() => {
    if (status !== "ready") return;
    const interval = setInterval(() => {
      const sim = simRef.current;
      if (!sim) return;
      setStats(sim.stats());
    }, STATS_POLL_MS);
    return () => clearInterval(interval);
  }, [status]);

  function handleWater() {
    simRef.current?.water(WATER_DOSE);
  }

  function handleTimeScaleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const value = Number(e.target.value);
    setTimeScale(value);
    simRef.current?.set_time_scale(value);
  }

  function handleAutoWaterToggle(e: React.ChangeEvent<HTMLInputElement>) {
    const enabled = e.target.checked;
    setAutoWater(enabled);
    simRef.current?.set_auto_water(enabled);
  }

  function handleSpeciesChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const next = e.target.value;
    setSpecies(next);
    // Switching species starts a fresh plant/soil under the new habit (see
    // Simulation::set_species) — the old stats snapshot no longer describes
    // anything that still exists, so clear it rather than show one stale
    // frame of the previous plant's numbers next to the new one's zeros.
    setStats(null);
    simRef.current?.set_species(next);
  }

  return (
    <div className={styles.wrapper}>
      <canvas ref={canvasRef} className={styles.canvas} />
      {status !== "ready" && (
        <div className={styles.status}>
          {status === "loading" && <span>Loading engine…</span>}
          {status === "error" && (
            <div className={styles.error}>
              <p>
                This demo needs a browser with WebGPU or WebGL2 support — try
                the latest Chrome, Edge, or Firefox.
              </p>
              {errorMessage && <p className={styles.errorDetail}>{errorMessage}</p>}
            </div>
          )}
        </div>
      )}
      {stats && (
        <div className={styles.hud}>
          <div className={styles.hudRow}>
            <label htmlFor="species">Species:</label>
            <select id="species" value={species} onChange={handleSpeciesChange} className={styles.speciesSelect}>
              {SPECIES_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div className={styles.hudRow}>
            <span>
              {stats.stage} · {stats.is_daytime ? "☀️" : "🌙"} {formatTimeOfDay(stats.day_progress)}
            </span>
          </div>
          <div className={styles.hudRow}>
            <span>Height: {formatHeight(stats.height)}</span>
            <span>
              Leaves: {stats.leaf_count} · Branches: {stats.branch_count}
            </span>
          </div>
          <div className={styles.hudRow}>
            <span>💧 Water: {formatPercent(stats.water_level)}</span>
            <button
              type="button"
              onClick={handleWater}
              className={styles.waterButton}
              disabled={autoWater}
            >
              Water
            </button>
            <label className={styles.autoWaterLabel}>
              <input type="checkbox" checked={autoWater} onChange={handleAutoWaterToggle} />
              Auto-water
            </label>
          </div>
          <div className={styles.hudRow}>
            <label htmlFor="time-scale">Speed: {timeScale.toFixed(2)}x</label>
            <input
              id="time-scale"
              type="range"
              min={0.25}
              max={5}
              step={0.25}
              value={timeScale}
              onChange={handleTimeScaleChange}
              className={styles.timeScaleSlider}
            />
          </div>
        </div>
      )}
    </div>
  );
}
