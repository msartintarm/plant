"use client";

import { useEffect, useRef, useState } from "react";
import { basePath } from "@/lib/basePath";
import {
  formatHeight,
  formatNutrient,
  formatPercent,
  formatTemperature,
  formatTimeOfDay,
} from "@/lib/formatStats";
import styles from "./EngineCanvas.module.css";

interface EngineStats {
  day_progress: number;
  is_daytime: boolean;
  stage: string;
  height: number;
  leaf_count: number;
  branch_count: number;
  water_level: number;
  temperature_c: number;
  nutrient_level: number;
  humidity_level: number;
  root_health: number;
  pest_infestation: number;
  day_length_factor: number;
  pot_position: number;
}

interface EngineSimulation {
  start(): void;
  stop(): void;
  resize(width: number, height: number): void;
  water(amount: number): void;
  fertilize(amount: number): void;
  mist(amount: number): void;
  treat_pests(): void;
  prune_main_stem(): boolean;
  prune_branch(index: number): boolean;
  repot(): boolean;
  take_cutting(): boolean;
  set_pot_position(position: number): void;
  set_time_scale(multiplier: number): void;
  set_auto_water(enabled: boolean): void;
  set_species(species: string): void;
  restart(): void;
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
const FERTILIZE_DOSE = 0.3;
const MIST_DOSE = 0.3;
const DEFAULT_TIME_SCALE = 1.0;
const DEFAULT_SPECIES = "dracaena";
const SPECIES_OPTIONS: { value: string; label: string }[] = [
  { value: "dracaena", label: "Dracaena (branching)" },
  { value: "peace_lily", label: "Peace Lily (rosette)" },
  { value: "pothos", label: "Pothos (climbing)" },
];

// Below this, a gauge's row switches to the warning style — a plant can
// still recover from any of these, but a player should notice before it
// gets there.
const ROOT_HEALTH_WARNING_THRESHOLD = 0.4;
const PEST_WARNING_THRESHOLD = 0.4;
const NUTRIENT_WARNING_THRESHOLD = 0.15;

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
  const [potPosition, setPotPosition] = useState(0);

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

  const isDead = stats?.stage === "Dead";

  function handleWater() {
    simRef.current?.water(WATER_DOSE);
  }

  function handleFertilize() {
    simRef.current?.fertilize(FERTILIZE_DOSE);
  }

  function handleMist() {
    simRef.current?.mist(MIST_DOSE);
  }

  function handleTreatPests() {
    simRef.current?.treat_pests();
  }

  function handlePruneMainStem() {
    simRef.current?.prune_main_stem();
  }

  // Targets the most recently formed branch — a simple, predictable default
  // rather than exposing a separate control per branch, which would need
  // its own layout that grows/shrinks with the crown itself.
  function handlePruneBranch() {
    const branchCount = stats?.branch_count ?? 0;
    if (branchCount <= 0) return;
    simRef.current?.prune_branch(branchCount - 1);
  }

  function handleRepot() {
    simRef.current?.repot();
  }

  function handleTakeCutting() {
    simRef.current?.take_cutting();
  }

  function handleRestart() {
    simRef.current?.restart();
  }

  function handlePotPositionChange(e: React.ChangeEvent<HTMLInputElement>) {
    const value = Number(e.target.value);
    setPotPosition(value);
    simRef.current?.set_pot_position(value);
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
      {isDead && (
        <div className={styles.deadOverlay}>
          <p>💀 Your plant has died.</p>
          <button type="button" onClick={handleRestart} className={styles.restartButton}>
            Start a new seed
          </button>
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
            <span>🌡️ {formatTemperature(stats.temperature_c)}</span>
            <span title="How much winter's shorter days are slowing growth right now">
              {stats.day_length_factor > 0.85 ? "🌱 Growing" : stats.day_length_factor > 0.6 ? "🍂 Slowing" : "❄️ Dormant"}
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
              className={styles.actionButton}
              disabled={autoWater || isDead}
            >
              Water
            </button>
            <label className={styles.autoWaterLabel}>
              <input type="checkbox" checked={autoWater} onChange={handleAutoWaterToggle} disabled={isDead} />
              Auto-water
            </label>
          </div>
          <div
            className={
              stats.nutrient_level < NUTRIENT_WARNING_THRESHOLD ? `${styles.hudRow} ${styles.warning}` : styles.hudRow
            }
          >
            <span>🌱 Nutrient: {formatNutrient(stats.nutrient_level)}</span>
            <button type="button" onClick={handleFertilize} className={styles.actionButton} disabled={isDead}>
              Fertilize
            </button>
          </div>
          <div className={styles.hudRow}>
            <span>💨 Humidity: {formatPercent(stats.humidity_level)}</span>
            <button type="button" onClick={handleMist} className={styles.actionButton} disabled={isDead}>
              Mist
            </button>
          </div>
          <div
            className={
              stats.root_health < ROOT_HEALTH_WARNING_THRESHOLD ? `${styles.hudRow} ${styles.warning}` : styles.hudRow
            }
          >
            <span title="Drops from sustained overwatering or over-fertilizing — a damaged root system can wilt even when the soil reads fully watered">
              🪴 Root health: {formatPercent(stats.root_health)}
            </span>
            <button type="button" onClick={handleRepot} className={styles.actionButton} disabled={isDead}>
              Repot
            </button>
          </div>
          <div
            className={
              stats.pest_infestation > PEST_WARNING_THRESHOLD ? `${styles.hudRow} ${styles.warning}` : styles.hudRow
            }
          >
            <span title="Spider mites — thrive in dry air, suppressed by misting">
              🐛 Pests: {formatPercent(stats.pest_infestation)}
            </span>
            <button
              type="button"
              onClick={handleTreatPests}
              className={styles.actionButton}
              disabled={isDead || stats.pest_infestation <= 0}
            >
              Treat
            </button>
          </div>

          <div className={styles.hudRow}>
            <label htmlFor="pot-position" title="Closer to the window means more light but a colder night draft; farther back is dimmer but climate-stable">
              🪟 Pot placement:
            </label>
            <input
              id="pot-position"
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={potPosition}
              onChange={handlePotPositionChange}
              className={styles.timeScaleSlider}
              disabled={isDead}
            />
            <span>{potPosition < 0.34 ? "At window" : potPosition < 0.67 ? "Nearby" : "Across room"}</span>
          </div>

          <div className={styles.hudRow}>
            <button type="button" onClick={handlePruneMainStem} className={styles.actionButton} disabled={isDead}>
              Prune stem
            </button>
            <button
              type="button"
              onClick={handlePruneBranch}
              className={styles.actionButton}
              disabled={isDead || stats.branch_count <= 0}
            >
              Prune branch
            </button>
            <button type="button" onClick={handleTakeCutting} className={styles.actionButton} disabled={isDead}>
              Take cutting
            </button>
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
