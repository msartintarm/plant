"use client";

import { useEffect, useRef, useState } from "react";
import { basePath } from "@/lib/basePath";
import {
  formatDays,
  formatHeight,
  formatNutrient,
  formatPercent,
  formatTemperature,
  formatTimeOfDay,
} from "@/lib/formatStats";
import {
  humidityTier,
  nutrientTier,
  pestTier,
  rootHealthTier,
  temperatureTier,
  waterTier,
  type HealthTier,
} from "@/lib/healthTiers";
import { CHALLENGES, loadCompletedChallenges, newlyCompletedChallenges, saveCompletedChallenges } from "@/lib/challenges";
import { loadHighScores, mergeHighScores, saveHighScores, type PlantMetrics } from "@/lib/scoring";
import styles from "./EngineCanvas.module.css";

// Maps a gauge's current tier (see src/lib/healthTiers.ts) to its CSS
// module class — "good" gets its own subtle green rather than falling back
// to the default text color, so a healthy reading is a positive, visible
// signal rather than just the absence of a warning.
function tierClassName(tier: HealthTier): string {
  if (tier === "bad") return styles.tierBad;
  if (tier === "caution") return styles.tierCaution;
  return styles.tierGood;
}

interface EngineStats {
  day_progress: number;
  is_daytime: boolean;
  stage: string;
  height: number;
  leaf_count: number;
  branch_count: number;
  stem_segment_count: number;
  water_level: number;
  temperature_c: number;
  nutrient_level: number;
  humidity_level: number;
  root_health: number;
  pest_infestation: number;
  day_length_factor: number;
  pot_position: number;
  auto_water_enabled: boolean;
  realistic_scale: boolean;
  death_cause: string;
  season: string;
  days_elapsed: number;
  hover_active: boolean;
  moon_illuminated_fraction: number;
  max_height_reached: number;
  max_leaves_at_once: number;
  leaves_produced_total: number;
  alive_duration: number;
  alive_days: number;
  bloom_intensity: number;
}

interface ScreenPosition {
  x: number;
  y: number;
}

// One per plant, polled every cycle alongside `plantTabs` — unlike
// `EngineStats` (which only ever describes the *selected* plant), this
// backs the small water gauge/button drawn under *every* pot at once (see
// `Simulation::plant_pot_hud`).
interface PlantPotHud {
  x: number;
  y: number;
  water_level: number;
  auto_water_enabled: boolean;
  is_dead: boolean;
}

interface EngineSimulation {
  start(): void;
  stop(): void;
  resize(width: number, height: number, devicePixelRatio: number): void;
  water(amount: number): void;
  water_plant(index: number, amount: number): void;
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
  set_species(species: string, realisticScale: boolean): void;
  restart(): void;
  set_pointer_position(x: number, y: number): void;
  clear_pointer_position(): void;
  pan_camera(dx: number, dy: number): void;
  reset_camera_pan(): void;
  has_hover_target(): boolean;
  prune_hovered(): boolean;
  stats(): EngineStats | undefined;
  set_active_tool(tool: string): void;
  plant_count(): number;
  plant_species(index: number): string;
  selected_plant_index(): number;
  set_selected_plant(index: number): void;
  inventory_count(): number;
  inventory_species(index: number): string;
  grant_seed(species: string): void;
  plant_cutting(index: number, realisticScale: boolean): boolean;
  plant_pot_hud(index: number): PlantPotHud | undefined;
  lamp_screen_position(): ScreenPosition;
  free(): void;
}

interface EngineModule {
  default: (input?: unknown) => Promise<unknown>;
  Simulation: {
    create(
      canvas: HTMLCanvasElement,
      devicePixelRatio: number,
      seedYear: number,
      seedMonth: number,
      seedDay: number,
    ): Promise<EngineSimulation>;
  };
}

// Location is display-only (see the seed-info tag in the JSX below) — the
// moon's *phase* doesn't depend on where you are, only real moonrise/set
// *timing* would, and this stylized side-profile scene doesn't model that
// at all (see engine/src/sim/moon.rs). Fixed rather than pulled from the
// browser's geolocation API: there's no actual calculation for a real
// location to feed, so asking for that permission would just be a prompt
// with nothing behind it.
const SEED_LOCATION = "San Francisco, CA";

type Status = "loading" | "ready" | "error";

// How often the HUD re-reads engine state — far coarser than the render
// loop's own per-frame pace, since nothing here needs to be read that
// often for a human to watch it change.
const STATS_POLL_MS = 250;
// Total pointer travel (CSS px) before a press-and-move gesture counts as a
// drag-to-pan rather than a click/tap-to-prune — small enough to feel
// immediate, large enough that a slightly wobbly tap still prunes.
const DRAG_THRESHOLD_PX = 4;
const WATER_DOSE = 0.05;
const FERTILIZE_DOSE = 0.3;
const MIST_DOSE = 0.3;
const DEFAULT_TIME_SCALE = 2.0;
const DEFAULT_SPECIES = "dracaena";
const SPECIES_OPTIONS: { value: string; label: string }[] = [
  { value: "dracaena", label: "Dracaena (branching)" },
  { value: "peace_lily", label: "Peace Lily (rosette)" },
  { value: "pothos", label: "Pothos (climbing)" },
];
const UNLOCKABLE_SPECIES = ["peace_lily", "pothos"];

// A short label/icon for a species name coming back from the engine (plant
// tabs, inventory items) — falls back to the raw name for anything not in
// `SPECIES_OPTIONS` rather than showing nothing.
const SPECIES_ICONS: Record<string, string> = {
  dracaena: "🌴",
  peace_lily: "🌼",
  pothos: "🍃",
};

function speciesLabel(species: string): string {
  return SPECIES_OPTIONS.find((option) => option.value === species)?.label ?? species;
}

// One short, actionable clause per cause — the death overlay stays to 1-2
// sentences total.
const DEATH_EXPLANATIONS: Record<string, string> = {
  "Root rot": "Water it less often next time.",
  Starvation: "Give it more light or water next time.",
};

const SEASON_ICONS: Record<string, string> = {
  Spring: "🌱",
  Summer: "🌻",
  Autumn: "🍂",
  Winter: "❄️",
};

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

// Canvas-relative CSS pixels — what `Simulation::set_pointer_position`
// expects (see that method's own doc comment on why it takes CSS pixels,
// not the devicePixelRatio-scaled backing-buffer ones `resize` takes).
function canvasRelativePosition(canvas: HTMLCanvasElement, clientX: number, clientY: number): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  return { x: clientX - rect.left, y: clientY - rect.top };
}

export default function EngineCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<EngineSimulation | null>(null);
  // Drag-to-pan gesture state — see `handlePointerMove`/`handleClick` in
  // the effect below. Plain refs, not state: this updates every pointer
  // event, far too often to justify a re-render.
  const dragRef = useRef<{ lastX: number; lastY: number; totalDistance: number; didPan: boolean } | null>(null);
  const didPanRef = useRef(false);
  const [status, setStatus] = useState<Status>("loading");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [stats, setStats] = useState<EngineStats | null>(null);
  const [timeScale, setTimeScale] = useState(DEFAULT_TIME_SCALE);
  // Optimistic local mirror of the *selected* plant's own `Stats::
  // auto_water_enabled` — set immediately on toggle for a responsive
  // checkbox, then resynced from `stats` every poll (like `potPosition`
  // below) so switching plants shows that plant's own real setting.
  const [autoWater, setAutoWater] = useState(false);
  const [species, setSpecies] = useState(DEFAULT_SPECIES);
  const [potPosition, setPotPosition] = useState(0);
  const [showSeedInfo, setShowSeedInfo] = useState(false);
  const [activeTool, setActiveTool] = useState<"prune" | "trim" | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  // One entry per plant currently in the room, its own species name (see
  // `Simulation::plant_species`) — index lines up with `set_selected_
  // plant`'s own `index` param. Refreshed every poll alongside `stats`,
  // not just on the actions that change it, so switching plants/adding one
  // elsewhere always shows the room's real current state.
  const [plantTabs, setPlantTabs] = useState<string[]>([]);
  const [selectedPlantIndex, setSelectedPlantIndex] = useState(0);
  // One reading per plant, indexed the same as `plantTabs` — positions and
  // waters the small gauge/button drawn under each pot in the scene itself
  // (see `Simulation::plant_pot_hud`), independent of which plant the
  // settings HUD currently has selected.
  const [potHuds, setPotHuds] = useState<PlantPotHud[]>([]);
  const [lampPosition, setLampPosition] = useState<ScreenPosition | null>(null);
  const [debugUnlocked, setDebugUnlocked] = useState(false);
  const [newPlantSpecies, setNewPlantSpecies] = useState(DEFAULT_SPECIES);
  const [newPlantRealisticScale, setNewPlantRealisticScale] = useState(false);
  // One entry per stem cutting waiting to be planted (see `Simulation::
  // take_cutting`/`plant_cutting`), its own species name.
  const [inventory, setInventory] = useState<string[]>([]);
  // Best-ever reading per metric across every plant this browser has ever
  // grown (see src/lib/scoring.ts) and which milestone challenges (see
  // src/lib/challenges.ts) have ever been earned — both persisted to
  // localStorage, so they survive a restart/replaced plant rather than
  // resetting with it. Lazy initializers (like `seedDate` below), not a
  // mount effect — `loadHighScores`/`loadCompletedChallenges` already fall
  // back to zero/empty when `localStorage` isn't available.
  const [highScores, setHighScores] = useState<PlantMetrics>(() => loadHighScores());
  const [completedChallenges, setCompletedChallenges] = useState<Set<string>>(() => loadCompletedChallenges());
  const [showScores, setShowScores] = useState(false);
  // Captured once, at mount, rather than read fresh each render — this has
  // to be the *exact* same date actually handed to `Simulation.create`
  // below, not a second, independently-taken `new Date()` that could in
  // principle land a moment later.
  const [seedDate] = useState(() => new Date());

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
        const created = await mod.Simulation.create(
          canvas,
          window.devicePixelRatio || 1,
          seedDate.getFullYear(),
          seedDate.getMonth() + 1, // JS months are 0-indexed, the engine's aren't
          seedDate.getDate(),
        );
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
        if (loadCompletedChallenges().size >= CHALLENGES.length) {
          UNLOCKABLE_SPECIES.forEach((s) => created.grant_seed(s));
        }
        created.set_time_scale(DEFAULT_TIME_SCALE);
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
      simRef.current.resize(width, height, window.devicePixelRatio || 1);
    }

    // The prune tool — always active for now (see the tool badge in the
    // JSX below), so any hover/click over the canvas targets whatever leaf
    // is currently under the cursor (see `Simulation::set_pointer_position`
    // and its GPU pick-pass doc comment for how that's actually resolved).
    function handlePointerMove(event: PointerEvent) {
      const canvas = canvasRef.current;
      if (!canvas || !simRef.current) return;
      const { x, y } = canvasRelativePosition(canvas, event.clientX, event.clientY);
      simRef.current.set_pointer_position(x, y);

      // Dragging the background pans the room's view (see `Simulation::
      // pan_camera`) — `event.buttons !== 0` covers both a held mouse
      // button and an in-contact touch/pen, so mouse and touch drag the
      // same way. Below `DRAG_THRESHOLD_PX` of total movement, this stays
      // a plain click/tap (see `handleClick`) rather than a pan, so a
      // static tap still prunes.
      const drag = dragRef.current;
      if (drag && event.buttons !== 0) {
        const dx = event.clientX - drag.lastX;
        const dy = event.clientY - drag.lastY;
        drag.totalDistance += Math.hypot(dx, dy);
        drag.lastX = event.clientX;
        drag.lastY = event.clientY;
        if (drag.totalDistance > DRAG_THRESHOLD_PX) {
          drag.didPan = true;
          simRef.current.pan_camera(dx, dy);
        }
      }
    }

    function handlePointerDown(event: PointerEvent) {
      dragRef.current = { lastX: event.clientX, lastY: event.clientY, totalDistance: 0, didPan: false };
    }

    function handlePointerUpOrCancel() {
      // `click` fires after `pointerup`, so whether this gesture panned has
      // to survive into a ref `handleClick` can still read once `dragRef`
      // itself is cleared below.
      didPanRef.current = dragRef.current?.didPan ?? false;
      dragRef.current = null;
    }

    function handlePointerLeave() {
      simRef.current?.clear_pointer_position();
    }

    function handleClick() {
      // A drag that actually panned shouldn't also prune whatever's under
      // the pointer where it happened to end up — `dragRef` is already
      // cleared by pointerup by the time `click` fires, so the pan
      // decision is captured here instead, at the end of the gesture that
      // set it.
      if (didPanRef.current) {
        didPanRef.current = false;
        return;
      }
      simRef.current?.prune_hovered();
    }

    // Recenters a dragged-away view — double-click/double-tap is a common
    // enough "reset this" convention (maps, image viewers) to need no
    // extra on-screen control for it.
    function handleDoubleClick() {
      simRef.current?.reset_camera_pan();
    }

    load();
    window.addEventListener("resize", handleResize);
    const canvasEl = canvasRef.current;
    canvasEl?.addEventListener("pointermove", handlePointerMove);
    canvasEl?.addEventListener("pointerdown", handlePointerDown);
    canvasEl?.addEventListener("pointerup", handlePointerUpOrCancel);
    canvasEl?.addEventListener("pointercancel", handlePointerUpOrCancel);
    canvasEl?.addEventListener("pointerleave", handlePointerLeave);
    canvasEl?.addEventListener("click", handleClick);
    canvasEl?.addEventListener("dblclick", handleDoubleClick);
    return () => {
      cancelled = true;
      window.removeEventListener("resize", handleResize);
      canvasEl?.removeEventListener("pointermove", handlePointerMove);
      canvasEl?.removeEventListener("pointerdown", handlePointerDown);
      canvasEl?.removeEventListener("pointerup", handlePointerUpOrCancel);
      canvasEl?.removeEventListener("pointercancel", handlePointerUpOrCancel);
      canvasEl?.removeEventListener("pointerleave", handlePointerLeave);
      canvasEl?.removeEventListener("click", handleClick);
      canvasEl?.removeEventListener("dblclick", handleDoubleClick);
      simRef.current?.stop();
      simRef.current?.free();
      simRef.current = null;
    };
  }, [seedDate]);

  // A separate effect (rather than folding this into the load effect above)
  // since it only needs `status`, not anything from the load/cleanup
  // lifecycle — starts once the sim is actually ready, stops on unmount.
  useEffect(() => {
    if (status !== "ready") return;
    const interval = setInterval(() => {
      const sim = simRef.current;
      if (!sim) return;
      const currentStats = sim.stats();
      setStats(currentStats ?? null);
      // A pointer cursor over a hover-picked leaf or stem segment is the
      // only affordance that the prune tool is about to do something on
      // click — set directly on the element (not React state) since this
      // only needs to happen at the same coarse poll rate as everything
      // else here, not trigger its own render.
      const canvas = canvasRef.current;
      if (canvas) canvas.style.cursor = sim.has_hover_target() ? "pointer" : "default";

      // Every plant in the room (see `Simulation::plant_species`), the
      // room's shared cutting inventory, and which plant is currently
      // selected — polled fresh every cycle (not just after an action that
      // changes one of them) so a plant added or a cutting taken/planted
      // from any code path always shows up promptly.
      const plantCount = sim.plant_count();
      const tabs: string[] = [];
      const huds: PlantPotHud[] = [];
      for (let i = 0; i < plantCount; i++) {
        tabs.push(sim.plant_species(i));
        const hud = sim.plant_pot_hud(i);
        if (hud) huds.push(hud);
      }
      setPlantTabs(tabs);
      setPotHuds(huds);
      setLampPosition(sim.lamp_screen_position());
      const selected = sim.selected_plant_index();
      setSelectedPlantIndex(selected);
      // Keeps the HUD's own species selector and pot-placement slider
      // showing *this* plant's real state rather than whatever was left
      // over from a previously-selected plant.
      if (tabs[selected]) setSpecies(tabs[selected]);
      if (currentStats) {
        setPotPosition(currentStats.pot_position);
        setAutoWater(currentStats.auto_water_enabled);

        // Rolls this plant's current lifetime metrics into the browser's
        // running best-ever record (see src/lib/scoring.ts), then checks
        // whether that updated record just crossed any milestone challenge
        // threshold (src/lib/challenges.ts) — both persisted to
        // localStorage only when something actually changed, so a
        // steady-state plant that isn't setting any new records doesn't
        // write to storage every poll tick.
        const freshMetrics: PlantMetrics = {
          maxHeightReached: currentStats.max_height_reached,
          maxLeavesAtOnce: currentStats.max_leaves_at_once,
          leavesProducedTotal: currentStats.leaves_produced_total,
          aliveDays: currentStats.alive_days,
        };
        setHighScores((previousScores) => {
          const { scores, improved } = mergeHighScores(previousScores, freshMetrics);
          if (improved.length === 0) return previousScores;
          saveHighScores(scores);
          setCompletedChallenges((previousCompleted) => {
            const newlyDone = newlyCompletedChallenges(scores, previousCompleted);
            if (newlyDone.length === 0) return previousCompleted;
            const nextCompleted = new Set(previousCompleted);
            newlyDone.forEach((id) => nextCompleted.add(id));
            saveCompletedChallenges(nextCompleted);
            if (previousCompleted.size < CHALLENGES.length && nextCompleted.size >= CHALLENGES.length) {
              UNLOCKABLE_SPECIES.forEach((s) => sim.grant_seed(s));
            }
            return nextCompleted;
          });
          return scores;
        });
      }

      const inventoryCount = sim.inventory_count();
      const items: string[] = [];
      for (let i = 0; i < inventoryCount; i++) items.push(sim.inventory_species(i));
      setInventory(items);
    }, STATS_POLL_MS);
    return () => clearInterval(interval);
  }, [status]);

  const isDead = stats?.stage === "Dead";
  const inventoryCounts = inventory.reduce<Record<string, number>>((acc, s) => {
    acc[s] = (acc[s] ?? 0) + 1;
    return acc;
  }, {});
  const availableSpecies = Object.keys(inventoryCounts);
  const effectiveNewPlantSpecies = availableSpecies.includes(newPlantSpecies) ? newPlantSpecies : availableSpecies[0];

  function handleWater() {
    simRef.current?.water(WATER_DOSE);
  }

  // Waters a specific pot directly (see the per-pot gauge in the JSX below)
  // without disturbing which plant the settings HUD has selected.
  function handleWaterPlant(index: number) {
    simRef.current?.water_plant(index, WATER_DOSE);
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

  function handleResetProgress() {
    if (!window.confirm("Erase all saved high scores and challenge progress?")) return;
    saveHighScores({ maxHeightReached: 0, maxLeavesAtOnce: 0, leavesProducedTotal: 0, aliveDays: 0 });
    saveCompletedChallenges(new Set());
    window.location.reload();
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

  function handleToolChange(tool: "prune" | "trim") {
    const next = activeTool === tool ? null : tool;
    setActiveTool(next);
    simRef.current?.set_active_tool(next ?? "");
  }

  // Switches which plant the HUD/actions target — every plant in the room
  // keeps rendering/growing regardless (see `Simulation::set_selected_
  // plant`'s own doc comment), this just changes which one this component
  // reads/controls. Clears the stale stats snapshot the same way switching
  // species does, since the newly-selected plant's own numbers are what
  // should show, not whatever the previous one's last poll left behind.
  function handleSelectPlant(index: number) {
    simRef.current?.set_selected_plant(index);
    setStats(null);
  }

  function handlePlantFromInventory(species: string) {
    const index = inventory.indexOf(species);
    if (index === -1) return;
    const planted = simRef.current?.plant_cutting(index, newPlantRealisticScale);
    if (planted) setStats(null);
  }

  return (
    <div className={styles.wrapper}>
      <canvas ref={canvasRef} className={styles.canvas} />
      {status === "ready" && lampPosition && !debugUnlocked && (
        <button
          type="button"
          onClick={() => setDebugUnlocked(true)}
          className={styles.lampHotspot}
          style={{ left: lampPosition.x, top: lampPosition.y }}
          aria-label="lamp"
        />
      )}
      {status === "ready" &&
        potHuds.map((hud, index) => (
          <div
            key={index}
            className={styles.potWaterBadge}
            style={{ left: hud.x, top: hud.y }}
            title={`Plant ${index + 1} — ${Math.round(hud.water_level * 100)}% water`}
          >
            <span className={tierClassName(waterTier(hud.water_level))}>💧 {formatPercent(hud.water_level)}</span>
            <button
              type="button"
              onClick={() => handleWaterPlant(index)}
              className={styles.actionButton}
              disabled={hud.auto_water_enabled || hud.is_dead}
              aria-label={`Water plant ${index + 1}`}
            >
              Water
            </button>
          </div>
        ))}
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
          <p>
            💀 Your plant died{stats?.death_cause ? ` of ${stats.death_cause.toLowerCase()}` : ""}.
            {stats?.death_cause && DEATH_EXPLANATIONS[stats.death_cause] && ` ${DEATH_EXPLANATIONS[stats.death_cause]}`}
          </p>
          <button type="button" onClick={handleRestart} className={styles.restartButton}>
            Start a new seed
          </button>
        </div>
      )}
      {status === "ready" && (
        <div className={styles.plantSelectorBar}>
          {plantTabs.map((tabSpecies, index) => (
            <button
              key={index}
              type="button"
              onClick={() => handleSelectPlant(index)}
              className={`${styles.plantTab} ${index === selectedPlantIndex ? styles.plantTabSelected : ""}`}
              title={speciesLabel(tabSpecies)}
            >
              {SPECIES_ICONS[tabSpecies] ?? "🪴"}
              {index + 1}
            </button>
          ))}
          {availableSpecies.length > 0 && (
            <>
              <select
                aria-label="New plant species"
                value={effectiveNewPlantSpecies}
                onChange={(e) => setNewPlantSpecies(e.target.value)}
                className={styles.speciesSelect}
              >
                {availableSpecies.map((s) => (
                  <option key={s} value={s}>
                    {speciesLabel(s)} x{inventoryCounts[s]}
                  </option>
                ))}
              </select>
              {debugUnlocked && (
                <label className={styles.realisticScaleLabel} title="Cap growth near a real houseplant's mature size instead of growing indefinitely">
                  <input
                    type="checkbox"
                    checked={newPlantRealisticScale}
                    onChange={(e) => setNewPlantRealisticScale(e.target.checked)}
                  />
                  Realistic scale
                </label>
              )}
              <button
                type="button"
                onClick={() => handlePlantFromInventory(effectiveNewPlantSpecies)}
                className={styles.plantTab}
                title="Plant one from your inventory"
              >
                Add plant
              </button>
            </>
          )}
        </div>
      )}
      {stats && (stats.leaf_count > 0 || stats.stem_segment_count > 0) && (
        <div className={styles.toolTag}>
          {stats.stem_segment_count > 0 && (
            <button
              type="button"
              onClick={() => handleToolChange("trim")}
              className={`${styles.toolButton} ${activeTool === "trim" ? styles.toolButtonActive : ""}`}
              title="Hover the stem (even under a leaf) and click to cut it at that point"
            >
              ✂️ Trim
            </button>
          )}
          {stats.leaf_count > 0 && (
            <button
              type="button"
              onClick={() => handleToolChange("prune")}
              className={`${styles.toolButton} ${activeTool === "prune" ? styles.toolButtonActive : ""}`}
              title="Hover a leaf and click to remove it"
            >
              🔪 Prune
            </button>
          )}
        </div>
      )}
      {stats && (
        <div className={styles.topRightStack}>
          <div className={styles.seasonPlaque} title={`Day ${stats.days_elapsed} of this plant's life`}>
            <span className={styles.seasonIcon}>{SEASON_ICONS[stats.season] ?? "🗓️"}</span>
            <span className={styles.seasonText}>
              {stats.season}
              <br />
              Day {stats.days_elapsed}
            </span>
          </div>
          <div className={styles.seedTagRow}>
            <button
              type="button"
              className={styles.seedTag}
              onClick={() => setShowSeedInfo((shown) => !shown)}
              aria-expanded={showSeedInfo}
            >
              🌱 Seed info
            </button>
            {showSeedInfo && (
              <div className={styles.seedInfoPanel} title="Grounds the moon's starting phase in a real date">
                Seeded {seedDate.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" })}
                <br />
                {SEED_LOCATION}
                <br />
                Moon: {formatPercent(stats.moon_illuminated_fraction)} lit
              </div>
            )}
          </div>
          <div className={styles.seedTagRow}>
            <button
              type="button"
              className={styles.seedTag}
              onClick={() => setShowScores((shown) => !shown)}
              aria-expanded={showScores}
            >
              🏆 Scores ({completedChallenges.size}/{CHALLENGES.length})
            </button>
            {showScores && (
              <div className={styles.scoresPanel}>
                <div className={styles.scoresPanelTitle}>Best ever (this browser)</div>
                <div className={styles.scoreRow}>
                  <span>📏 Tallest</span>
                  <span>{formatHeight(highScores.maxHeightReached)}</span>
                </div>
                <div className={styles.scoreRow}>
                  <span>🍃 Most leaves at once</span>
                  <span>{highScores.maxLeavesAtOnce}</span>
                </div>
                <div className={styles.scoreRow}>
                  <span>🌿 Leaves grown, lifetime</span>
                  <span>{highScores.leavesProducedTotal}</span>
                </div>
                <div className={styles.scoreRow}>
                  <span>⏳ Longest lived</span>
                  <span>{formatDays(highScores.aliveDays)}</span>
                </div>
                <div className={styles.scoresPanelTitle}>Challenges</div>
                <ul className={styles.challengeList}>
                  {CHALLENGES.map((challenge) => (
                    <li
                      key={challenge.id}
                      className={`${styles.challengeItem} ${
                        completedChallenges.has(challenge.id) ? styles.challengeItemDone : ""
                      }`}
                    >
                      <span>{completedChallenges.has(challenge.id) ? "✅" : "⬜"}</span>
                      <span>
                        {challenge.icon} {challenge.label}
                      </span>
                    </li>
                  ))}
                </ul>
                <button type="button" onClick={handleResetProgress} className={styles.resetProgressButton}>
                  Reset progress
                </button>
              </div>
            )}
          </div>
        </div>
      )}
      {stats && (
        <div className={styles.hudContainer}>
          <button
            type="button"
            className={styles.settingsToggle}
            onClick={() => setShowSettings((shown) => !shown)}
            aria-expanded={showSettings}
          >
            ⚙️ Settings
          </button>
          <div className={`${styles.hud} ${showSettings ? styles.hudOpen : ""}`}>
              <div className={styles.hudRow}>
                <span>Species: {speciesLabel(species)}</span>
              </div>
              {debugUnlocked && (
                <div className={styles.hudRow}>
                  <span>{stats.realistic_scale ? "🌿 realistic" : "🌳 gigantic"}</span>
                </div>
              )}
              <div className={styles.hudRow}>
                <span>
                  {stats.stage} · {stats.is_daytime ? "☀️" : "🌙"} {formatTimeOfDay(stats.day_progress)}
                </span>
                <span className={tierClassName(temperatureTier(stats.temperature_c))}>
                  🌡️ {formatTemperature(stats.temperature_c)}
                </span>
                <span title="How much winter's shorter days are slowing growth right now">
                  {stats.day_length_factor > 0.85 ? "🌱 Growing" : stats.day_length_factor > 0.6 ? "🍂 Slowing" : "❄️ Dormant"}
                </span>
              </div>
              <div className={styles.hudRow}>
                <span>Height: {formatHeight(stats.height)}</span>
                <span>
                  Leaves: {stats.leaf_count} · Branches: {stats.branch_count}
                </span>
                <span>🌸 Bloom: {formatPercent(stats.bloom_intensity)}</span>
              </div>

              <div className={`${styles.hudRow} ${tierClassName(waterTier(stats.water_level))}`}>
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
              <div className={`${styles.hudRow} ${tierClassName(nutrientTier(stats.nutrient_level))}`}>
                <span>🌱 Nutrient: {formatNutrient(stats.nutrient_level)}</span>
                <button type="button" onClick={handleFertilize} className={styles.actionButton} disabled={isDead}>
                  Fertilize
                </button>
              </div>
              <div className={`${styles.hudRow} ${tierClassName(humidityTier(stats.humidity_level))}`}>
                <span>💨 Humidity: {formatPercent(stats.humidity_level)}</span>
                <button type="button" onClick={handleMist} className={styles.actionButton} disabled={isDead}>
                  Mist
                </button>
              </div>
              <div className={`${styles.hudRow} ${tierClassName(rootHealthTier(stats.root_health))}`}>
                <span title="Drops from sustained overwatering or over-fertilizing — a damaged root system can wilt even when the soil reads fully watered">
                  🪴 Root health: {formatPercent(stats.root_health)}
                </span>
                <button type="button" onClick={handleRepot} className={styles.actionButton} disabled={isDead}>
                  Repot
                </button>
              </div>
              <div className={`${styles.hudRow} ${tierClassName(pestTier(stats.pest_infestation))}`}>
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
                  max={20}
                  step={0.25}
                  value={timeScale}
                  onChange={handleTimeScaleChange}
                  className={styles.timeScaleSlider}
                />
              </div>
          </div>
        </div>
      )}
    </div>
  );
}
