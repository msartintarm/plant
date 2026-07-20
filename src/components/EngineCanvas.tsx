"use client";

import { useEffect, useRef, useState } from "react";
import { basePath } from "@/lib/basePath";
import styles from "./EngineCanvas.module.css";

interface EngineSimulation {
  start(): void;
  stop(): void;
  resize(width: number, height: number): void;
  free(): void;
}

interface EngineModule {
  default: (input?: unknown) => Promise<unknown>;
  Simulation: {
    create(canvas: HTMLCanvasElement): Promise<EngineSimulation>;
  };
}

type Status = "loading" | "ready" | "error";

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
  const [status, setStatus] = useState<Status>("loading");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let sim: EngineSimulation | null = null;

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
        sim = created;
        sim.start();
        setStatus("ready");
      } catch (err) {
        if (cancelled) return;
        setErrorMessage(err instanceof Error ? err.message : String(err));
        setStatus("error");
      }
    }

    function handleResize() {
      const canvas = canvasRef.current;
      if (!canvas || !sim) return;
      const { width, height } = syncCanvasBackingSize(canvas);
      sim.resize(width, height);
    }

    load();
    window.addEventListener("resize", handleResize);
    return () => {
      cancelled = true;
      window.removeEventListener("resize", handleResize);
      sim?.stop();
      sim?.free();
    };
  }, []);

  return (
    <div className={styles.wrapper}>
      <canvas ref={canvasRef} className={styles.canvas} />
      {status !== "ready" && (
        <div className={styles.status}>
          {status === "loading" && <span>Loading engine…</span>}
          {status === "error" && (
            <div className={styles.error}>
              Engine failed to load{errorMessage ? `: ${errorMessage}` : ""}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
