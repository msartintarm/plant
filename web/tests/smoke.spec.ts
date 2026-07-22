import { test, expect, type Page } from "@playwright/test";

// This sim runs on its own live wgpu clock, not deterministic gameplay
// state, so these tests assert on things that actually indicate a
// regression (the wasm engine loads, the canvas renders, the HUD reflects
// live engine state, the controls don't error) rather than exact values.

function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(`console.error: ${msg.text()}`);
  });
  return errors;
}

async function addFirstPlant(page: Page) {
  await page.getByRole("button", { name: "Add plant" }).click();
  await expect(page.getByText(/Seed|Sprout|Vegetative/)).toBeVisible({ timeout: 2_000 });
}

async function openSettings(page: Page) {
  await page.getByRole("button", { name: "⚙️ Settings" }).click();
}

// Regression test for a real incident: a WGSL shader change (the cursor
// specular highlight) started reading `instance.*` fields from the
// fragment stage, but the group(0) bind group layout was still declared
// `visibility: VERTEX` only (correct back when only the vertex stage ever
// touched it). That's a *pipeline creation* validation error, not a
// runtime one — it doesn't just skip the new effect, it invalidates the
// whole render pipeline, which invalidates every command buffer built
// against it, which means literally nothing draws. The console-error
// checks already threaded through every test below would have caught this
// (WebGPU reports it as a `console.error`), but nothing was actually re-run
// against a real browser after that change landed — this test exists so
// there's always at least one that's explicitly about "does the canvas
// have real content," not just "did anything throw."
//
// A plain `drawImage`+`getImageData` readback of the canvas reliably
// returns all-zero pixels for a WebGPU-backed canvas in this Playwright/
// Chromium setup regardless of what's actually rendered (confirmed by hand
// against a real screenshot showing correct output while that technique
// reported nothing) — so this instead screenshots the canvas (the browser's
// own compositor, not JS-side canvas readback) and checks the PNG's byte
// size. A solid-color/blank canvas compresses to ~3KB; this scene's actual
// mix of colors/edges/gradients compresses to 50KB+, so a wide, low
// threshold cleanly separates "genuinely rendering" from "blank" without
// needing a PNG decoder or a checked-in baseline image to compare against.
test("the canvas renders real (non-blank) content, not just a solid clear color", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });
  await addFirstPlant(page);
  // A few real frames, not just the first one — some effects (the cursor
  // light/specular, the pick pass) only run once the loop has been going a
  // moment.
  await page.waitForTimeout(2_000);

  const screenshot = await page.locator("canvas").screenshot();
  expect(
    screenshot.length,
    `canvas screenshot was only ${screenshot.length} bytes — that's blank-canvas-sized, not a rendered scene`,
  ).toBeGreaterThan(10_000);

  expect(errors).toEqual([]);
});

test("loads the page and the wasm engine without erroring", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Houseplant" })).toBeVisible();
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });
  await expect(page.locator("canvas")).toBeVisible();

  expect(errors).toEqual([]);
});

test("the room starts with no plant until one is added", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });
  await expect(page.getByRole("button", { name: "Add plant" })).toBeVisible();
  await expect(page.getByText(/Seed|Sprout|Vegetative/)).not.toBeVisible();

  await addFirstPlant(page);
  await expect(page.getByRole("button", { name: "Add plant" })).toBeVisible();

  expect(errors).toEqual([]);
});

test("the starting inventory is spent as plants are added", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });

  const addButton = page.getByRole("button", { name: "Add plant" });
  await addButton.click();
  await addButton.click();
  await addButton.click();
  await expect(addButton).not.toBeVisible();

  expect(errors).toEqual([]);
});

test("the HUD reflects live engine state once it starts running", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });
  await addFirstPlant(page);
  await openSettings(page);

  await expect(page.getByText(/Height: \d/)).toBeVisible();
  await expect(page.getByText(/Leaves: \d+ · Branches: \d+/)).toBeVisible();
  await expect(page.getByText(/💧 Water: \d+%/)).toBeVisible();
  await expect(page.getByText(/🌡️ -?\d+°C/)).toBeVisible();

  expect(errors).toEqual([]);
});

test("the water button and time-scale slider don't error when used", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });
  await addFirstPlant(page);
  await openSettings(page);

  await page.getByRole("button", { name: "Water" }).click();
  const slider = page.getByLabel(/Speed:/);
  await slider.fill("2.5");
  await expect(page.getByText("Speed: 2.50x")).toBeVisible();

  // Give the sim a moment to keep running at the new speed without
  // erroring, rather than just checking the click/fill themselves worked.
  await page.waitForTimeout(500);
  expect(errors).toEqual([]);
});

test("the auto-water toggle enables without erroring and disables the manual button", async ({
  page,
}) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });
  await addFirstPlant(page);
  await openSettings(page);

  const autoWaterCheckbox = page.getByLabel("Auto-water");
  const waterButton = page.getByRole("button", { name: "Water" });
  await expect(waterButton).toBeEnabled();

  await autoWaterCheckbox.check();
  await expect(waterButton).toBeDisabled();

  await page.waitForTimeout(500);
  expect(errors).toEqual([]);

  await autoWaterCheckbox.uncheck();
  await expect(waterButton).toBeEnabled();
});

test("a default no-input session grows past the first leaf (real browser/wasm loop, not the native sim harness)", async ({
  page,
}) => {
  // Regression test for a real browser-vs-native divergence: `cargo test`
  // playthrough harnesses predicted multiple leaves within seconds, but
  // manual testing showed the actual page stuck at "Leaves: 1" for a full
  // minute. Running through the real render loop (requestAnimationFrame,
  // actual wasm calls) is the only way to catch a bug that's specific to
  // that path rather than the native step-by-step harness.
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });
  await addFirstPlant(page);
  await openSettings(page);

  // Speed up sim time so this doesn't need a full real minute of wall-clock
  // waiting — 5x is the slider's own max.
  const slider = page.getByLabel(/Speed:/);
  await slider.fill("5");

  await page.waitForTimeout(15_000);

  const leafText = await page.getByText(/Leaves: \d+/).textContent();
  const leafCount = Number(leafText?.match(/Leaves: (\d+)/)?.[1]);
  expect(leafCount, `expected leaf growth past the first leaf, got "${leafText}"`).toBeGreaterThan(1);

  expect(errors).toEqual([]);
});

test("switching species resets the HUD and doesn't error", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });
  await addFirstPlant(page);
  await openSettings(page);

  const speciesSelect = page.getByLabel("Species:");
  await speciesSelect.selectOption("peace_lily");
  // Switching species starts a fresh plant — germination happens almost
  // immediately at this demo's pacing (soil starts well above the
  // threshold), so "Seed" can flash by faster than a poll cycle; "no
  // branches yet" is the durable post-reset signal instead.
  await expect(page.getByText(/Branches: 0/)).toBeVisible({ timeout: 2_000 });

  await page.waitForTimeout(500);
  expect(errors).toEqual([]);
});

test("the prune and trim tools are independently selectable", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });
  await addFirstPlant(page);
  await openSettings(page);

  const slider = page.getByLabel(/Speed:/);
  await slider.fill("5");

  const pruneButton = page.getByRole("button", { name: "🔪 Prune" });
  const trimButton = page.getByRole("button", { name: "✂️ Trim" });
  await expect(pruneButton).toBeVisible();
  await expect(trimButton).toBeVisible({ timeout: 15_000 });

  await trimButton.click();
  await pruneButton.click();

  expect(errors).toEqual([]);
});

// Regression test for a real bug report: the moon's phase looked frozen
// across a session with a few restarts in it. Root cause was `render::mod`
// driving the moon off `Plant::total_time` (which resets to 0 on every
// restart/species-switch/cutting) instead of a persistent session clock —
// every restart snapped the moon back near its starting phase. `sim::moon`'s
// own design is a real, ongoing astronomical cycle that shouldn't care
// whether the current plant is brand new, so this asserts the fix directly:
// switching species must never move the displayed moon phase backward.
test("the moon's phase keeps advancing across a species switch instead of resetting", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });
  await addFirstPlant(page);

  await page.getByRole("button", { name: "🌱 Seed info" }).click();
  const moonReading = page.getByText(/Moon: \d+% lit/);
  await expect(moonReading).toBeVisible();

  await openSettings(page);
  const slider = page.getByLabel(/Speed:/);
  await slider.fill("5");
  await page.waitForTimeout(2_000);

  const readMoonPercent = async () => {
    const text = await moonReading.textContent();
    const match = text?.match(/Moon: (\d+)% lit/);
    return match ? Number(match[1]) : null;
  };

  const beforeSwitch = await readMoonPercent();
  expect(beforeSwitch, `couldn't read the moon reading from "${await moonReading.textContent()}"`).not.toBeNull();

  const speciesSelect = page.getByLabel("Species:");
  await speciesSelect.selectOption("pothos");
  await expect(page.getByText(/Branches: 0/)).toBeVisible({ timeout: 2_000 });

  await page.waitForTimeout(2_000);
  const afterSwitch = await readMoonPercent();
  expect(afterSwitch, `couldn't read the moon reading from "${await moonReading.textContent()}"`).not.toBeNull();

  expect(
    afterSwitch,
    `moon reading went from ${beforeSwitch}% to ${afterSwitch}% across a species switch — it should only ever move forward`,
  ).toBeGreaterThanOrEqual(beforeSwitch!);

  expect(errors).toEqual([]);
});

// Regression test for a user report that the moon "looks stuck" on one
// phase. A moon-phase snapshot alone can't tell whether it's genuinely
// frozen or just slow near a flat point of the illumination curve, so this
// watches the same "Moon: X% lit" reading move over a real accelerated
// window instead of inferring anything from the underlying math.
test("the moon's illuminated-fraction reading visibly changes over a short accelerated window", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });
  await addFirstPlant(page);

  await page.getByRole("button", { name: "🌱 Seed info" }).click();
  const moonReading = page.getByText(/Moon: \d+% lit/);
  await expect(moonReading).toBeVisible();

  await openSettings(page);
  const slider = page.getByLabel(/Speed:/);
  await slider.fill("5");

  const readMoonPercent = async () => {
    const text = await moonReading.textContent();
    const match = text?.match(/Moon: (\d+)% lit/);
    return match ? Number(match[1]) : null;
  };

  const before = await readMoonPercent();
  expect(before, `couldn't read the moon reading from "${await moonReading.textContent()}"`).not.toBeNull();

  await page.waitForTimeout(6_000);

  const after = await readMoonPercent();
  expect(after, `couldn't read the moon reading from "${await moonReading.textContent()}"`).not.toBeNull();

  expect(after, `moon reading stayed at ${before}% lit over 6s at 5x speed — expected it to visibly move`).not.toBe(before);

  expect(errors).toEqual([]);
});
