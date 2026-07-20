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

test("loads the page and the wasm engine without erroring", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Houseplant" })).toBeVisible();
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });
  await expect(page.locator("canvas")).toBeVisible();

  expect(errors).toEqual([]);
});

test("the HUD reflects live engine state once it starts running", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });

  // Stage starts at "Seed" and the HUD polls every 250ms — this should
  // appear well within a couple of poll cycles.
  await expect(page.getByText(/Seed|Sprout|Vegetative/)).toBeVisible({ timeout: 2_000 });
  await expect(page.getByText(/Height: \d/)).toBeVisible();
  await expect(page.getByText(/Leaves: \d+ · Branches: \d+/)).toBeVisible();
  await expect(page.getByText(/💧 Water: \d+%/)).toBeVisible();

  expect(errors).toEqual([]);
});

test("the water button and time-scale slider don't error when used", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });

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

test("switching species resets the HUD and doesn't error", async ({ page }) => {
  const errors = collectConsoleErrors(page);

  await page.goto("/");
  await expect(page.getByText("Loading engine…")).toBeHidden({ timeout: 15_000 });
  await expect(page.getByText(/Seed|Sprout|Vegetative/)).toBeVisible({ timeout: 2_000 });

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
