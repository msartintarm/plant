import { test, expect, type Page } from "@playwright/test";

// Real gameplay/rendering e2e coverage lands once there's an actual plant
// to assert on (see the engine's wgpu bring-up + growth sim passes). For
// now this is a wiring smoke test: does the wasm module actually reach the
// browser and construct without erroring.

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
