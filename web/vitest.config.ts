import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Vitest's default glob would otherwise also match tests/*.spec.ts,
    // which are Playwright specs (import from @playwright/test, not
    // vitest) and would fail to run under this runner.
    include: ["src/**/*.test.ts"],
  },
});
