import { afterEach, describe, expect, it } from "vitest";
import { basePath } from "./basePath";

describe("basePath", () => {
  const original = process.env.NEXT_PUBLIC_BASE_PATH;

  afterEach(() => {
    process.env.NEXT_PUBLIC_BASE_PATH = original;
  });

  it("defaults to empty when unset", () => {
    delete process.env.NEXT_PUBLIC_BASE_PATH;
    expect(basePath()).toBe("");
  });

  it("returns the configured prefix", () => {
    process.env.NEXT_PUBLIC_BASE_PATH = "/plant";
    expect(basePath()).toBe("/plant");
  });
});
