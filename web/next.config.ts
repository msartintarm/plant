import type { NextConfig } from "next";

// Set at build time only (e.g. NEXT_PUBLIC_BASE_PATH=/plant in the GitHub
// Pages deploy workflow) — empty locally, so `npm run dev`/`npm run build`
// behave exactly as before unless this is explicitly set. Read the same way
// wherever a raw absolute path (not next/link or next/image, which basePath
// rewrites automatically) needs the prefix too — see the wasm-pkg import.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  // GitHub Pages only serves static files — no Node server, no API
  // routes/SSR. This app has neither, so nothing here is on Next's
  // static-export unsupported-features list.
  output: "export",
  basePath,
  // GH Pages serves literal files, not a server that rewrites /foo ->
  // /foo/index.html — trailingSlash makes Next emit /foo/index.html for a
  // /foo route directly, so a bare directory request resolves without a
  // custom server-side rewrite rule.
  trailingSlash: true,
};

export default nextConfig;
