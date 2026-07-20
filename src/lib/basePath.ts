// Mirrors next.config.ts's basePath — empty locally, e.g. "/plant" once
// deployed under a GitHub Pages subpath. Needed anywhere a raw absolute
// path (unlike next/link or next/image, which basePath rewrites
// automatically) is built by hand, such as the wasm-pkg import below.
export function basePath(): string {
  return process.env.NEXT_PUBLIC_BASE_PATH ?? "";
}
