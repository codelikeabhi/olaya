#!/usr/bin/env bun
/**
 * The OpenCode → Olaya codemod. Deterministic and re-runnable, driven by script/rename-map.json.
 *
 *   bun script/olaya-rename.ts --dry-run     # report what would change
 *   bun script/olaya-rename.ts --apply       # rewrite contents, then git-mv renamed paths
 *   bun script/olaya-rename.ts --check       # exit 1 if anything is left to rename (idempotence)
 *
 * Per file: mask every protected token with a placeholder, apply the ordered replacements,
 * restore the placeholders. Masking is what makes protection exact: a broad rule such as
 * opencode → olaya can never rewrite part of a protected URL or provider id.
 *
 * The same rules translate an upstream patch into Olaya's namespace (product-plan W8).
 */

import fs from "fs"
import path from "path"

type Rules = {
  exclude: string[]
  protect: { why: string; re: string }[]
  protectInFiles: { files: string[]; re: string }
  protectPaths: string[]
  replace: [string, string][]
}

const root = path.resolve(import.meta.dir, "..")
const rules: Rules = JSON.parse(fs.readFileSync(path.join(import.meta.dir, "rename-map.json"), "utf8"))
const mode = process.argv.includes("--apply") ? "apply" : process.argv.includes("--check") ? "check" : "dry-run"

const excluded = rules.exclude.map((g) => new Bun.Glob(g))
const protectRes = rules.protect.map((p) => new RegExp(p.re, "g"))
const fileProtect = new Set(rules.protectInFiles.files)
const fileProtectRe = new RegExp(rules.protectInFiles.re, "g")

/** Rewrite a string: mask protected tokens, apply replacements in order, restore. */
export function rename(text: string, file = ""): { out: string; hits: number[] } {
  const saved: string[] = []
  const mask = (re: RegExp) => {
    text = text.replace(re, (m) => {
      saved.push(m)
      return `\u0000${saved.length - 1}\u0000`
    })
  }
  for (const re of protectRes) mask(re)
  if (fileProtect.has(file)) mask(fileProtectRe)
  const hits = rules.replace.map(([from, to]) => {
    let n = 0
    text = text.split(from).reduce((acc, part, i) => (i === 0 ? part : (n++, acc + to + part)), "")
    return n
  })
  const out = text.replace(/\u0000(\d+)\u0000/g, (_, i) => saved[Number(i)]!)
  return { out, hits }
}

function renamePath(p: string): string {
  if (rules.protectPaths.some((prefix) => p.startsWith(prefix))) return p
  return rename(p).out
}

function isBinary(buf: Buffer): boolean {
  return buf.subarray(0, 8192).includes(0)
}

function main() {
  const tracked = new TextDecoder()
    .decode(Bun.spawnSync(["git", "ls-files", "-z"], { cwd: root }).stdout)
    .split("\0")
    .filter(Boolean)

  const totals = rules.replace.map(() => 0)
  let changedFiles = 0
  const moves: [string, string][] = []

  for (const file of tracked) {
    if (excluded.some((g) => g.match(file))) continue
    const abs = path.join(root, file)
    if (!fs.existsSync(abs) || fs.lstatSync(abs).isSymbolicLink()) continue
    const buf = fs.readFileSync(abs)
    if (!isBinary(buf)) {
      const { out, hits } = rename(buf.toString("utf8"), file)
      if (hits.some((h) => h > 0)) {
        changedFiles++
        hits.forEach((h, i) => (totals[i] += h))
        if (mode === "apply") fs.writeFileSync(abs, out)
      }
    }
    const target = renamePath(file)
    if (target !== file) moves.push([file, target])
  }

  if (mode === "apply") {
    for (const [from, to] of moves) {
      fs.mkdirSync(path.dirname(path.join(root, to)), { recursive: true })
      const r = Bun.spawnSync(["git", "mv", "-k", from, to], { cwd: root })
      if (r.exitCode !== 0) console.error("git mv failed:", from, "->", to, new TextDecoder().decode(r.stderr))
    }
  }

  console.log(JSON.stringify({
    mode,
    changedFiles,
    movedPaths: moves.length,
    replacements: Object.fromEntries(rules.replace.map(([f, t], i) => [`${f} -> ${t}`, totals[i]])),
  }, null, 1))

  if (mode === "check" && (changedFiles > 0 || moves.length > 0)) process.exit(1)
}

if (import.meta.main) main()
