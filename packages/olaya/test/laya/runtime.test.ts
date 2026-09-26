import { expect, test } from "bun:test"
import os from "os"
import path from "path"
import { launch, python, setupPlan, sourceDir, venvPython } from "../../src/laya/runtime"

test("an explicit interpreter wins; otherwise the managed venv, if it exists; otherwise python3", () => {
  expect(python("/opt/py/bin/python", "/nonexistent/venv/bin/python")).toBe("/opt/py/bin/python")
  expect(python(undefined, "/nonexistent/venv/bin/python")).toBe("python3")
  expect(python(undefined, process.execPath)).toBe(process.execPath)
})

test("a compiled binary has no source tree, so the installed module runs isolated from a neutral cwd", () => {
  expect(sourceDir("/$bunfs/root")).toBeUndefined()
  const l = launch("/venv/bin/python", {}, undefined)
  expect(l.argv).toEqual(["/venv/bin/python", "-I", "-m", "service"])
  expect(l.cwd).toBe(os.tmpdir())
})

test("a source checkout, or OLAYA_LAYA_DIR, runs service.py in place", () => {
  expect(sourceDir(path.join(import.meta.dir, "../../src/laya"))).toMatch(/laya$/)
  expect(launch("py", {}, "/repo/laya")).toEqual({ argv: ["py", "service.py"], cwd: "/repo/laya" })
  expect(launch("py", { OLAYA_LAYA_DIR: "/mine" }, "/repo/laya")).toEqual({ argv: ["py", "service.py"], cwd: "/mine" })
})

test("setup prefers uv, installs a release's own tag, and falls back to venv + pip", () => {
  const venv = "/data/laya/venv"
  const uv = setupPlan({ uv: "/bin/uv", python3: "/bin/python3", venv, version: "0.2.0" })
  expect(uv).toEqual([
    ["/bin/uv", "venv", "--python", "3.12", venv],
    ["/bin/uv", "pip", "install", "--python", venvPython(venv), "--torch-backend", "auto",
      "olaya-laya @ git+https://github.com/codelikeabhi/olaya@v0.2.0#subdirectory=laya"],
  ])
  const pip = setupPlan({ python3: "/bin/python3", venv, version: "0.1.0-dev", source: "/repo/laya" })
  expect(pip).toEqual([
    ["/bin/python3", "-m", "venv", venv],
    [venvPython(venv), "-m", "pip", "install", "--upgrade", "pip"],
    [venvPython(venv), "-m", "pip", "install", "-e", "/repo/laya"],
  ])
  expect(setupPlan({ venv, version: "0.1.0" })).toHaveProperty("error")
  expect(venvPython("C:\\v", "win32")).toContain("Scripts")
})
