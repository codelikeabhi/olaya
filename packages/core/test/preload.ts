import path from "path"

process.env.OLAYA_DB = ":memory:"
process.env.NPM_CONFIG_AUDIT = "false"
process.env.OLAYA_MODELS_PATH = path.join(import.meta.dir, "plugin", "fixtures", "models-dev.json")
process.env.OLAYA_DISABLE_MODELS_FETCH = "true"
