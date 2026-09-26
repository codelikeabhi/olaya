import path from "path"

process.env.OLAYA_DB = ":memory:"
process.env.NPM_CONFIG_AUDIT = "false"
process.env.OLAYA_MODELS_PATH = path.join(import.meta.dir, "plugin", "fixtures", "models-dev.json")
process.env.OLAYA_DISABLE_MODELS_FETCH = "true"
// Fixture commits must not depend on the developer's git config: with commit signing on, every
// `git commit` in a fixture waits on a key passphrase until the test times out.
process.env.GIT_CONFIG_COUNT = "1"
process.env.GIT_CONFIG_KEY_0 = "commit.gpgsign"
process.env.GIT_CONFIG_VALUE_0 = "false"
