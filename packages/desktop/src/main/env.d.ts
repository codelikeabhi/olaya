interface ImportMetaEnv {
  readonly OLAYA_CHANNEL: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

declare module "virtual:olaya-server" {
  export namespace Server {
    export const listen: typeof import("../../../olaya/dist/types/src/node").Server.listen
    export type Listener = import("../../../olaya/dist/types/src/node").Server.Listener
  }
  export namespace Config {
    export const get: typeof import("../../../olaya/dist/types/src/node").Config.get
    export type Info = import("../../../olaya/dist/types/src/node").Config.Info
  }
  export const bootstrap: typeof import("../../../olaya/dist/types/src/node").bootstrap
}
