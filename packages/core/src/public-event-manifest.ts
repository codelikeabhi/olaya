export * as PublicEventManifest from "./public-event-manifest"

import { Event } from "@olaya/schema/event"
import { EventManifest } from "@olaya/schema/event-manifest"

export const Definitions = EventManifest.ServerDefinitions
export const Latest = Event.latest(Definitions)
