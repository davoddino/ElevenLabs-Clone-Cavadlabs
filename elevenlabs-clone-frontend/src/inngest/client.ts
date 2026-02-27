import { Inngest } from "inngest";

const isDev = process.env.NODE_ENV !== "production";

// In local development default to the Inngest dev server to avoid cloud 401s.
const eventKey = process.env.INNGEST_EVENT_KEY ?? (isDev ? "local" : undefined);
const baseUrl = process.env.INNGEST_BASE_URL ?? (isDev ? "http://127.0.0.1:8288" : undefined);

// Create a client to send and receive events
export const inngest = new Inngest({
  id: "elevenlabs-clone",
  ...(eventKey ? { eventKey } : {}),
  ...(baseUrl ? { baseUrl } : {}),
});
