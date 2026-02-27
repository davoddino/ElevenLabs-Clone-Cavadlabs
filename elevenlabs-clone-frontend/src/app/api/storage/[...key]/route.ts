import { constants as fsConstants } from "fs";
import { access, mkdir, readFile, writeFile } from "fs/promises";
import path from "path";
import { NextRequest } from "next/server";
import { env } from "~/env";

const DEFAULT_STORAGE_ROOT = path.resolve(process.cwd(), "../local-storage");
const PRIMARY_STORAGE_ROOT = path.resolve(env.LOCAL_STORAGE_ROOT ?? DEFAULT_STORAGE_ROOT);
const FALLBACK_STORAGE_ROOT = path.resolve(
  process.env.LOCAL_STORAGE_FALLBACK_ROOT ?? "/tmp/elevenlabs-local-storage",
);

const STORAGE_ROOTS = Array.from(
  new Set([PRIMARY_STORAGE_ROOT, FALLBACK_STORAGE_ROOT]),
);

const MIME_BY_EXTENSION: Record<string, string> = {
  ".wav": "audio/wav",
  ".mp3": "audio/mpeg",
  ".ogg": "audio/ogg",
  ".webm": "audio/webm",
  ".txt": "text/plain; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

const resolveStoragePath = (storageRoot: string, key: string) => {
  const normalized = key
    .split("/")
    .filter(Boolean)
    .join("/");

  if (!normalized || normalized.includes("..")) {
    throw new Error("Invalid storage key");
  }

  const resolved = path.resolve(storageRoot, normalized);
  if (!(resolved === storageRoot || resolved.startsWith(`${storageRoot}${path.sep}`))) {
    throw new Error("Invalid storage path");
  }

  return resolved;
};

const decodeKey = (segments: string[]) =>
  segments.map((segment) => decodeURIComponent(segment)).join("/");

const notLocalResponse = () =>
  Response.json(
    { error: "Local storage endpoint is disabled when STORAGE_BACKEND=s3" },
    { status: 400 },
  );

const getWritableStorageRoot = async () => {
  for (const root of STORAGE_ROOTS) {
    try {
      await mkdir(root, { recursive: true });
      await access(root, fsConstants.W_OK | fsConstants.X_OK);
      return root;
    } catch {
      // Try next candidate.
    }
  }

  throw new Error(`No writable storage root found. Checked: ${STORAGE_ROOTS.join(", ")}`);
};

const readFromStorageRoots = async (key: string) => {
  for (const root of STORAGE_ROOTS) {
    try {
      const filePath = resolveStoragePath(root, key);
      const file = await readFile(filePath);
      return { file, filePath };
    } catch {
      // Try next candidate.
    }
  }

  throw new Error("File not found in any storage root");
};

export async function GET(
  request: NextRequest,
  { params }: { params: { key: string[] } },
) {
  if (env.STORAGE_BACKEND !== "local") {
    return notLocalResponse();
  }

  try {
    const awaitedParams = await params;
    const key = decodeKey(awaitedParams.key ?? []);
    const { file, filePath } = await readFromStorageRoots(key);

    const contentType =
      MIME_BY_EXTENSION[path.extname(filePath).toLowerCase()] ??
      "application/octet-stream";

    return new Response(file, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch {
    return Response.json({ error: "File not found" }, { status: 404 });
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { key: string[] } },
) {
  if (env.STORAGE_BACKEND !== "local") {
    return notLocalResponse();
  }

  try {
    const awaitedParams = await params;
    const key = decodeKey(awaitedParams.key ?? []);
    const storageRoot = await getWritableStorageRoot();
    const filePath = resolveStoragePath(storageRoot, key);
    const body = Buffer.from(await request.arrayBuffer());

    await mkdir(path.dirname(filePath), { recursive: true });
    await writeFile(filePath, body);

    return Response.json({ ok: true, key });
  } catch (error: any) {
    return Response.json(
      { error: error?.message ?? "Failed to write file" },
      { status: 400 },
    );
  }
}
