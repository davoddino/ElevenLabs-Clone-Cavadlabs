import { constants as fsConstants } from "fs";
import { access, mkdir, readFile, writeFile } from "fs/promises";
import path from "path";
import { NextRequest } from "next/server";
import { env } from "~/env";

const STORAGE_ROOT = env.LOCAL_STORAGE_ROOT
  ? path.resolve(env.LOCAL_STORAGE_ROOT)
  : null;

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

const missingStorageRootResponse = () =>
  Response.json(
    {
      error:
        "LOCAL_STORAGE_ROOT is required in frontend env when STORAGE_BACKEND=local",
    },
    { status: 500 },
  );

const invalidStorageRootResponse = (message: string) =>
  Response.json(
    {
      error: "LOCAL_STORAGE_ROOT is not accessible",
      storageRoot: STORAGE_ROOT,
      detail: message,
    },
    { status: 500 },
  );

const ensureStorageRootAccessible = async () => {
  if (!STORAGE_ROOT) {
    throw new Error("LOCAL_STORAGE_ROOT is missing");
  }
  await access(STORAGE_ROOT, fsConstants.R_OK | fsConstants.X_OK);
};

export async function GET(
  request: NextRequest,
  { params }: { params: { key: string[] } },
) {
  if (env.STORAGE_BACKEND !== "local") {
    return notLocalResponse();
  }
  if (!STORAGE_ROOT) {
    return missingStorageRootResponse();
  }
  try {
    await ensureStorageRootAccessible();
  } catch (error: any) {
    return invalidStorageRootResponse(error?.message ?? "unknown error");
  }

  try {
    const awaitedParams = await params;
    const key = decodeKey(awaitedParams.key ?? []);
    const filePath = resolveStoragePath(STORAGE_ROOT, key);
    const file = await readFile(filePath);
    const contentType =
      MIME_BY_EXTENSION[path.extname(filePath).toLowerCase()] ??
      "application/octet-stream";

    return new Response(file, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch (error: any) {
    const code = error?.code;
    if (code === "ENOENT") {
      const awaitedParams = await params;
      const key = decodeKey(awaitedParams.key ?? []);
      const filePath = resolveStoragePath(STORAGE_ROOT, key);
      console.warn("Local storage file not found", {
        key,
        filePath,
        storageRoot: STORAGE_ROOT,
      });
      return Response.json(
        { error: "File not found", key, filePath, storageRoot: STORAGE_ROOT },
        { status: 404 },
      );
    }

    console.error("Failed to read local storage file", {
      storageRoot: STORAGE_ROOT,
      message: error?.message,
      code,
    });

    return Response.json(
      {
        error: "Failed to read local file",
        detail: error?.message ?? "unknown error",
      },
      { status: 500 },
    );
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { key: string[] } },
) {
  if (env.STORAGE_BACKEND !== "local") {
    return notLocalResponse();
  }
  if (!STORAGE_ROOT) {
    return missingStorageRootResponse();
  }
  try {
    // Ensure parent root is reachable before writes.
    await access(path.dirname(STORAGE_ROOT), fsConstants.W_OK | fsConstants.X_OK);
  } catch (error: any) {
    return invalidStorageRootResponse(error?.message ?? "unknown error");
  }

  try {
    const awaitedParams = await params;
    const key = decodeKey(awaitedParams.key ?? []);
    const filePath = resolveStoragePath(STORAGE_ROOT, key);
    const body = Buffer.from(await request.arrayBuffer());

    await mkdir(path.dirname(filePath), { recursive: true });
    await writeFile(filePath, body);

    return Response.json({ ok: true, key });
  } catch (error: any) {
    console.error("Failed to write local storage file", {
      storageRoot: STORAGE_ROOT,
      message: error?.message,
      code: error?.code,
    });
    return Response.json(
      {
        error: "Failed to write file",
        detail: error?.message ?? "unknown error",
      },
      { status: 400 },
    );
  }
}
