import { randomUUID } from "crypto";
import { GetObjectCommand, PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { env } from "~/env";

const isS3Storage = env.STORAGE_BACKEND === "s3";

const s3Client = isS3Storage
  ? new S3Client({
      region: env.AWS_REGION,
      credentials:
        env.AWS_ACCESS_KEY_ID && env.AWS_SECRET_ACCESS_KEY
          ? {
              accessKeyId: env.AWS_ACCESS_KEY_ID,
              secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
            }
          : undefined,
    })
  : null;

const extensionFromMime = (mimeType: string) => {
  const mimeToExt: Record<string, string> = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
  };

  return mimeToExt[mimeType] ?? "bin";
};

const encodeStorageKey = (key: string) =>
  key
    .split("/")
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join("/");

const validateS3Env = () => {
  if (!env.S3_BUCKET_NAME) {
    throw new Error("S3_BUCKET_NAME is required when STORAGE_BACKEND=s3");
  }

  if (!s3Client) {
    throw new Error("S3 client is not initialized");
  }
};

const getS3Client = () => {
  validateS3Env();
  return s3Client as S3Client;
};

export async function getPresignedUrl({
  key,
  expiresIn = 3600,
}: {
  key: string;
  expiresIn?: number;
}) {
  if (!isS3Storage) {
    return `/api/storage/${encodeStorageKey(key)}`;
  }

  validateS3Env();

  const command = new GetObjectCommand({ Bucket: env.S3_BUCKET_NAME, Key: key });

  return await getSignedUrl(getS3Client(), command, { expiresIn });
}

export async function getUploadUrl(fileType: string) {
  const extension = extensionFromMime(fileType);
  const key = `uploads/${randomUUID()}.${extension}`;

  if (!isS3Storage) {
    return {
      uploadUrl: `/api/storage/${encodeStorageKey(key)}`,
      s3Key: key,
    };
  }

  validateS3Env();

  const command = new PutObjectCommand({
    Bucket: env.S3_BUCKET_NAME,
    Key: key,
    ContentType: fileType,
  });

  const uploadUrl = await getSignedUrl(getS3Client(), command, {
    expiresIn: 3600,
  });

  return { uploadUrl, s3Key: key };
}
