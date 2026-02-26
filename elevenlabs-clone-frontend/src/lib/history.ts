import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { ServiceType } from "~/types/services";
import { getPresignedUrl } from "./s3";

export interface HistoryItem {
  id: string;
  title: string;
  voice: string | null;
  audioUrl: string | null;
  service: ServiceType;
  date: string;
  time: string;
}

const isServiceType = (service: string): service is ServiceType => {
  return ["styletts2", "qwen-tts", "seedvc", "make-an-audio"].includes(service);
};

const titleFromClip = (text: string | null, fallback: string) => {
  if (!text || text.trim().length === 0) {
    return fallback;
  }

  return text.length > 80 ? `${text.slice(0, 80)}...` : text;
};

export async function getHistoryItems(service: ServiceType): Promise<HistoryItem[]> {
  const session = await auth();
  const userId = session?.user?.id;
  if (!userId) {
    return [] as HistoryItem[];
  }

  const clips = await db.generatedAudioClip.findMany({
    where: {
      userId: userId,
      service,
      failed: false,
      s3Key: {
        not: null,
      },
    },
    orderBy: {
      createdAt: "desc",
    },
    take: 100,
    select: {
      id: true,
      text: true,
      voice: true,
      s3Key: true,
      createdAt: true,
      service: true,
      originalVoiceS3Key: true,
    },
  });

  const historyItems = await Promise.all(
    clips
      .filter((clip) => isServiceType(clip.service))
      .map(async (clip) => {
        const fallbackTitle =
          clip.service === "seedvc" ? "Voice conversion" : "Generated audio";

        return {
          id: clip.id,
          title: titleFromClip(clip.text, fallbackTitle),
          voice: clip.voice,
          audioUrl: clip.s3Key
            ? await getPresignedUrl({ key: clip.s3Key })
            : null,
          service: clip.service as ServiceType,
          date: clip.createdAt.toLocaleDateString(),
          time: clip.createdAt.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
        };
      }),
  );

  return historyItems;
}
