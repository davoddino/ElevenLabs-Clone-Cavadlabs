import { PageLayout } from "~/components/client/page-layout";
import { getHistoryItems } from "~/lib/history";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { env } from "~/env";
import { TextToSpeechEditor } from "../../../../components/client/speech-synthesis/text-to-speech-editor";

export default async function TextToSpeechPage() {
  const session = await auth();
  const userId = session?.user?.id;

  let credits = 0;

  if (userId) {
    const user = await db.user.findUnique({
      where: { id: userId },
      select: {
        credits: true,
      },
    });
    credits = user?.credits ?? 0;
  }

  const service = env.TTS_DEFAULT_SERVICE;

  const historyItems = await getHistoryItems(service);

  return (
    <PageLayout
      title={"Text to Speech"}
      service={service}
      showSidebar={true}
      historyItems={historyItems}
    >
      <TextToSpeechEditor service={service} credits={credits} />
    </PageLayout>
  );
}
