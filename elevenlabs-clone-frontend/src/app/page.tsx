import { redirect } from "next/navigation";

export default function HomePage() {
  redirect("/app/speech-synthesis/text-to-speech");
}
