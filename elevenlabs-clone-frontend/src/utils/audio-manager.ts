class AudioManager {
  private audioElement: HTMLAudioElement | null = null;
  private blobObjectUrl: string | null = null;

  initialize(): HTMLAudioElement | null {
    if (this.audioElement) return this.audioElement;

    if (typeof window !== "undefined") {
      this.audioElement = new Audio();
      return this.audioElement;
    }

    return null;
  }

  getAudio(): HTMLAudioElement | null {
    return this.audioElement;
  }

  getCurrentTime(): number {
    return this.audioElement?.currentTime || 0;
  }

  getDuration(): number {
    return this.audioElement?.duration || 0;
  }

  getProgress(): number {
    if (!this.audioElement || !this.audioElement.duration) return 0;
    return (this.audioElement.currentTime / this.audioElement.duration) * 100;
  }

  setAudioSource(url: string): void {
    if (this.audioElement) {
      this.clearBlobUrl();
      this.audioElement.src = url;
      this.audioElement.load();
    }
  }

  play(): Promise<void> | undefined {
    return this.audioElement?.play();
  }

  private clearBlobUrl(): void {
    if (this.blobObjectUrl) {
      URL.revokeObjectURL(this.blobObjectUrl);
      this.blobObjectUrl = null;
    }
  }

  private async setAudioSourceFromBlob(url: string): Promise<void> {
    if (!this.audioElement) return;

    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to fetch audio: ${response.status} ${response.statusText}`);
    }

    const blob = await response.blob();
    this.clearBlobUrl();
    this.blobObjectUrl = URL.createObjectURL(
      new Blob([blob], { type: "audio/wav" }),
    );
    this.audioElement.src = this.blobObjectUrl;
    this.audioElement.load();
  }

  async playFromSource(url: string): Promise<void> {
    if (!this.audioElement) return;

    this.setAudioSource(url);
    try {
      await this.audioElement.play();
      return;
    } catch (error: any) {
      const isNotSupported = error?.name === "NotSupportedError";
      if (!isNotSupported) {
        throw error;
      }
    }

    await this.setAudioSourceFromBlob(url);
    await this.audioElement.play();
  }

  pause(): void {
    return this.audioElement?.pause();
  }

  skipForward(seconds: number = 10): void {
    if (this.audioElement) {
      this.audioElement.currentTime = Math.min(
        this.audioElement.duration || 0,
        this.audioElement.currentTime + seconds,
      );
    }
  }

  skipBackward(seconds: number = 10): void {
    if (this.audioElement) {
      this.audioElement.currentTime = Math.max(
        0,
        this.audioElement.currentTime - seconds,
      );
    }
  }
}

export const audioManager = new AudioManager();
