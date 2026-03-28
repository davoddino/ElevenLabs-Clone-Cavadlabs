import { create } from "zustand";

export interface TTSModelOption {
  id: string;
  name: string;
}

const DEFAULT_TTS_MODELS: TTSModelOption[] = [
  {
    id: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    name: "Qwen 0.6B (CustomVoice)",
  },
  {
    id: "mistralai/Voxtral-4B-TTS-2603",
    name: "Voxtral 4B TTS",
  },
];

interface ModelState {
  models: TTSModelOption[];
  selectedModelByService: Record<string, string>;
  getModelsForService: (service: string) => TTSModelOption[];
  getSelectedModelForService: (service: string) => TTSModelOption | null;
  selectModelForService: (service: string, modelId: string) => void;
}

export const useModelStore = create<ModelState>((set, get) => ({
  models: DEFAULT_TTS_MODELS,
  selectedModelByService: {
    "qwen-tts": DEFAULT_TTS_MODELS[0]!.id,
  },
  getModelsForService: (service) => {
    if (service === "qwen-tts") {
      return get().models;
    }
    return [];
  },
  getSelectedModelForService: (service) => {
    const models = get().getModelsForService(service);
    if (models.length === 0) {
      return null;
    }

    const selectedId = get().selectedModelByService[service];
    const selectedModel = models.find((model) => model.id === selectedId);
    return selectedModel ?? models[0]!;
  },
  selectModelForService: (service, modelId) => {
    set((state) => ({
      selectedModelByService: {
        ...state.selectedModelByService,
        [service]: modelId,
      },
    }));
  },
}));
