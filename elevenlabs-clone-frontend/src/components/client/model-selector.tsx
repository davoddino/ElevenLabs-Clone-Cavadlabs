import { useEffect, useRef, useState } from "react";
import { IoChevronDown, IoChevronUp } from "react-icons/io5";
import { ServiceType } from "~/types/services";
import { useModelStore } from "~/stores/model-store";

export function ModelSelector({ service }: { service: ServiceType }) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const getModelsForService = useModelStore((state) => state.getModelsForService);
  const getSelectedModelForService = useModelStore(
    (state) => state.getSelectedModelForService,
  );
  const selectModelForService = useModelStore((state) => state.selectModelForService);

  const models = getModelsForService(service);
  const selectedModel = getSelectedModelForService(service);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  if (models.length === 0) {
    return null;
  }

  return (
    <div className="relative" ref={dropdownRef}>
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between rounded-xl border border-gray-200 px-3 py-2 hover:cursor-pointer hover:bg-gray-100 hover:bg-opacity-30"
      >
        <span className="text-sm">{selectedModel?.name ?? "No model selected"}</span>
        {isOpen ? (
          <IoChevronUp className="h-4 w-4 text-gray-400" />
        ) : (
          <IoChevronDown className="h-4 w-4 text-gray-400" />
        )}
      </div>

      {isOpen && (
        <div className="absolute left-0 right-0 z-10 mt-1 max-h-60 overflow-auto rounded-lg border border-gray-200 bg-white shadow-lg">
          {models.map((model) => (
            <div
              key={model.id}
              className={`px-3 py-2 text-sm hover:cursor-pointer hover:bg-gray-100 ${model.id === selectedModel?.id ? "bg-gray-50" : ""}`}
              onClick={() => {
                selectModelForService(service, model.id);
                setIsOpen(false);
              }}
            >
              {model.name}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
