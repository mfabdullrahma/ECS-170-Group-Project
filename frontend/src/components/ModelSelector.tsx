"use client";

import { Brain, Layers } from "lucide-react";

type ModelType = "mlp" | "lstm";

interface ModelSelectorProps {
  selectedModel: ModelType;
  onModelChange: (model: ModelType) => void;
}

export function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  return (
    <div className="flex items-center gap-1 p-1 rounded-full border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60 backdrop-blur-xl">
      <button
        onClick={() => onModelChange("mlp")}
        className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm transition-all ${
          selectedModel === "mlp"
            ? "bg-blue-500 text-white shadow-lg shadow-blue-500/30"
            : "hover:bg-black/5 dark:hover:bg-white/5"
        }`}
      >
        <Brain className="w-4 h-4" />
        MLP
      </button>
      <button
        onClick={() => onModelChange("lstm")}
        className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm transition-all ${
          selectedModel === "lstm"
            ? "bg-blue-500 text-white shadow-lg shadow-blue-500/30"
            : "hover:bg-black/5 dark:hover:bg-white/5"
        }`}
      >
        <Layers className="w-4 h-4" />
        LSTM
      </button>
    </div>
  );
}

