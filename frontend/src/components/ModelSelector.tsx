"use client";

import { Brain, Database } from "lucide-react";

export type ModelType = "mlp" | "kaggle-mlp";

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
        MLP Model
      </button>
      <button
        onClick={() => onModelChange("kaggle-mlp")}
        className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm transition-all ${
          selectedModel === "kaggle-mlp"
            ? "bg-green-500 text-white shadow-lg shadow-green-500/30"
            : "hover:bg-black/5 dark:hover:bg-white/5"
        }`}
      >
        <Database className="w-4 h-4" />
        Kaggle MLP Model
      </button>
    </div>
  );
}
