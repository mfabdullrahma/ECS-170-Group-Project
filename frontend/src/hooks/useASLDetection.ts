"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import { Hands, Results } from "@mediapipe/hands";
import { ModelType } from "@/components/ModelSelector";

interface Prediction {
  letter: string;
  confidence: number;
}

interface UseASLDetectionProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  isActive: boolean;
  modelType: ModelType;
  onDetection: (letter: string, confidence: number, topPredictions?: Prediction[]) => void;
}

// Label mapping for MLP model (from hand-gesture-recognition-previous)
// Note: This model does NOT include J or Z
const MLP_LABELS: Record<number, string> = {
  0: "Open", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I",
  10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q",
  17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y"
};

// Label mapping for Kaggle MLP model (A-Z + del + space)
const KAGGLE_LABELS: Record<number, string> = {
  0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
  9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q",
  17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y",
  25: "Z", 26: "del", 27: "space"
};

const MIN_CONSECUTIVE_SAME = 3;

export function useASLDetection({
  videoRef,
  canvasRef,
  isActive,
  modelType,
  onDetection,
}: UseASLDetectionProps) {
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isHandsLoaded, setIsHandsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadedModelType, setLoadedModelType] = useState<ModelType | null>(null);

  const mlpModelRef = useRef<tf.GraphModel | null>(null);
  const kaggleModelRef = useRef<tf.LayersModel | null>(null);
  const handsRef = useRef<Hands | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const mlpLabelsRef = useRef<Record<number, string>>(MLP_LABELS);
  const kaggleLabelsRef = useRef<Record<number, string>>(KAGGLE_LABELS);

  // Prediction smoothing for Kaggle model
  const lastStablePredictionRef = useRef<string>("");
  const consecutiveCountRef = useRef<number>(0);
  const lastRawLetterRef = useRef<string>("");
  const topPredictionsRef = useRef<Prediction[]>([]);

  // Normalize keypoints (same as training - preprocess.py)
  const normalizeKeypoints = useCallback((landmarks: { x: number; y: number; z: number }[]) => {
    const coords = landmarks.map((lm) => [lm.x, lm.y]);
    const baseX = coords[0][0];
    const baseY = coords[0][1];
    const translated = coords.map((c) => [c[0] - baseX, c[1] - baseY]);
    const flat = translated.flatMap((c) => [c[0], c[1]]);
    const maxVal = Math.max(...flat.map(Math.abs));

    if (maxVal > 0) {
      return flat.map((v) => v / maxVal);
    }
    return flat;
  }, []);

  // Smooth predictions for Kaggle model (uses consecutive-frame voting)
  const smoothPrediction = useCallback(
    (letter: string, confidence: number, topPreds?: Prediction[]) => {
      if (topPreds) {
        topPredictionsRef.current = topPreds;
      }

      if (letter === lastRawLetterRef.current) {
        consecutiveCountRef.current++;
      } else {
        consecutiveCountRef.current = 1;
        lastRawLetterRef.current = letter;
      }

      const shouldUpdate =
        consecutiveCountRef.current >= MIN_CONSECUTIVE_SAME ||
        (confidence > 0.7 && letter !== lastStablePredictionRef.current);

      if (shouldUpdate && letter !== lastStablePredictionRef.current) {
        lastStablePredictionRef.current = letter;
        onDetection(letter, confidence, topPredictionsRef.current);
      }
    },
    [onDetection]
  );

  // Output prediction immediately for MLP model (no smoothing)
  const outputPrediction = useCallback(
    (letter: string, confidence: number, topPreds?: Prediction[]) => {
      if (topPreds) {
        topPredictionsRef.current = topPreds;
      }
      onDetection(letter, confidence, topPredictionsRef.current);
    },
    [onDetection]
  );

  // Run MLP inference (graph model from hand-gesture-recognition)
  const runMLPInference = useCallback(
    async (keypoints: number[]) => {
      const model = mlpModelRef.current;

      if (model) {
        try {
          if (keypoints.length !== 42) {
            console.error(`Wrong keypoints length: ${keypoints.length}, expected 42`);
            return;
          }

          const input = tf.tensor2d([keypoints], [1, 42]);
          const prediction = model.predict(input) as tf.Tensor;
          const probabilities = await prediction.data();

          const probArray = Array.from(probabilities);
          const indexed = probArray.map((prob, idx) => ({ idx, prob }));
          indexed.sort((a, b) => b.prob - a.prob);

          const topPredictions: Prediction[] = indexed.slice(0, 5).map(({ idx, prob }) => ({
            letter: mlpLabelsRef.current[idx] || String.fromCharCode(65 + idx),
            confidence: prob,
          }));

          const bestLetter = topPredictions[0].letter;
          const bestConfidence = topPredictions[0].confidence;

          outputPrediction(bestLetter, bestConfidence, topPredictions);

          input.dispose();
          prediction.dispose();
        } catch (e) {
          console.error("MLP inference error:", e);
        }
      }
    },
    [outputPrediction]
  );

  // Run Kaggle MLP inference (layers model)
  const runKaggleMLPInference = useCallback(
    async (keypoints: number[]) => {
      const model = kaggleModelRef.current;

      if (model) {
        try {
          if (keypoints.length !== 42) {
            console.error(`Wrong keypoints length: ${keypoints.length}, expected 42`);
            return;
          }

          const input = tf.tensor2d([keypoints], [1, 42]);
          const prediction = model.predict(input) as tf.Tensor;
          const probabilities = await prediction.data();

          const probArray = Array.from(probabilities);
          const indexed = probArray.map((prob, idx) => ({ idx, prob }));
          indexed.sort((a, b) => b.prob - a.prob);

          const topPredictions: Prediction[] = indexed.slice(0, 5).map(({ idx, prob }) => ({
            letter: kaggleLabelsRef.current[idx] || String.fromCharCode(65 + idx),
            confidence: prob,
          }));

          // Use smoothing for Kaggle model
          smoothPrediction(topPredictions[0].letter, topPredictions[0].confidence, topPredictions);

          input.dispose();
          prediction.dispose();
        } catch (e) {
          console.error("Kaggle MLP inference error:", e);
        }
      }
    },
    [smoothPrediction]
  );

  // Load labels from JSON files
  useEffect(() => {
    const loadLabels = async () => {
      try {
        // Load MLP labels
        const mlpResponse = await fetch("/models/mlp/labels.json");
        if (mlpResponse.ok) {
          const labels = await mlpResponse.json();
          mlpLabelsRef.current = labels;
        }
      } catch (e) {
        console.log("Using default MLP labels");
      }

      try {
        // Load Kaggle labels
        const kaggleResponse = await fetch("/models/kaggle-mlp/labels.json");
        if (kaggleResponse.ok) {
          const rawLabels = await kaggleResponse.json();
          const labels: Record<number, string> = {};
          for (const [key, value] of Object.entries(rawLabels)) {
            labels[parseInt(key)] = value as string;
          }
          kaggleLabelsRef.current = labels;
        }
      } catch (e) {
        console.log("Using default Kaggle labels");
      }
    };
    loadLabels();
  }, []);

  // Load TensorFlow.js models based on modelType
  useEffect(() => {
    const loadModels = async () => {
      try {
        setError(null);
        setIsModelLoaded(false);

        if (modelType === "mlp") {
          // Load MLP model (GraphModel from TFLite)
          if (!mlpModelRef.current) {
            try {
              const mlpModel = await tf.loadGraphModel("/models/mlp/model.json");
              mlpModelRef.current = mlpModel;
              console.log("MLP model loaded");
            } catch (e) {
              console.error("MLP model load error:", e);
              setError("Failed to load MLP model");
            }
          }
        } else if (modelType === "kaggle-mlp") {
          // Load Kaggle MLP model (LayersModel from Keras)
          if (!kaggleModelRef.current) {
            try {
              const kaggleModel = await tf.loadLayersModel("/models/kaggle-mlp/model.json");
              kaggleModelRef.current = kaggleModel;
              console.log("Kaggle MLP model loaded");
              console.log("Input shape:", kaggleModel.inputs[0].shape);
              console.log("Output shape:", kaggleModel.outputs[0].shape);
            } catch (e) {
              console.error("Kaggle MLP model load error:", e);
              setError("Failed to load Kaggle MLP model");
            }
          }
        }

        setLoadedModelType(modelType);
        setIsModelLoaded(true);
      } catch (err) {
        console.error("Error loading models:", err);
        setError("Failed to load models");
        setIsModelLoaded(true);
      }
    };

    loadModels();
  }, [modelType]);

  // Cleanup models on unmount
  useEffect(() => {
    return () => {
      mlpModelRef.current?.dispose();
      kaggleModelRef.current?.dispose();
    };
  }, []);

  // Initialize MediaPipe Hands
  useEffect(() => {
    const initHands = async () => {
      try {
        const hands = new Hands({
          locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
          },
        });

        hands.setOptions({
          maxNumHands: 1,
          modelComplexity: 1,
          minDetectionConfidence: 0.7,
          minTrackingConfidence: 0.5,
        });

        handsRef.current = hands;
        setIsHandsLoaded(true);
        console.log("MediaPipe Hands initialized");
      } catch (err) {
        console.error("Error initializing MediaPipe Hands:", err);
        setError("Failed to initialize hand detection");
      }
    };

    initHands();

    return () => {
      handsRef.current?.close();
    };
  }, []);

  // Draw hand landmarks on canvas
  const drawHandLandmarks = useCallback((
    ctx: CanvasRenderingContext2D,
    landmarks: { x: number; y: number; z: number }[],
    width: number,
    height: number
  ) => {
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8], // Index
      [0, 9], [9, 10], [10, 11], [11, 12], // Middle
      [0, 13], [13, 14], [14, 15], [15, 16], // Ring
      [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
      [5, 9], [9, 13], [13, 17], // Palm
    ];

    // Use green for Kaggle model, blue for MLP model
    const strokeColor = modelType === "kaggle-mlp" ? "#22c55e" : "#3b82f6";
    
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 3;

    for (const [i, j] of connections) {
      const start = landmarks[i];
      const end = landmarks[j];
      ctx.beginPath();
      ctx.moveTo(start.x * width, start.y * height);
      ctx.lineTo(end.x * width, end.y * height);
      ctx.stroke();
    }

    // Draw landmarks
    ctx.fillStyle = modelType === "kaggle-mlp" ? "#16a34a" : "#22c55e";
    for (const landmark of landmarks) {
      ctx.beginPath();
      ctx.arc(landmark.x * width, landmark.y * height, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, [modelType]);

  // Clear smoothing state when model changes
  useEffect(() => {
    lastStablePredictionRef.current = "";
    consecutiveCountRef.current = 0;
    lastRawLetterRef.current = "";
  }, [modelType]);

  // Clear state when camera stops
  useEffect(() => {
    if (!isActive) {
      lastStablePredictionRef.current = "";
      consecutiveCountRef.current = 0;
      lastRawLetterRef.current = "";
    }
  }, [isActive]);

  // Process video frames
  useEffect(() => {
    if (!isActive || !isHandsLoaded || !videoRef.current || !handsRef.current) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      return;
    }

    const hands = handsRef.current;

    // Set up the results handler
    hands.onResults((results: Results) => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");

      if (!canvas || !ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];

        drawHandLandmarks(ctx, landmarks, canvas.width, canvas.height);

        const keypoints = normalizeKeypoints(landmarks);

        // Run appropriate inference based on model type
        if (modelType === "mlp" && mlpModelRef.current) {
          runMLPInference(keypoints);
        } else if (modelType === "kaggle-mlp" && kaggleModelRef.current) {
          runKaggleMLPInference(keypoints);
        }
      }
    });

    const processFrame = async () => {
      if (videoRef.current && videoRef.current.readyState >= 2) {
        await hands.send({ image: videoRef.current });
      }
      animationFrameRef.current = requestAnimationFrame(processFrame);
    };

    processFrame();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isActive, isHandsLoaded, modelType, videoRef, canvasRef, normalizeKeypoints, drawHandLandmarks, runMLPInference, runKaggleMLPInference]);

  return {
    isModelLoaded,
    isHandsLoaded,
    error,
    isReady: isModelLoaded && isHandsLoaded && loadedModelType === modelType,
  };
}
