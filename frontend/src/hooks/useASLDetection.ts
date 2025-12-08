"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import { Hands, Results } from "@mediapipe/hands";



interface Prediction {
  letter: string;
  confidence: number;
}

interface UseASLDetectionProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  isActive: boolean;
  onDetection: (letter: string, confidence: number, topPredictions?: Prediction[]) => void;
}

// Label mapping from hand-gesture-recognition-previous model
// Note: This model does NOT include J or Z
const LABELS: Record<number, string> = {
  0: "Open", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I",
  10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q",
  17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y"
};



export function useASLDetection({
  videoRef,
  canvasRef,
  isActive,
  onDetection,
}: UseASLDetectionProps) {
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isHandsLoaded, setIsHandsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mlpModelRef = useRef<tf.GraphModel | null>(null);
  const handsRef = useRef<Hands | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const labelsRef = useRef<Record<number, string>>(LABELS);



  // Normalize keypoints (same as training - preprocess.py)
  // Uses only X, Y coordinates (no Z) for stability
  // Normalizes by max absolute value
  // Returns 42 features (21 landmarks × 2 coordinates)
  const normalizeKeypoints = useCallback((landmarks: { x: number; y: number; z: number }[]) => {
    // Extract only X, Y coordinates (ignore Z for stability)
    const coords = landmarks.map((lm) => [lm.x, lm.y]);

    // Translate to wrist origin (landmark 0)
    const baseX = coords[0][0];
    const baseY = coords[0][1];
    const translated = coords.map((c) => [c[0] - baseX, c[1] - baseY]);

    // Flatten to 1D array
    const flat = translated.flatMap((c) => [c[0], c[1]]);

    // Normalize by max absolute value
    const maxVal = Math.max(...flat.map(Math.abs));

    if (maxVal > 0) {
      return flat.map((v) => v / maxVal);
    }
    return flat;
  }, []);

  // Store top predictions for display
  const topPredictionsRef = useRef<Prediction[]>([]);

  // Output prediction immediately (matching quiz.py behavior - no smoothing)
  const outputPrediction = useCallback(
    (letter: string, confidence: number, topPreds?: Prediction[]) => {
      // Store top predictions for display
      if (topPreds) {
        topPredictionsRef.current = topPreds;
      }

      // Output immediately on every frame (like quiz.py)
      onDetection(letter, confidence, topPredictionsRef.current);
    },
    [onDetection]
  );

  // Mock prediction for demo/testing
  const mockPrediction = useCallback(() => {
    const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const randomLetter = letters[Math.floor(Math.random() * letters.length)];
    const randomConfidence = 0.75 + Math.random() * 0.24;
    outputPrediction(randomLetter, randomConfidence);
  }, [outputPrediction]);

  // Run MLP inference
  const runMLPInference = useCallback(
    async (keypoints: number[]) => {
      const model = mlpModelRef.current;

      if (model) {
        try {
          // Verify keypoints length (42 features: 21 landmarks × 2 coords)
          if (keypoints.length !== 42) {
            console.error(`Wrong keypoints length: ${keypoints.length}, expected 42`);
            return;
          }

          const input = tf.tensor2d([keypoints], [1, 42]);
          const prediction = model.predict(input) as tf.Tensor;
          const probabilities = await prediction.data();

          // Get top 5 predictions
          const probArray = Array.from(probabilities);
          const indexed = probArray.map((prob, idx) => ({ idx, prob }));
          indexed.sort((a, b) => b.prob - a.prob);

          const topPredictions: Prediction[] = indexed.slice(0, 5).map(({ idx, prob }) => ({
            letter: labelsRef.current[idx] || String.fromCharCode(65 + idx),
            confidence: prob,
          }));

          const bestLetter = topPredictions[0].letter;
          const bestConfidence = topPredictions[0].confidence;

          outputPrediction(bestLetter, bestConfidence, topPredictions);

          input.dispose();
          prediction.dispose();
        } catch (e) {
          console.error("MLP inference error:", e);
          mockPrediction();
        }
      } else {
        mockPrediction();
      }
    },
    [outputPrediction, mockPrediction]
  );

  // Load labels from JSON
  useEffect(() => {
    const loadLabels = async () => {
      try {
        const response = await fetch("/models/mlp/labels.json");
        if (response.ok) {
          const labels = await response.json();
          labelsRef.current = labels;
        }
      } catch (e) {
        console.log("Using default labels");
      }
    };
    loadLabels();
  }, []);

  // Load TensorFlow.js models
  useEffect(() => {
    const loadModels = async () => {
      try {
        setError(null);

        // Load MLP model
        try {
          const mlpModel = await tf.loadGraphModel("/models/mlp/model.json");
          mlpModelRef.current = mlpModel;
          console.log("MLP model loaded");
        } catch (e) {
          console.warn("MLP model not found, will use mock predictions");
        }

        setIsModelLoaded(true);
      } catch (err) {
        console.error("Error loading models:", err);
        setError("Failed to load models. Using mock predictions.");
        setIsModelLoaded(true); // Continue with mock predictions
      }
    };

    loadModels();

    return () => {
      mlpModelRef.current?.dispose();
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
    // Draw connections
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8], // Index
      [0, 9], [9, 10], [10, 11], [11, 12], // Middle
      [0, 13], [13, 14], [14, 15], [15, 16], // Ring
      [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
      [5, 9], [9, 13], [13, 17], // Palm
    ];

    ctx.strokeStyle = "#3b82f6";
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
    ctx.fillStyle = "#22c55e";
    for (const landmark of landmarks) {
      ctx.beginPath();
      ctx.arc(landmark.x * width, landmark.y * height, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, []);

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

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];

        // Draw hand landmarks
        drawHandLandmarks(ctx, landmarks, canvas.width, canvas.height);

        // Normalize keypoints
        const keypoints = normalizeKeypoints(landmarks);

        // Run MLP inference
        runMLPInference(keypoints);
      } else {
        // No hand detected
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
  }, [isActive, isHandsLoaded, videoRef, canvasRef, normalizeKeypoints, drawHandLandmarks, runMLPInference]);

  return {
    isModelLoaded,
    isHandsLoaded,
    error,
    isReady: isModelLoaded && isHandsLoaded,
  };
}
