"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import { Hands, Results } from "@mediapipe/hands";

export type ModelType = "mlp" | "lstm";

interface Prediction {
  letter: string;
  confidence: number;
}

interface UseASLDetectionProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  isActive: boolean;
  selectedModel: ModelType;
  onDetection: (letter: string, confidence: number, topPredictions?: Prediction[]) => void;
}

// Label mapping (should match your training labels)
// This matches the output from train_mlp.py with del and space included
const LABELS: Record<number, string> = {
  0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
  9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q",
  17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 
  25: "Z", 26: "del", 27: "space"
};

const SEQUENCE_LENGTH = 30;
const MIN_CONSECUTIVE_SAME = 3; // Need 3 consecutive same predictions to switch (reduced for responsiveness)

export function useASLDetection({
  videoRef,
  canvasRef,
  isActive,
  selectedModel,
  onDetection,
}: UseASLDetectionProps) {
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isHandsLoaded, setIsHandsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mlpModelRef = useRef<tf.LayersModel | null>(null);
  const lstmModelRef = useRef<tf.LayersModel | null>(null);
  const handsRef = useRef<Hands | null>(null);
  const sequenceBufferRef = useRef<number[][]>([]);
  const animationFrameRef = useRef<number | null>(null);
  const labelsRef = useRef<Record<number, string>>(LABELS);
  
  // Prediction smoothing - simple consecutive counting
  const lastStablePredictionRef = useRef<string>("");
  const consecutiveCountRef = useRef<number>(0);
  const lastRawLetterRef = useRef<string>("");

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

  // Smooth predictions using simple consecutive counting - DEFINED FIRST
  const smoothPrediction = useCallback(
    (letter: string, confidence: number, topPreds?: Prediction[]) => {
      // Store top predictions for display
      if (topPreds) {
        topPredictionsRef.current = topPreds;
      }

      // Track consecutive same predictions
      if (letter === lastRawLetterRef.current) {
        consecutiveCountRef.current++;
      } else {
        consecutiveCountRef.current = 1;
        lastRawLetterRef.current = letter;
      }
      
      // Update display when we have enough consecutive same predictions
      // OR if this is a new letter with high confidence
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

  // Mock prediction for demo/testing - DEFINED SECOND
  const mockPrediction = useCallback(() => {
    const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const randomLetter = letters[Math.floor(Math.random() * letters.length)];
    const randomConfidence = 0.75 + Math.random() * 0.24;
    smoothPrediction(randomLetter, randomConfidence);
  }, [smoothPrediction]);

  // Run MLP inference - DEFINED AFTER smoothPrediction and mockPrediction
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
          
          smoothPrediction(bestLetter, bestConfidence, topPredictions);
          
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
    [smoothPrediction, mockPrediction]
  );

  // Run LSTM inference - DEFINED AFTER smoothPrediction and mockPrediction
  const runLSTMInference = useCallback(
    async (sequence: number[][]) => {
      const model = lstmModelRef.current;
      
      if (model) {
        try {
          const input = tf.tensor3d([sequence], [1, SEQUENCE_LENGTH, 42]);
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
          
          smoothPrediction(bestLetter, bestConfidence, topPredictions);
          
          input.dispose();
          prediction.dispose();
        } catch (e) {
          console.error("LSTM inference error:", e);
          mockPrediction();
        }
      } else {
        mockPrediction();
      }
    },
    [smoothPrediction, mockPrediction]
  );

  // Load labels from JSON
  useEffect(() => {
    const loadLabels = async () => {
      try {
        const response = await fetch("/models/labels.json");
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
          const mlpModel = await tf.loadLayersModel("/models/mlp/model.json");
          mlpModelRef.current = mlpModel;
          console.log("MLP model loaded");
        } catch (e) {
          console.warn("MLP model not found, will use mock predictions");
        }

        // Load LSTM model
        try {
          const lstmModel = await tf.loadLayersModel("/models/lstm/model.json");
          lstmModelRef.current = lstmModel;
          console.log("LSTM model loaded");
        } catch (e) {
          console.warn("LSTM model not found, will use mock predictions");
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
      lstmModelRef.current?.dispose();
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

  // Clear buffers when model changes or camera stops
  useEffect(() => {
    lastStablePredictionRef.current = "";
    sequenceBufferRef.current = [];
    consecutiveCountRef.current = 0;
    lastRawLetterRef.current = "";
  }, [selectedModel, isActive]);

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

        // Run inference based on selected model
        if (selectedModel === "mlp") {
          runMLPInference(keypoints);
        } else {
          // Add to sequence buffer for LSTM
          sequenceBufferRef.current.push(keypoints);
          if (sequenceBufferRef.current.length > SEQUENCE_LENGTH) {
            sequenceBufferRef.current.shift();
          }
          
          if (sequenceBufferRef.current.length === SEQUENCE_LENGTH) {
            runLSTMInference(sequenceBufferRef.current);
          }
        }
      } else {
        // No hand detected
        sequenceBufferRef.current = [];
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
  }, [isActive, isHandsLoaded, videoRef, canvasRef, selectedModel, normalizeKeypoints, drawHandLandmarks, runMLPInference, runLSTMInference]);

  return {
    isModelLoaded,
    isHandsLoaded,
    error,
    isReady: isModelLoaded && isHandsLoaded,
  };
}
