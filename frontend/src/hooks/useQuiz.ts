"use client";

import { useState, useCallback, useRef, useEffect } from "react";

export type QuizState = "setup" | "ready" | "active" | "feedback" | "results";

export interface QuizQuestion {
    targetLetter: string;
    questionNumber: number;
    totalQuestions: number;
}

export interface QuizResult {
    targetLetter: string;
    isCorrect: boolean;
    timeElapsed: number;
}

interface UseQuizReturn {
    // State
    quizState: QuizState;
    currentQuestion: QuizQuestion | null;
    score: number;
    results: QuizResult[];
    timeRemaining: number;
    stabilityCount: number;
    lastFeedback: { isCorrect: boolean; letter: string } | null;

    // Actions
    startQuiz: (numQuestions: number) => void;
    nextQuestion: () => void;
    checkAnswer: (detectedLetter: string) => void;
    resetStability: () => void;
    resetQuiz: () => void;
    skipToResults: () => void;
}

const STABLE_THRESHOLD = 15; // Number of consecutive frames required for correct answer (0.5s at 30fps)
const QUESTION_TIME_LIMIT = 10; // Seconds per question

// Available letters (A-Y, excluding J and Z - not in hand-gesture-recognition-previous model)
const AVAILABLE_LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("");

export function useQuiz(): UseQuizReturn {
    const [quizState, setQuizState] = useState<QuizState>("setup");
    const [currentQuestion, setCurrentQuestion] = useState<QuizQuestion | null>(null);
    const [score, setScore] = useState(0);
    const [results, setResults] = useState<QuizResult[]>([]);
    const [timeRemaining, setTimeRemaining] = useState(QUESTION_TIME_LIMIT);
    const [stabilityCount, setStabilityCount] = useState(0);
    const [lastFeedback, setLastFeedback] = useState<{ isCorrect: boolean; letter: string } | null>(null);

    const questionsRef = useRef<string[]>([]);
    const questionIndexRef = useRef(0);
    const questionStartTimeRef = useRef<number>(0);
    const timerIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const lastDetectedLetterRef = useRef<string>("");
    const stabilityCounterRef = useRef<number>(0);

    // Generate random questions without consecutive repeats
    const generateQuestions = useCallback((numQuestions: number): string[] => {
        const questions: string[] = [];
        let previousLetter: string | null = null;

        for (let i = 0; i < numQuestions; i++) {
            let letter: string;
            do {
                letter = AVAILABLE_LETTERS[Math.floor(Math.random() * AVAILABLE_LETTERS.length)];
            } while (letter === previousLetter && AVAILABLE_LETTERS.length > 1);

            questions.push(letter);
            previousLetter = letter;
        }

        return questions;
    }, []);

    // Start the quiz
    const startQuiz = useCallback((numQuestions: number) => {
        const questions = generateQuestions(numQuestions);
        questionsRef.current = questions;
        questionIndexRef.current = 0;
        setScore(0);
        setResults([]);
        setQuizState("ready");

        // Automatically move to first question after a brief delay
        setTimeout(() => {
            setCurrentQuestion({
                targetLetter: questions[0],
                questionNumber: 1,
                totalQuestions: numQuestions,
            });
            questionStartTimeRef.current = Date.now();
            setTimeRemaining(QUESTION_TIME_LIMIT);
            setStabilityCount(0);
            stabilityCounterRef.current = 0;
            setQuizState("active");
        }, 1000);
    }, [generateQuestions]);

    // Timer countdown
    useEffect(() => {
        if (quizState === "active") {
            timerIntervalRef.current = setInterval(() => {
                const elapsed = (Date.now() - questionStartTimeRef.current) / 1000;
                const remaining = Math.max(0, QUESTION_TIME_LIMIT - elapsed);
                setTimeRemaining(remaining);

                // Time's up - incorrect answer
                if (remaining <= 0) {
                    if (timerIntervalRef.current) {
                        clearInterval(timerIntervalRef.current);
                    }

                    const timeElapsed = (Date.now() - questionStartTimeRef.current) / 1000;
                    setResults(prev => [...prev, {
                        targetLetter: currentQuestion!.targetLetter,
                        isCorrect: false,
                        timeElapsed,
                    }]);

                    setLastFeedback({
                        isCorrect: false,
                        letter: currentQuestion!.targetLetter,
                    });
                    setQuizState("feedback");
                }
            }, 100);

            return () => {
                if (timerIntervalRef.current) {
                    clearInterval(timerIntervalRef.current);
                }
            };
        }
    }, [quizState, currentQuestion]);

    // Check if detected letter matches target
    const checkAnswer = useCallback((detectedLetter: string) => {
        if (quizState !== "active" || !currentQuestion) return;

        const targetLetter = currentQuestion.targetLetter;

        // Track stability
        if (detectedLetter === targetLetter) {
            if (detectedLetter === lastDetectedLetterRef.current) {
                // Increment internal counter
                stabilityCounterRef.current += 1;

                // Update UI
                setStabilityCount(stabilityCounterRef.current);

                // Check threshold immediately using ref
                if (stabilityCounterRef.current >= STABLE_THRESHOLD) {
                    if (timerIntervalRef.current) {
                        clearInterval(timerIntervalRef.current);
                    }

                    const timeElapsed = (Date.now() - questionStartTimeRef.current) / 1000;
                    setResults(prev => [...prev, {
                        targetLetter,
                        isCorrect: true,
                        timeElapsed,
                    }]);
                    setScore(prev => prev + 1);

                    setLastFeedback({
                        isCorrect: true,
                        letter: targetLetter,
                    });
                    setQuizState("feedback");
                }
            } else {
                stabilityCounterRef.current = 1;
                setStabilityCount(1);
            }
        } else {
            stabilityCounterRef.current = 0;
            setStabilityCount(0);
        }

        lastDetectedLetterRef.current = detectedLetter;
    }, [quizState, currentQuestion]);

    // Reset stability counter
    const resetStability = useCallback(() => {
        setStabilityCount(0);
        stabilityCounterRef.current = 0;
        lastDetectedLetterRef.current = "";
    }, []);

    // Move to next question
    const nextQuestion = useCallback(() => {
        questionIndexRef.current += 1;

        // Check if quiz is complete
        if (questionIndexRef.current >= questionsRef.current.length) {
            setQuizState("results");
            setCurrentQuestion(null);
            return;
        }

        // Load next question
        const nextLetter = questionsRef.current[questionIndexRef.current];
        setCurrentQuestion({
            targetLetter: nextLetter,
            questionNumber: questionIndexRef.current + 1,
            totalQuestions: questionsRef.current.length,
        });

        questionStartTimeRef.current = Date.now();
        setTimeRemaining(QUESTION_TIME_LIMIT);
        setTimeRemaining(QUESTION_TIME_LIMIT);
        setStabilityCount(0);
        stabilityCounterRef.current = 0;
        lastDetectedLetterRef.current = "";
        setQuizState("active");
    }, []);

    // Reset quiz to setup
    const resetQuiz = useCallback(() => {
        setQuizState("setup");
        setCurrentQuestion(null);
        setScore(0);
        setResults([]);
        setTimeRemaining(QUESTION_TIME_LIMIT);
        setStabilityCount(0);
        stabilityCounterRef.current = 0;
        setLastFeedback(null);
        questionsRef.current = [];
        questionIndexRef.current = 0;
        lastDetectedLetterRef.current = "";

        if (timerIntervalRef.current) {
            clearInterval(timerIntervalRef.current);
        }
    }, []);

    // Skip to results (for testing or abort)
    const skipToResults = useCallback(() => {
        if (timerIntervalRef.current) {
            clearInterval(timerIntervalRef.current);
        }
        setQuizState("results");
    }, []);

    return {
        quizState,
        currentQuestion,
        score,
        results,
        timeRemaining,
        stabilityCount,
        lastFeedback,
        startQuiz,
        nextQuestion,
        checkAnswer,
        resetStability,
        resetQuiz,
        skipToResults,
    };
}
