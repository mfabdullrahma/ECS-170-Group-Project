"use client";

import { CheckCircle2, XCircle, ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";

interface QuizFeedbackProps {
    isCorrect: boolean;
    targetLetter: string;
    onNext: () => void;
}

export function QuizFeedback({ isCorrect, targetLetter, onNext }: QuizFeedbackProps) {
    const [showConfetti, setShowConfetti] = useState(false);

    useEffect(() => {
        if (isCorrect) {
            setShowConfetti(true);
            const timer = setTimeout(() => setShowConfetti(false), 3000);
            return () => clearTimeout(timer);
        }
    }, [isCorrect]);

    return (
        <div className="min-h-[600px] flex items-center justify-center relative">
            {/* Animated Background */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div
                    className={`absolute w-[600px] h-[600px] rounded-full blur-[120px] animate-spotlight-centered ${isCorrect
                        ? "bg-green-500/40 dark:bg-green-500/50"
                        : "bg-red-500/40 dark:bg-red-500/50"
                        }`}
                />
            </div>

            {/* Confetti Effect for Correct Answers */}
            {showConfetti && (
                <div className="absolute inset-0 pointer-events-none overflow-hidden">
                    {[...Array(50)].map((_, i) => (
                        <div
                            key={i}
                            className="absolute w-2 h-2 rounded-full animate-confetti"
                            style={{
                                left: `${Math.random() * 100}%`,
                                top: `-10px`,
                                backgroundColor: `hsl(${Math.random() * 360}, 70%, 60%)`,
                                animationDelay: `${Math.random() * 0.5}s`,
                                animationDuration: `${2 + Math.random() * 2}s`,
                            }}
                        />
                    ))}
                </div>
            )}

            {/* Feedback Card */}
            <div className="relative z-10 w-full max-w-2xl">
                <div className="rounded-2xl border border-black/20 dark:border-white/20 backdrop-blur-xl bg-white/60 dark:bg-black/60 shadow-2xl overflow-hidden">
                    {/* Content */}
                    <div className="p-12 text-center">
                        {/* Icon */}
                        <div className="flex justify-center mb-6">
                            {isCorrect ? (
                                <CheckCircle2 className="w-24 h-24 text-green-600 dark:text-green-400 animate-scale-in" />
                            ) : (
                                <XCircle className="w-24 h-24 text-red-600 dark:text-red-400 animate-scale-in" />
                            )}
                        </div>

                        {/* Message */}
                        <h2
                            className={`text-4xl font-bold mb-4 ${isCorrect
                                ? "text-green-600 dark:text-green-400"
                                : "text-red-600 dark:text-red-400"
                                }`}
                        >
                            {isCorrect ? "Correct!" : "Incorrect"}
                        </h2>

                        {!isCorrect && (
                            <div className="mb-8">
                                <p className="text-lg text-black/70 dark:text-white/70 mb-4">
                                    The correct answer was:
                                </p>
                                <div className="inline-block px-8 py-4 rounded-xl bg-black/10 dark:bg-white/10 border border-black/20 dark:border-white/20">
                                    <span className="text-6xl font-bold">{targetLetter}</span>
                                </div>

                                {/* ASL Reference Image */}
                                <div className="mt-6">
                                    <p className="text-sm text-black/60 dark:text-white/60 mb-3">
                                        Reference sign:
                                    </p>
                                    <div className="flex justify-center">
                                        <div className="rounded-xl overflow-hidden border border-black/20 dark:border-white/20 shadow-lg">
                                            <img
                                                src={`/asl-letters/ASL_${targetLetter}.JPG`}
                                                alt={`ASL sign for ${targetLetter}`}
                                                className="w-64 h-64 object-cover"
                                                onError={(e) => {
                                                    // Fallback if image doesn't exist
                                                    e.currentTarget.style.display = "none";
                                                }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {isCorrect && (
                            <p className="text-lg text-black/70 dark:text-white/70 mb-8">
                                Great job! Keep it up! 🎉
                            </p>
                        )}

                        {/* Next Button */}
                        <button
                            onClick={onNext}
                            className="inline-flex items-center gap-3 px-8 py-4 rounded-xl bg-gradient-to-r from-blue-500 to-purple-500 text-white font-medium shadow-lg hover:shadow-xl hover:scale-[1.02] active:scale-[0.98] transition-all"
                        >
                            Next Question
                            <ArrowRight className="w-5 h-5" />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
