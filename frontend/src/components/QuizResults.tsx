"use client";

import { Trophy, RotateCcw, Home, Star } from "lucide-react";
import { QuizResult } from "@/hooks/useQuiz";

interface QuizResultsProps {
    score: number;
    totalQuestions: number;
    results: QuizResult[];
    onTryAgain: () => void;
    onBackToDetection: () => void;
}

export function QuizResults({
    score,
    totalQuestions,
    results,
    onTryAgain,
    onBackToDetection,
}: QuizResultsProps) {
    const percentage = (score / totalQuestions) * 100;

    // Performance message based on score
    const getPerformanceMessage = () => {
        if (percentage === 100) return "Perfect! You're an ASL master! 🏆";
        if (percentage >= 80) return "Excellent work! Keep it up! ⭐";
        if (percentage >= 60) return "Good job! You're improving! 👍";
        if (percentage >= 40) return "Not bad! Keep practicing! 💪";
        return "Keep learning! Practice makes perfect! 📚";
    };

    const getPerformanceColor = () => {
        if (percentage >= 80) return "text-green-600 dark:text-green-400";
        if (percentage >= 60) return "text-blue-600 dark:text-blue-400";
        if (percentage >= 40) return "text-yellow-600 dark:text-yellow-400";
        return "text-orange-600 dark:text-orange-400";
    };

    return (
        <div className="min-h-[600px] flex items-center justify-center relative">
            {/* Animated Background */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div
                    className={`absolute w-[700px] h-[700px] rounded-full blur-[130px] animate-spotlight-centered ${percentage >= 80
                            ? "bg-green-500/40 dark:bg-green-500/50"
                            : "bg-blue-500/40 dark:bg-blue-500/50"
                        }`}
                />
            </div>

            {/* Celebration Effect for High Scores */}
            {percentage >= 80 && (
                <div className="absolute inset-0 pointer-events-none overflow-hidden">
                    {[...Array(30)].map((_, i) => (
                        <Star
                            key={i}
                            className="absolute text-yellow-400 animate-float"
                            style={{
                                left: `${Math.random() * 100}%`,
                                top: `${Math.random() * 100}%`,
                                width: `${20 + Math.random() * 20}px`,
                                height: `${20 + Math.random() * 20}px`,
                                animationDelay: `${Math.random() * 2}s`,
                                animationDuration: `${3 + Math.random() * 2}s`,
                                opacity: 0.6,
                            }}
                        />
                    ))}
                </div>
            )}

            {/* Results Card */}
            <div className="relative z-10 w-full max-w-3xl">
                <div className="rounded-2xl border border-black/20 dark:border-white/20 backdrop-blur-xl bg-white/60 dark:bg-black/60 shadow-2xl overflow-hidden">
                    {/* Header */}
                    <div className="p-8 border-b border-black/10 dark:border-white/10 bg-gradient-to-br from-blue-500/20 to-purple-500/20 text-center">
                        <Trophy className="w-16 h-16 mx-auto mb-4 text-yellow-600 dark:text-yellow-400" />
                        <h2 className="text-3xl font-bold mb-2">Quiz Complete!</h2>
                    </div>

                    {/* Score Display */}
                    <div className="p-12 text-center border-b border-black/10 dark:border-white/10">
                        <p className="text-lg text-black/60 dark:text-white/60 mb-4">Your Score</p>
                        <div className="text-7xl font-bold mb-2">
                            <span className={getPerformanceColor()}>
                                {score}/{totalQuestions}
                            </span>
                        </div>
                        <div className="text-3xl font-semibold text-black/50 dark:text-white/50 mb-6">
                            {percentage.toFixed(0)}%
                        </div>
                        <p className={`text-xl font-medium ${getPerformanceColor()}`}>
                            {getPerformanceMessage()}
                        </p>
                    </div>

                    {/* Detailed Results */}
                    <div className="p-8 border-b border-black/10 dark:border-white/10">
                        <h3 className="text-lg font-semibold mb-4">Question Breakdown</h3>
                        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-3">
                            {results.map((result, index) => (
                                <div
                                    key={index}
                                    className={`p-3 rounded-lg border text-center ${result.isCorrect
                                            ? "bg-green-500/10 border-green-500/30"
                                            : "bg-red-500/10 border-red-500/30"
                                        }`}
                                >
                                    <div className="text-2xl font-bold mb-1">{result.targetLetter}</div>
                                    <div className="text-xs text-black/60 dark:text-white/60">
                                        {result.isCorrect ? "✓" : "✗"}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="p-8 flex flex-col sm:flex-row gap-4">
                        <button
                            onClick={onTryAgain}
                            className="flex-1 flex items-center justify-center gap-3 px-6 py-4 rounded-xl bg-gradient-to-r from-blue-500 to-purple-500 text-white font-medium shadow-lg hover:shadow-xl hover:scale-[1.02] active:scale-[0.98] transition-all"
                        >
                            <RotateCcw className="w-5 h-5" />
                            Try Again
                        </button>
                        <button
                            onClick={onBackToDetection}
                            className="flex-1 flex items-center justify-center gap-3 px-6 py-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60 backdrop-blur-xl hover:bg-white/80 dark:hover:bg-black/80 transition-all"
                        >
                            <Home className="w-5 h-5" />
                            Back to Detection
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
