"use client";

import { QuizQuestion } from "@/hooks/useQuiz";
import { Timer, Trophy } from "lucide-react";

interface QuizActiveProps {
    question: QuizQuestion;
    score: number;
    timeRemaining: number;
    stabilityCount: number;
    stabilityThreshold?: number;
}

export function QuizActive({
    question,
    score,
    timeRemaining,
    stabilityCount,
    stabilityThreshold = 40,
}: QuizActiveProps) {
    const { targetLetter, questionNumber, totalQuestions } = question;

    // Calculate progress percentages
    const timerProgress = (timeRemaining / 10) * 100;
    const stabilityProgress = (stabilityCount / stabilityThreshold) * 100;

    // Timer color based on remaining time
    const getTimerColor = () => {
        if (timeRemaining > 6) return "text-green-600 dark:text-green-400";
        if (timeRemaining > 3) return "text-yellow-600 dark:text-yellow-400";
        return "text-red-600 dark:text-red-400";
    };

    return (
        <div className="space-y-6">
            {/* Header Stats */}
            <div className="flex items-center justify-between">
                {/* Progress */}
                <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/60 dark:bg-black/60 backdrop-blur-xl border border-black/20 dark:border-white/20">
                    <span className="text-sm font-medium">
                        Question {questionNumber}/{totalQuestions}
                    </span>
                </div>

                {/* Score */}
                <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/60 dark:bg-black/60 backdrop-blur-xl border border-black/20 dark:border-white/20">
                    <Trophy className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
                    <span className="text-sm font-medium">Score: {score}</span>
                </div>
            </div>

            {/* Target Letter Display */}
            <div className="relative rounded-2xl overflow-hidden border border-black/20 dark:border-white/20 backdrop-blur-xl bg-white/40 dark:bg-black/40 shadow-2xl">
                {/* Animated Background */}
                <div className="absolute inset-0 overflow-hidden pointer-events-none">
                    <div className="absolute inset-0 bg-gradient-to-br from-blue-500/30 via-purple-500/30 to-pink-500/30 animate-gradient" />
                    <div className="absolute w-[400px] h-[400px] bg-blue-500/40 rounded-full blur-[100px] animate-spotlight-centered" />
                </div>

                {/* Content */}
                <div className="relative z-10 p-16 flex flex-col items-center justify-center min-h-[400px]">
                    <p className="text-lg text-black/60 dark:text-white/60 mb-4">Make this sign:</p>

                    {/* Giant Letter */}
                    <div className="text-[12rem] font-bold leading-none bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent animate-pulse-slow">
                        {targetLetter}
                    </div>

                    {/* Stability Progress Bar */}
                    {stabilityCount > 0 && (
                        <div className="mt-8 w-full max-w-md">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-black/60 dark:text-white/60">Hold steady...</span>
                                <span className="text-sm font-medium text-blue-600 dark:text-blue-400">
                                    {Math.round(stabilityProgress)}%
                                </span>
                            </div>
                            <div className="h-3 bg-black/10 dark:bg-white/10 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-blue-500 to-green-500 transition-all duration-100 rounded-full"
                                    style={{ width: `${stabilityProgress}%` }}
                                />
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Timer */}
            <div className="flex items-center justify-center gap-3 p-4 rounded-xl bg-white/60 dark:bg-black/60 backdrop-blur-xl border border-black/20 dark:border-white/20">
                <Timer className={`w-5 h-5 ${getTimerColor()}`} />
                <span className={`text-2xl font-bold ${getTimerColor()}`}>
                    {Math.ceil(timeRemaining)}s
                </span>

                {/* Timer Progress Bar */}
                <div className="flex-1 max-w-xs h-2 bg-black/10 dark:bg-white/10 rounded-full overflow-hidden">
                    <div
                        className={`h-full transition-all duration-100 rounded-full ${timeRemaining > 6
                                ? "bg-green-500"
                                : timeRemaining > 3
                                    ? "bg-yellow-500"
                                    : "bg-red-500"
                            }`}
                        style={{ width: `${timerProgress}%` }}
                    />
                </div>
            </div>
        </div>
    );
}
