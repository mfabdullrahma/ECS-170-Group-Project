"use client";

import { useState } from "react";
import { Play, Info } from "lucide-react";

interface QuizSetupProps {
    onStart: (numQuestions: number) => void;
}

export function QuizSetup({ onStart }: QuizSetupProps) {
    const [numQuestions, setNumQuestions] = useState(5);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (numQuestions > 0 && numQuestions <= 24) {
            onStart(numQuestions);
        }
    };

    return (
        <div className="min-h-[600px] flex items-center justify-center relative">
            {/* Animated Background */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute w-[600px] h-[600px] bg-blue-500/30 dark:bg-blue-500/40 rounded-full blur-[120px] animate-spotlight-centered" />
            </div>

            {/* Setup Card */}
            <div className="relative z-10 w-full max-w-2xl">
                <div className="rounded-2xl border border-black/20 dark:border-white/20 backdrop-blur-xl bg-white/60 dark:bg-black/60 shadow-2xl overflow-hidden">
                    {/* Header */}
                    <div className="p-8 border-b border-black/10 dark:border-white/10 bg-gradient-to-br from-blue-500/20 to-purple-500/20">
                        <h2 className="text-3xl font-bold mb-2">ASL Quiz Mode</h2>
                        <p className="text-black/70 dark:text-white/70">
                            Test your ASL knowledge and earn points!
                        </p>
                    </div>

                    {/* Content */}
                    <div className="p-8">
                        {/* Instructions */}
                        <div className="mb-8 p-6 rounded-xl bg-blue-500/10 border border-blue-500/20">
                            <div className="flex items-start gap-3">
                                <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
                                <div className="space-y-2 text-sm">
                                    <p className="font-medium text-blue-900 dark:text-blue-100">How it works:</p>
                                    <ul className="space-y-1 text-black/70 dark:text-white/70">
                                        <li>• A random letter will be displayed on screen</li>
                                        <li>• Make the corresponding ASL hand sign</li>
                                        <li>• Hold the sign steady for a few seconds to earn a point</li>
                                        <li>• You have 10 seconds per question</li>
                                        <li>• Try to get the highest score possible!</li>
                                    </ul>
                                </div>
                            </div>
                        </div>

                        {/* Form */}
                        <form onSubmit={handleSubmit} className="space-y-6">
                            <div>
                                <label htmlFor="numQuestions" className="block mb-3 font-medium">
                                    Number of Questions
                                </label>
                                <div className="flex items-center gap-4">
                                    <input
                                        type="range"
                                        id="numQuestions"
                                        min="1"
                                        max="24"
                                        value={numQuestions}
                                        onChange={(e) => setNumQuestions(parseInt(e.target.value))}
                                        className="flex-1 h-2 bg-black/10 dark:bg-white/10 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                    />
                                    <div className="w-16 text-center">
                                        <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                                            {numQuestions}
                                        </span>
                                    </div>
                                </div>
                                <div className="flex justify-between mt-2 text-xs text-black/50 dark:text-white/50">
                                    <span>1 question</span>
                                    <span>24 questions</span>
                                </div>
                            </div>

                            {/* Start Button */}
                            <button
                                type="submit"
                                className="w-full flex items-center justify-center gap-3 px-8 py-4 rounded-xl bg-gradient-to-r from-blue-500 to-purple-500 text-white font-medium shadow-lg hover:shadow-xl hover:scale-[1.02] active:scale-[0.98] transition-all"
                            >
                                <Play className="w-5 h-5" />
                                Start Quiz
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    );
}
