"use client";

import { useState } from "react";
import { CameraView } from "@/components/CameraView";
import { DetectionDisplay } from "@/components/DetectionDisplay";
import { ModelSelector, type ModelType } from "@/components/ModelSelector";
import { ResearchPaper } from "@/components/ResearchPaper";
import { ThemeToggle } from "@/components/ThemeToggle";
import { QuizSetup, QuizActive, QuizFeedback, QuizResults } from "@/components";
import { useTheme } from "@/hooks/useTheme";
import { useQuiz } from "@/hooks/useQuiz";
import { GraduationCap, Eye } from "lucide-react";

interface Prediction {
  letter: string;
  confidence: number;
}

export default function Home() {
  const [detectedLetter, setDetectedLetter] = useState<string>("");
  const [confidence, setConfidence] = useState<number>(0);
  const [topPredictions, setTopPredictions] = useState<Prediction[]>([]);
  const [isQuizMode, setIsQuizMode] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelType>("mlp");
  const { theme, toggleTheme } = useTheme();

  const quiz = useQuiz();

  // Handle detection in quiz mode
  const handleDetection = (letter: string, conf: number, preds?: Prediction[]) => {
    setDetectedLetter(letter);
    setConfidence(conf);
    if (preds) setTopPredictions(preds);

    // If in quiz mode and active, check the answer
    if (isQuizMode && quiz.quizState === "active") {
      quiz.checkAnswer(letter);
    }
  };

  // Toggle between quiz and detection mode
  const toggleMode = () => {
    setIsQuizMode(!isQuizMode);
    if (isQuizMode) {
      // Exiting quiz mode - reset quiz
      quiz.resetQuiz();
    }
  };

  // Determine spotlight colors based on model
  const isKaggleModel = selectedModel === "kaggle-mlp";
  const spotlightColor1 = isKaggleModel 
    ? "bg-green-500/40 dark:bg-green-500/50" 
    : "bg-blue-500/40 dark:bg-blue-500/50";
  const spotlightColor2 = isKaggleModel 
    ? "bg-green-400/30 dark:bg-green-400/40" 
    : "bg-blue-400/30 dark:bg-blue-400/40";

  return (
    <div className={theme === "dark" ? "dark" : ""}>
      <div className="min-h-screen bg-white dark:bg-black text-black dark:text-white transition-colors duration-300">
        {/* Header */}
        <header className="border-b border-black/20 dark:border-white/20 backdrop-blur-xl bg-white/80 dark:bg-black/80 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
            <div>
              <h1 className="tracking-tight">ASL Live Translator</h1>
              <p className="text-black/60 dark:text-white/60 mt-1">
                {isQuizMode ? "Quiz Mode - Test Your Skills" : "Real-time American Sign Language Recognition"}
              </p>
            </div>
            <div className="flex items-center gap-4">
              {/* Model Selector - only show in detection mode */}
              {!isQuizMode && (
                <ModelSelector 
                  selectedModel={selectedModel} 
                  onModelChange={setSelectedModel} 
                />
              )}
              
              {/* Quiz Mode Toggle */}
              <button
                onClick={toggleMode}
                className="flex items-center gap-2 px-4 py-2 rounded-full border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60 backdrop-blur-xl hover:bg-white/80 dark:hover:bg-black/80 transition-all"
              >
                {isQuizMode ? (
                  <>
                    <Eye className="w-4 h-4" />
                    <span className="text-sm">Detection Mode</span>
                  </>
                ) : (
                  <>
                    <GraduationCap className="w-4 h-4" />
                    <span className="text-sm">Quiz Mode</span>
                  </>
                )}
              </button>

              <ThemeToggle theme={theme} onToggle={toggleTheme} />
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-6 py-8 relative z-10">
          {isQuizMode ? (
            /* Quiz Mode */
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-20 relative">
              {/* Animated Background Spotlights */}
              <div className="absolute inset-0 pointer-events-none overflow-hidden">
                <div className="absolute w-[700px] h-[700px] bg-purple-500/40 dark:bg-purple-500/50 rounded-full blur-[130px] animate-spotlight-bg" />
                <div className="absolute w-[500px] h-[500px] bg-purple-400/30 dark:bg-purple-400/40 rounded-full blur-[100px] animate-spotlight-bg-2" />
              </div>

              {/* Quiz Content - Takes 3 columns */}
              <div className="lg:col-span-3 relative z-10">
                {quiz.quizState === "setup" && (
                  <QuizSetup onStart={quiz.startQuiz} />
                )}

                {(quiz.quizState === "ready" || quiz.quizState === "active") && quiz.currentQuestion && (
                  <div className="space-y-6">
                    <QuizActive
                      question={quiz.currentQuestion}
                      score={quiz.score}
                      timeRemaining={quiz.timeRemaining}
                      stabilityCount={quiz.stabilityCount}
                    />

                    {/* Camera View for Quiz */}
                    <CameraView
                      onDetection={handleDetection}
                      modelType={selectedModel}
                    />
                  </div>
                )}

                {quiz.quizState === "feedback" && quiz.lastFeedback && (
                  <QuizFeedback
                    isCorrect={quiz.lastFeedback.isCorrect}
                    targetLetter={quiz.lastFeedback.letter}
                    onNext={quiz.nextQuestion}
                  />
                )}

                {quiz.quizState === "results" && (
                  <QuizResults
                    score={quiz.score}
                    totalQuestions={quiz.results.length}
                    results={quiz.results}
                    onTryAgain={() => quiz.startQuiz(quiz.results.length)}
                    onBackToDetection={toggleMode}
                  />
                )}
              </div>

              {/* Detection Display - Takes 1 column (only show during active quiz) */}
              {(quiz.quizState === "active" || quiz.quizState === "ready") && (
                <div className="lg:col-span-1 relative z-10">
                  <DetectionDisplay
                    letter={detectedLetter}
                    confidence={confidence}
                    topPredictions={topPredictions}
                  />
                </div>
              )}
            </div>
          ) : (
            /* Detection Mode */
            <>
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-20 relative">
                {/* Animated Background Spotlights - color changes based on model */}
                <div className="absolute inset-0 pointer-events-none overflow-hidden">
                  <div className={`absolute w-[700px] h-[700px] ${spotlightColor1} rounded-full blur-[130px] animate-spotlight-bg transition-colors duration-500`} />
                  <div className={`absolute w-[500px] h-[500px] ${spotlightColor2} rounded-full blur-[100px] animate-spotlight-bg-2 transition-colors duration-500`} />
                </div>

                {/* Camera View - Takes 3 columns */}
                <div className="lg:col-span-3 relative z-10">
                  <CameraView
                    onDetection={handleDetection}
                    modelType={selectedModel}
                  />
                </div>

                {/* Detection Display - Takes 1 column */}
                <div className="lg:col-span-1 relative z-10">
                  <DetectionDisplay
                    letter={detectedLetter}
                    confidence={confidence}
                    topPredictions={topPredictions}
                  />
                </div>
              </div>

              {/* Research Paper Section */}
              <ResearchPaper />
            </>
          )}
        </main>
      </div>
    </div>
  );
}
