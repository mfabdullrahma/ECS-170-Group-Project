"use client";

import { Brain, Layers, Zap, CheckCircle, AlertTriangle, Users, Code, Lightbulb, Database, Hand } from "lucide-react";

export function ResearchPaper() {
  return (
    <div className="rounded-2xl overflow-hidden border border-black/20 dark:border-white/20 bg-white/40 dark:bg-black/40 backdrop-blur-xl shadow-2xl">
      <div className="p-12 max-w-4xl mx-auto">
        {/* Title Section */}
        <div className="text-center mb-12 pb-8 border-b border-black/20 dark:border-white/20">
          <h1 className="mb-4 text-2xl font-medium">
            Real-Time ASL Recognition Using Neural Networks
          </h1>
          <p className="text-black/60 dark:text-white/60">
            ECS 170 - Introduction to Artificial Intelligence
          </p>
          <div className="flex items-center justify-center gap-6 mt-6 text-sm text-black/50 dark:text-white/50">
            <span>Group Project</span>
            <span>•</span>
            <span>Fall 2025</span>
          </div>
        </div>

        {/* Project Overview */}
        <section className="mb-12">
          <div className="inline-block px-4 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 backdrop-blur-xl mb-4">
            <span className="text-blue-600 dark:text-blue-400 text-sm">Project Overview</span>
          </div>
          <p className="text-black/70 dark:text-white/70 leading-relaxed mb-4">
            This web application demonstrates real-time American Sign Language (ASL) alphabet 
            recognition using deep learning and computer vision. The system captures video from 
            your webcam, detects hand landmarks using MediaPipe, and classifies the hand gesture 
            into one of 26 ASL letters (A-Z) using custom-trained neural networks.
          </p>
          <p className="text-black/70 dark:text-white/70 leading-relaxed">
            The entire inference pipeline runs directly in your browser using TensorFlow.js, 
            requiring no server-side processing. This enables real-time predictions without 
            any backend server, making deployment simple and accessible.
          </p>
        </section>

        {/* System Architecture */}
        <section className="mb-12">
          <h2 className="mb-6 text-xl font-medium">System Architecture</h2>

          {/* Architecture Diagram */}
          <div className="mb-8 p-8 rounded-xl bg-white/60 dark:bg-black/60 backdrop-blur-xl border border-black/20 dark:border-white/20 shadow-lg">
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <div className="flex-1 min-w-[120px]">
                <div className="p-4 rounded-lg border border-black/20 dark:border-white/20 bg-white/80 dark:bg-black/80 backdrop-blur-xl text-center shadow-lg">
                  <Code className="w-6 h-6 text-black/60 dark:text-white/60 mx-auto mb-2" />
                  <div className="text-sm mb-1 font-medium">Webcam</div>
                  <div className="text-black/50 dark:text-white/50 text-xs">Video Input</div>
                </div>
              </div>

              <div className="text-black/40 dark:text-white/40">→</div>

              <div className="flex-1 min-w-[120px]">
                <div className="p-4 rounded-lg border border-black/20 dark:border-white/20 bg-white/80 dark:bg-black/80 backdrop-blur-xl text-center shadow-lg">
                  <Layers className="w-6 h-6 text-black/60 dark:text-white/60 mx-auto mb-2" />
                  <div className="text-sm mb-1 font-medium">MediaPipe</div>
                  <div className="text-black/50 dark:text-white/50 text-xs">21 Landmarks</div>
                </div>
              </div>

              <div className="text-black/40 dark:text-white/40">→</div>

              <div className="flex-1 min-w-[120px]">
                <div className="p-4 rounded-lg border border-black/20 dark:border-white/20 bg-white/80 dark:bg-black/80 backdrop-blur-xl text-center shadow-lg">
                  <Zap className="w-6 h-6 text-black/60 dark:text-white/60 mx-auto mb-2" />
                  <div className="text-sm mb-1 font-medium">Normalize</div>
                  <div className="text-black/50 dark:text-white/50 text-xs">42 Features</div>
                </div>
              </div>

              <div className="text-black/40 dark:text-white/40">→</div>

              <div className="flex-1 min-w-[120px]">
                <div className="p-4 rounded-lg border border-blue-500/30 dark:border-blue-500/30 bg-blue-500/10 dark:bg-blue-500/20 backdrop-blur-xl text-center shadow-lg">
                  <Brain className="w-6 h-6 text-blue-600 dark:text-blue-400 mx-auto mb-2" />
                  <div className="text-sm mb-1 font-medium">MLP</div>
                  <div className="text-black/50 dark:text-white/50 text-xs">TensorFlow.js</div>
                </div>
              </div>
            </div>
            <p className="text-center text-black/50 dark:text-white/50 text-sm mt-6">
              Figure 1: End-to-end inference pipeline running entirely in the browser
            </p>
          </div>
        </section>

        {/* AI Methodologies */}
        <section className="mb-12">
          <h2 className="mb-6 text-xl font-medium">AI Methodologies & Techniques</h2>

          <div className="space-y-6">
            {/* Data Processing */}
            <div>
              <h3 className="mb-3 text-lg font-medium flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-blue-500" />
                Hand Landmark Detection
              </h3>
              <p className="text-black/70 dark:text-white/70 leading-relaxed mb-3">
                We use Google&apos;s MediaPipe Hands to detect 21 3D landmarks on each hand in real-time. 
                Each landmark represents a joint or fingertip position (x, y coordinates normalized to 0-1).
              </p>
              <div className="p-4 rounded-lg bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10">
                <code className="text-sm text-black/70 dark:text-white/70">
                  Landmarks: Wrist, Thumb (4), Index (4), Middle (4), Ring (4), Pinky (4) = 21 points
                </code>
              </div>
            </div>

            {/* Feature Engineering */}
            <div>
              <h3 className="mb-3 text-lg font-medium flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-blue-500" />
                Feature Engineering
              </h3>
              <p className="text-black/70 dark:text-white/70 leading-relaxed mb-3">
                Raw landmark coordinates are transformed into a normalized feature vector:
              </p>
              <ul className="space-y-2 text-black/70 dark:text-white/70">
                <li className="flex items-start gap-3">
                  <span className="text-blue-500 mt-1">1.</span>
                  <span><strong>Translation invariance:</strong> All coordinates are relative to the wrist position</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-500 mt-1">2.</span>
                  <span><strong>Scale invariance:</strong> Normalized by the maximum absolute coordinate value</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-500 mt-1">3.</span>
                  <span><strong>2D only:</strong> We use only X,Y coordinates (not Z depth) for stability</span>
                </li>
              </ul>
              <p className="text-black/60 dark:text-white/60 text-sm mt-3">
                Final feature vector: 21 landmarks × 2 coordinates = <strong>42 features</strong>
              </p>
            </div>

            {/* Models */}
            <div>
              <h3 className="mb-3 text-lg font-medium flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-blue-500" />
                Neural Network Models
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div className="p-5 rounded-xl border border-blue-500/30 bg-blue-500/5 dark:bg-blue-500/10 backdrop-blur-xl shadow-lg">
                  <div className="font-medium mb-2 flex items-center gap-2">
                    <Brain className="w-4 h-4 text-blue-500" />
                    MLP Model (Manual Training)
                  </div>
                  <div className="text-black/60 dark:text-white/60 text-sm space-y-1">
                    <p>• Input: 42 features</p>
                    <p>• Hidden: 128 → 64 → 32 neurons</p>
                    <p>• Activation: ReLU + Dropout(0.3, 0.3)</p>
                    <p>• Output: 25 classes (A-Y, excl. J, Z)</p>
                    <p className="text-blue-600 dark:text-blue-400 mt-2">Trained on custom hand gestures</p>
                  </div>
                </div>
                
                <div className="p-5 rounded-xl border border-green-500/30 bg-green-500/5 dark:bg-green-500/10 backdrop-blur-xl shadow-lg">
                  <div className="font-medium mb-2 flex items-center gap-2">
                    <Database className="w-4 h-4 text-green-500" />
                    Kaggle MLP Model
                  </div>
                  <div className="text-black/60 dark:text-white/60 text-sm space-y-1">
                    <p>• Input: 42 features</p>
                    <p>• Hidden: 256 → 128 → 64 neurons</p>
                    <p>• Activation: ReLU + BatchNorm + Dropout</p>
                    <p>• Output: 28 classes (A-Z + special)</p>
                    <p className="text-green-600 dark:text-green-400 mt-2">Trained on Kaggle ASL dataset</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Manual Model Training Process */}
            <div>
              <h3 className="mb-3 text-lg font-medium flex items-center gap-2">
                <Hand className="w-5 h-5 text-blue-500" />
                MLP Model Training (Manual Data Collection)
              </h3>
              <p className="text-black/70 dark:text-white/70 leading-relaxed mb-4">
                The MLP Model was trained using hand-gesture-recognition-using-mediapipe, 
                a methodology that allows for custom data collection and model training:
              </p>
              
              <div className="space-y-4">
                <div className="p-4 rounded-lg bg-blue-500/5 dark:bg-blue-500/10 border border-blue-500/20">
                  <h4 className="font-medium mb-2 text-blue-600 dark:text-blue-400">1. Data Collection Process</h4>
                  <ul className="text-sm text-black/70 dark:text-white/70 space-y-1">
                    <li>• Run the app.py script with webcam enabled</li>
                    <li>• Press &apos;k&apos; to enter keypoint logging mode</li>
                    <li>• Press 0-9 keys to assign class IDs to hand poses</li>
                    <li>• Keypoints are saved to keypoint.csv with class labels</li>
                    <li>• Each sample contains 42 normalized coordinates (21 landmarks × 2)</li>
                  </ul>
                </div>

                <div className="p-4 rounded-lg bg-blue-500/5 dark:bg-blue-500/10 border border-blue-500/20">
                  <h4 className="font-medium mb-2 text-blue-600 dark:text-blue-400">2. MLP Architecture</h4>
                  <div className="text-sm text-black/70 dark:text-white/70">
                    <p className="mb-2">The model uses a deeper architecture for improved performance:</p>
                    <div className="p-3 rounded bg-black/5 dark:bg-white/5 font-mono text-xs">
                      Input(42) → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(25, Softmax)
                    </div>
                    <p className="mt-2">Total parameters: ~16,700 (optimized for real-time inference)</p>
                  </div>
                </div>

                <div className="p-4 rounded-lg bg-blue-500/5 dark:bg-blue-500/10 border border-blue-500/20">
                  <h4 className="font-medium mb-2 text-blue-600 dark:text-blue-400">3. Training Configuration</h4>
                  <ul className="text-sm text-black/70 dark:text-white/70 space-y-1">
                    <li>• Optimizer: Adam</li>
                    <li>• Loss: Sparse Categorical Cross-entropy</li>
                    <li>• Train/Test Split: 75/25</li>
                    <li>• Epochs: Up to 1000 with early stopping (patience=20)</li>
                    <li>• Batch Size: 128</li>
                    <li>• Final Accuracy: ~96% on test set</li>
                  </ul>
                </div>

                <div className="p-4 rounded-lg bg-blue-500/5 dark:bg-blue-500/10 border border-blue-500/20">
                  <h4 className="font-medium mb-2 text-blue-600 dark:text-blue-400">4. Model Export for Web</h4>
                  <ul className="text-sm text-black/70 dark:text-white/70 space-y-1">
                    <li>• Trained model saved as .hdf5 (Keras format)</li>
                    <li>• Converted to TFLite with quantization for efficiency</li>
                    <li>• Converted to TensorFlow.js format (GraphModel)</li>
                    <li>• Deployed as static files for browser inference</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Kaggle Training */}
            <div>
              <h3 className="mb-3 text-lg font-medium flex items-center gap-2">
                <Database className="w-5 h-5 text-green-500" />
                Kaggle MLP Model Training
              </h3>
              <p className="text-black/70 dark:text-white/70 leading-relaxed mb-3">
                The Kaggle MLP Model was trained on the ASL Alphabet dataset from Kaggle:
              </p>
              <ul className="space-y-2 text-black/70 dark:text-white/70">
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>87,000+ images across 29 classes (A-Z, space, delete, nothing)</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>MediaPipe extracted keypoints from each image</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>80/10/10 train/validation/test split with stratification</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>Deeper architecture (256→128→64) with BatchNorm for stability</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>Early stopping and learning rate scheduling to prevent overfitting</span>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Model Comparison */}
        <section className="mb-12">
          <h2 className="mb-6 text-xl font-medium">Model Comparison</h2>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b border-black/20 dark:border-white/20">
                  <th className="text-left py-3 px-4 font-medium">Feature</th>
                  <th className="text-left py-3 px-4 font-medium text-blue-600 dark:text-blue-400">MLP Model</th>
                  <th className="text-left py-3 px-4 font-medium text-green-600 dark:text-green-400">Kaggle MLP Model</th>
                </tr>
              </thead>
              <tbody className="text-black/70 dark:text-white/70">
                <tr className="border-b border-black/10 dark:border-white/10">
                  <td className="py-3 px-4">Training Data</td>
                  <td className="py-3 px-4">Custom collected</td>
                  <td className="py-3 px-4">Kaggle ASL dataset</td>
                </tr>
                <tr className="border-b border-black/10 dark:border-white/10">
                  <td className="py-3 px-4">Classes</td>
                  <td className="py-3 px-4">25 (no J, Z)</td>
                  <td className="py-3 px-4">28 (A-Z + del, space)</td>
                </tr>
                <tr className="border-b border-black/10 dark:border-white/10">
                  <td className="py-3 px-4">Parameters</td>
                  <td className="py-3 px-4">~16,700</td>
                  <td className="py-3 px-4">~50,000+</td>
                </tr>
                <tr className="border-b border-black/10 dark:border-white/10">
                  <td className="py-3 px-4">Model Format</td>
                  <td className="py-3 px-4">GraphModel (TFLite)</td>
                  <td className="py-3 px-4">LayersModel (Keras)</td>
                </tr>
                <tr className="border-b border-black/10 dark:border-white/10">
                  <td className="py-3 px-4">Smoothing</td>
                  <td className="py-3 px-4">None (raw output)</td>
                  <td className="py-3 px-4">Consecutive-frame voting</td>
                </tr>
                <tr>
                  <td className="py-3 px-4">Best For</td>
                  <td className="py-3 px-4">Fast response, simple gestures</td>
                  <td className="py-3 px-4">Stable predictions, full alphabet</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Challenges */}
        <section className="mb-12">
          <h2 className="mb-6 text-xl font-medium flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            Challenges & Solutions
          </h2>

          <div className="space-y-6">
            <div className="p-5 rounded-xl border border-amber-500/30 bg-amber-500/5 dark:bg-amber-500/10">
              <h4 className="font-medium mb-2">Challenge: Prediction Flickering</h4>
              <p className="text-black/70 dark:text-white/70 text-sm mb-2">
                Raw model predictions changed rapidly frame-to-frame, causing the displayed letter 
                to flicker even when holding a steady pose.
              </p>
              <div className="flex items-start gap-2">
                <Lightbulb className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                <p className="text-green-700 dark:text-green-400 text-sm">
                  <strong>Solution:</strong> Implemented temporal smoothing with consecutive-frame voting 
                  for the Kaggle model. The MLP model outputs raw predictions for faster response.
                </p>
              </div>
            </div>

            <div className="p-5 rounded-xl border border-amber-500/30 bg-amber-500/5 dark:bg-amber-500/10">
              <h4 className="font-medium mb-2">Challenge: Training vs. Inference Mismatch</h4>
              <p className="text-black/70 dark:text-white/70 text-sm mb-2">
                The model trained on static Kaggle images performed poorly on live webcam input. 
                The Z-depth coordinate from MediaPipe was particularly unstable.
              </p>
              <div className="flex items-start gap-2">
                <Lightbulb className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                <p className="text-green-700 dark:text-green-400 text-sm">
                  <strong>Solution:</strong> Switched to 2D-only features (X,Y without Z), matching the 
                  approach used in proven hand gesture recognition systems. This improved stability significantly.
                </p>
              </div>
            </div>

            <div className="p-5 rounded-xl border border-amber-500/30 bg-amber-500/5 dark:bg-amber-500/10">
              <h4 className="font-medium mb-2">Challenge: Browser Deployment</h4>
              <p className="text-black/70 dark:text-white/70 text-sm mb-2">
                Running ML models in the browser while maintaining real-time performance required careful optimization.
              </p>
              <div className="flex items-start gap-2">
                <Lightbulb className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                <p className="text-green-700 dark:text-green-400 text-sm">
                  <strong>Solution:</strong> Used TensorFlow.js for model inference and MediaPipe&apos;s 
                  WebAssembly-accelerated hand detection. Models were converted from Keras/TFLite to TFJS format.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Team Contributions */}
        <section className="mb-8 pt-8 border-t border-black/20 dark:border-white/20">
          <h2 className="mb-6 text-xl font-medium flex items-center gap-2">
            <Users className="w-5 h-5 text-blue-500" />
            Team Contributions
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Caitlyn Ho</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Frontend Development, UI/UX Design, Deployment, Model Training
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Mohamed Abdullrahma</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Data Preprocessing, Feature Engineering, Model Architecture
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">David Bui</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Backend Integration, Testing
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Teddy Liu</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Model Training, Data Collection
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">David Arista</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Dataset Preparation, Research
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Johnny Betmansour</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Data Processing, Testing
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Carlos Rayo</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Documentation, Research
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Xiyan Zeng</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Data Augmentation, Model Export
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Norman Gutierrez-Ugalde</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Quality Assurance, Documentation
              </p>
            </div>
          </div>
        </section>

        {/* References */}
        <section className="pt-8 border-t border-black/20 dark:border-white/20">
          <h2 className="mb-4 text-xl font-medium">References</h2>
          <ol className="space-y-3 text-sm text-black/60 dark:text-white/60">
            <li>[1] Zhang, F., et al. &quot;MediaPipe Hands: On-device Real-time Hand Tracking.&quot; CVPR Workshop 2020.</li>
            <li>[2] Kaggle ASL Alphabet Dataset. https://www.kaggle.com/datasets/grassknoted/asl-alphabet</li>
            <li>[3] TensorFlow.js Documentation. https://www.tensorflow.org/js</li>
            <li>[4] hand-gesture-recognition-using-mediapipe. https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe</li>
          </ol>
        </section>
      </div>
    </div>
  );
}
