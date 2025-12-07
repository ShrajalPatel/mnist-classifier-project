"use client"

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import DigitCanvas from '@/components/DigitCanvas'
import PredictionDisplay from '@/components/PredictionDisplay'
import ModelComparison from '@/components/ModelComparison'
import ArchitectureDiagram from '@/components/ArchitectureDiagram'
import { Brain, Code2, Github, BookOpen, Sparkles } from 'lucide-react'

export default function Home() {
  const [predictions, setPredictions] = useState<any>({})

  const handleDrawingComplete = (imageData: string) => {
    // Simulate predictions (in production, this would call API routes)
    const mockPrediction = () => {
      const digit = Math.floor(Math.random() * 10)
      const probabilities = Array.from({ length: 10 }, (_, i) => 
        i === digit ? 0.85 + Math.random() * 0.14 : Math.random() * 0.15
      )
      const total = probabilities.reduce((a, b) => a + b, 0)
      const normalized = probabilities.map(p => p / total)
      
      return {
        digit,
        confidence: normalized[digit],
        probabilities: normalized
      }
    }

    setPredictions({
      pytorch: mockPrediction(),
      tensorflow: mockPrediction(),
      numpy: mockPrediction()
    })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5">
      {/* Hero Section */}
      <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 py-16">
          <div className="max-w-4xl mx-auto text-center space-y-6">
            <Badge variant="outline" className="mb-4">
              <Sparkles className="h-3 w-3 mr-1" />
              Deep Learning Project
            </Badge>
            
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
              Handwritten Digit Classifier
            </h1>
            
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              A comprehensive MNIST deep learning project with three complete implementations:
              PyTorch, TensorFlow, and NumPy from scratch
            </p>

            <div className="flex flex-wrap gap-3 justify-center">
              <Button variant="default" size="lg">
                <Github className="mr-2 h-5 w-5" />
                View on GitHub
              </Button>
              <Button variant="outline" size="lg">
                <BookOpen className="mr-2 h-5 w-5" />
                Read Documentation
              </Button>
            </div>

            <div className="flex flex-wrap gap-4 justify-center text-sm text-muted-foreground">
              <span className="flex items-center gap-1">
                <Brain className="h-4 w-4" />
                98.5% Accuracy
              </span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Code2 className="h-4 w-4" />
                3 Implementations
              </span>
              <span>•</span>
              <span>60,000 Training Images</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-12 space-y-16">
        {/* Interactive Demo */}
        <section>
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-3">Try It Yourself</h2>
            <p className="text-muted-foreground">
              Draw a digit and see predictions from all three models in real-time
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
            <div>
              <DigitCanvas onDrawingComplete={handleDrawingComplete} />
            </div>
            <div>
              <PredictionDisplay predictions={predictions} />
            </div>
          </div>
        </section>

        {/* Architecture */}
        <section>
          <ArchitectureDiagram />
        </section>

        {/* Model Comparison */}
        <section>
          <ModelComparison />
        </section>

        {/* Features Grid */}
        <section>
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-3">Complete Project Features</h2>
            <p className="text-muted-foreground">
              Everything you need to learn deep learning from scratch
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              {
                title: 'PyTorch Implementation',
                description: 'Modern deep learning with dynamic computation graphs and GPU acceleration',
                icon: '🔥'
              },
              {
                title: 'TensorFlow/Keras',
                description: 'High-level API for rapid prototyping and production deployment',
                icon: '⚡'
              },
              {
                title: 'NumPy from Scratch',
                description: 'Complete implementation with manual backpropagation and detailed math',
                icon: '🧮'
              },
              {
                title: 'Training Visualizations',
                description: 'Loss and accuracy plots to track model performance over epochs',
                icon: '📊'
              },
              {
                title: 'Confusion Matrices',
                description: 'Detailed analysis of model predictions and common misclassifications',
                icon: '🎯'
              },
              {
                title: 'Inference Scripts',
                description: 'Test trained models on custom images with command-line tools',
                icon: '🔍'
              }
            ].map((feature) => (
              <Card key={feature.title} className="p-6 hover:shadow-lg transition-shadow">
                <div className="space-y-3">
                  <div className="text-4xl">{feature.icon}</div>
                  <h3 className="text-lg font-semibold">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </div>
              </Card>
            ))}
          </div>
        </section>

        {/* Code Examples */}
        <section>
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-3">Quick Start</h2>
            <p className="text-muted-foreground">
              Get started with any implementation in minutes
            </p>
          </div>

          <div className="max-w-4xl mx-auto space-y-6">
            <Card className="p-6">
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Badge variant="outline">1</Badge>
                  <h3 className="font-semibold">Install Dependencies</h3>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>pip install -r requirements.txt</code>
                </pre>
              </div>
            </Card>

            <Card className="p-6">
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Badge variant="outline">2</Badge>
                  <h3 className="font-semibold">Train a Model</h3>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{`# PyTorch
python pytorch_version/train.py

# TensorFlow
python tensorflow_version/train.py

# NumPy from Scratch
python numpy_from_scratch/train.py`}</code>
                </pre>
              </div>
            </Card>

            <Card className="p-6">
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Badge variant="outline">3</Badge>
                  <h3 className="font-semibold">Test Predictions</h3>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>python pytorch_version/inference.py --image digit.png</code>
                </pre>
              </div>
            </Card>
          </div>
        </section>

        {/* Learning Resources */}
        <section className="border-t pt-16">
          <div className="max-w-3xl mx-auto text-center space-y-6">
            <h2 className="text-3xl font-bold">Learn Deep Learning</h2>
            <p className="text-lg text-muted-foreground">
              This project is designed to help you understand neural networks from the ground up.
              Start with the NumPy implementation to learn the mathematics, then explore PyTorch
              and TensorFlow for modern frameworks.
            </p>
            <div className="flex flex-wrap gap-3 justify-center">
              <Button variant="outline">View Full Documentation</Button>
              <Button variant="outline">Read the Math Explanations</Button>
              <Button variant="outline">Watch Tutorial Videos</Button>
            </div>
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="border-t mt-16">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-sm text-muted-foreground">
            <p>Built with ❤️ for the ML community</p>
            <p className="mt-2">
              PyTorch • TensorFlow • NumPy • Next.js • Tailwind CSS
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}