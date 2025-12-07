"use client"

import React from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Zap, Clock, HardDrive, Target } from 'lucide-react'

const modelData = [
  {
    name: 'PyTorch',
    color: 'border-blue-500',
    bgColor: 'bg-blue-500/10',
    accuracy: '98.5%',
    trainingTime: '~2 min',
    modelSize: '1.2 MB',
    framework: 'PyTorch 2.0+',
    description: 'Flexible, Pythonic API perfect for research and experimentation',
    features: ['Dynamic computation graphs', 'GPU acceleration', 'Easy debugging']
  },
  {
    name: 'TensorFlow',
    color: 'border-orange-500',
    bgColor: 'bg-orange-500/10',
    accuracy: '98.3%',
    trainingTime: '~2.5 min',
    modelSize: '1.5 MB',
    framework: 'TensorFlow/Keras',
    description: 'High-level Keras API with production-ready deployment',
    features: ['TensorFlow Serving', 'Mobile deployment', 'Easy export']
  },
  {
    name: 'NumPy',
    color: 'border-green-500',
    bgColor: 'bg-green-500/10',
    accuracy: '97.8%',
    trainingTime: '~15 min',
    modelSize: '800 KB',
    framework: 'Pure NumPy',
    description: 'Built from scratch to understand neural network mathematics',
    features: ['No dependencies', 'Educational', 'Full control']
  }
]

export default function ModelComparison() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold mb-2">Performance Comparison</h2>
        <p className="text-muted-foreground">
          All three implementations use the same architecture for fair comparison
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        {modelData.map((model) => (
          <Card key={model.name} className={`p-6 border-2 ${model.color} ${model.bgColor}`}>
            <div className="space-y-4">
              <div>
                <h3 className="text-xl font-bold mb-1">{model.name}</h3>
                <Badge variant="outline" className="text-xs">
                  {model.framework}
                </Badge>
              </div>

              <p className="text-sm text-muted-foreground">
                {model.description}
              </p>

              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Target className="h-4 w-4 text-muted-foreground" />
                  <div className="flex-1">
                    <p className="text-xs text-muted-foreground">Accuracy</p>
                    <p className="text-lg font-bold">{model.accuracy}</p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <div className="flex-1">
                    <p className="text-xs text-muted-foreground">Training Time</p>
                    <p className="text-sm font-semibold">{model.trainingTime}</p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <HardDrive className="h-4 w-4 text-muted-foreground" />
                  <div className="flex-1">
                    <p className="text-xs text-muted-foreground">Model Size</p>
                    <p className="text-sm font-semibold">{model.modelSize}</p>
                  </div>
                </div>
              </div>

              <div>
                <p className="text-xs font-medium mb-2">Key Features:</p>
                <ul className="space-y-1">
                  {model.features.map((feature) => (
                    <li key={feature} className="text-xs text-muted-foreground flex items-center gap-2">
                      <Zap className="h-3 w-3" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}
