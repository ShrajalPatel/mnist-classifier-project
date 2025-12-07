"use client"

import React from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { CheckCircle2, Brain } from 'lucide-react'

interface Prediction {
  digit: number
  confidence: number
  probabilities: number[]
}

interface PredictionDisplayProps {
  predictions: {
    pytorch?: Prediction
    tensorflow?: Prediction
    numpy?: Prediction
  }
}

export default function PredictionDisplay({ predictions }: PredictionDisplayProps) {
  const models = [
    { key: 'pytorch', name: 'PyTorch', color: 'bg-blue-500' },
    { key: 'tensorflow', name: 'TensorFlow', color: 'bg-orange-500' },
    { key: 'numpy', name: 'NumPy', color: 'bg-green-500' }
  ] as const

  const hasPredictions = Object.keys(predictions).length > 0

  if (!hasPredictions) {
    return (
      <Card className="p-8">
        <div className="text-center text-muted-foreground">
          <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>Draw a digit to see predictions</p>
        </div>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      {models.map((model) => {
        const prediction = predictions[model.key as keyof typeof predictions]
        if (!prediction) return null

        return (
          <Card key={model.key} className="p-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${model.color}`} />
                  <h4 className="font-semibold">{model.name}</h4>
                </div>
                <Badge variant="secondary" className="text-lg px-3 py-1">
                  {prediction.digit}
                </Badge>
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Confidence</span>
                  <span className="font-medium">{(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-secondary rounded-full h-2">
                  <div
                    className={`${model.color} h-2 rounded-full transition-all duration-300`}
                    style={{ width: `${prediction.confidence * 100}%` }}
                  />
                </div>
              </div>

              <div className="space-y-1">
                <p className="text-xs text-muted-foreground font-medium">Probability Distribution</p>
                {prediction.probabilities.map((prob, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-xs">
                    <span className="w-4 text-muted-foreground">{idx}</span>
                    <div className="flex-1 bg-secondary rounded-full h-1.5">
                      <div
                        className={`${model.color} h-1.5 rounded-full transition-all duration-300`}
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className="w-12 text-right text-muted-foreground">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        )
      })}
    </div>
  )
}
