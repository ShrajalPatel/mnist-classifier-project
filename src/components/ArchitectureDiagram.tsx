"use client"

import React from 'react'
import { Card } from '@/components/ui/card'
import { ArrowRight } from 'lucide-react'

export default function ArchitectureDiagram() {
  const layers = [
    { name: 'Input', neurons: 784, description: '28×28 pixels', color: 'bg-blue-500' },
    { name: 'Hidden 1', neurons: 128, description: 'ReLU activation', color: 'bg-purple-500' },
    { name: 'Hidden 2', neurons: 64, description: 'ReLU activation', color: 'bg-pink-500' },
    { name: 'Output', neurons: 10, description: 'Softmax (0-9)', color: 'bg-green-500' }
  ]

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div>
          <h3 className="text-xl font-bold mb-2">Neural Network Architecture</h3>
          <p className="text-sm text-muted-foreground">
            Simple feedforward architecture used across all implementations
          </p>
        </div>

        <div className="flex flex-col md:flex-row items-center justify-around gap-4">
          {layers.map((layer, idx) => (
            <React.Fragment key={layer.name}>
              <div className="flex flex-col items-center space-y-2">
                <div className={`${layer.color} rounded-lg p-4 text-white shadow-lg`}>
                  <div className="text-center">
                    <p className="text-xs font-medium opacity-90">{layer.name}</p>
                    <p className="text-2xl font-bold">{layer.neurons}</p>
                    <p className="text-xs opacity-80">neurons</p>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground text-center">
                  {layer.description}
                </p>
              </div>
              
              {idx < layers.length - 1 && (
                <ArrowRight className="h-6 w-6 text-muted-foreground hidden md:block" />
              )}
              {idx < layers.length - 1 && (
                <ArrowRight className="h-6 w-6 text-muted-foreground rotate-90 md:hidden" />
              )}
            </React.Fragment>
          ))}
        </div>

        <div className="border-t pt-4 space-y-2">
          <h4 className="text-sm font-semibold">Training Configuration</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground text-xs">Loss Function</p>
              <p className="font-medium">Cross-Entropy</p>
            </div>
            <div>
              <p className="text-muted-foreground text-xs">Optimizer</p>
              <p className="font-medium">SGD + Momentum</p>
            </div>
            <div>
              <p className="text-muted-foreground text-xs">Learning Rate</p>
              <p className="font-medium">0.01</p>
            </div>
            <div>
              <p className="text-muted-foreground text-xs">Batch Size</p>
              <p className="font-medium">64</p>
            </div>
          </div>
        </div>
      </div>
    </Card>
  )
}
