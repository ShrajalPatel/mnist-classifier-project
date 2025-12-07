"use client"

import React, { useRef, useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Eraser, Pencil, Download } from 'lucide-react'

interface DigitCanvasProps {
  onDrawingComplete?: (imageData: string) => void
}

export default function DigitCanvas({ onDrawingComplete }: DigitCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [hasDrawing, setHasDrawing] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set white background
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }, [])

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    setIsDrawing(true)
    setHasDrawing(true)
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.beginPath()
    
    if ('touches' in e) {
      const touch = e.touches[0]
      ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top)
    } else {
      ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top)
    }
  }

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.lineWidth = 20
    ctx.lineCap = 'round'
    ctx.strokeStyle = 'black'

    if ('touches' in e) {
      const touch = e.touches[0]
      ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top)
    } else {
      ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top)
    }
    
    ctx.stroke()
  }

  const stopDrawing = () => {
    setIsDrawing(false)
    
    // Get image data and notify parent
    const canvas = canvasRef.current
    if (canvas && hasDrawing && onDrawingComplete) {
      const imageData = canvas.toDataURL('image/png')
      onDrawingComplete(imageData)
    }
  }

  const clearCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    setHasDrawing(false)
  }

  const downloadDrawing = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const link = document.createElement('a')
    link.download = 'digit.png'
    link.href = canvas.toDataURL()
    link.click()
  }

  return (
    <Card className="p-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Pencil className="h-5 w-5" />
            Draw a Digit (0-9)
          </h3>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={clearCanvas}
              disabled={!hasDrawing}
            >
              <Eraser className="h-4 w-4 mr-2" />
              Clear
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={downloadDrawing}
              disabled={!hasDrawing}
            >
              <Download className="h-4 w-4 mr-2" />
              Save
            </Button>
          </div>
        </div>

        <div className="flex justify-center">
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            className="border-2 border-border rounded-lg cursor-crosshair bg-white touch-none"
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
          />
        </div>

        <p className="text-sm text-muted-foreground text-center">
          Draw a digit using your mouse or touch screen
        </p>
      </div>
    </Card>
  )
}
