'use client'

import React, { useEffect, useId, useState, useRef, useCallback } from 'react'
import { AlertCircle, CheckCircle, Maximize2, RotateCcw, ZoomIn, ZoomOut, Move, X, HelpCircle } from 'lucide-react'

interface InteractiveMermaidProps {
  code: string
  className?: string
  title?: string
}

interface MermaidState {
  svg: string
  error: string
  isLoading: boolean
  correctionAttempted: boolean
}

interface ViewportState {
  scale: number
  translateX: number
  translateY: number
  isDragging: boolean
  dragStart: { x: number; y: number }
  lastPanPoint: { x: number; y: number }
}

interface DiagramBounds {
  width: number
  height: number
  complexity: 'simple' | 'medium' | 'complex'
}

export default function InteractiveMermaid({ code, className, title }: InteractiveMermaidProps) {
  const [state, setState] = useState<MermaidState>({
    svg: '',
    error: '',
    isLoading: true,
    correctionAttempted: false
  })
  
  const [viewport, setViewport] = useState<ViewportState>({
    scale: 1,
    translateX: 0,
    translateY: 0,
    isDragging: false,
    dragStart: { x: 0, y: 0 },
    lastPanPoint: { x: 0, y: 0 }
  })
  
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [bounds, setBounds] = useState<DiagramBounds>({ width: 0, height: 0, complexity: 'simple' })
  const [showMinimap, setShowMinimap] = useState(false)
  const [showHelp, setShowHelp] = useState(false)
  
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<HTMLDivElement>(null)
  const id = useId().replace(/:/g, '-')

  // Preprocess code (same as original)
  const preprocessMermaid = (raw: string): string => {
    if (!raw) return ''
    let s = raw
      .replace(/\r\n/g, '\n')
      .replace(/\\n/g, '\n')
      .replace(/\uFF1B/g, ';')
      .trim()

    const hasFewNewlines = (s.match(/\n/g) || []).length < 2
    const isVeryLongSingleLine = !s.includes('\n') && s.length > 300

    if (hasFewNewlines || isVeryLongSingleLine) {
      const parts = s
        .split(';')
        .map(p => p.trim())
        .filter(p => p.length > 0)
      s = parts.join('\n')
    }

    const initMatch = s.match(/^(.*?%%\{\s*init[\s\S]*?\}%%);?/i)
    if (initMatch) {
      const initBlock = initMatch[1].trim()
      const remainder = s.slice(initMatch[0].length).trim()
      s = initBlock + '\n' + remainder
    }

    s = s.replace(/\n\s*\n\s*\n+/g, '\n\n')
    return s
  }

  // Sanitization (same as original)
  const sanitizeCode = (rawCode: string): string => {
    if (!rawCode) return ''
    return rawCode
      .replace(/[‑–—]/g, '-')
      .replace(/[\u2018\u2019\u201A]/g, "'")
      .replace(/[\u201C\u201D\u201E]/g, '"')
      .replace(/\u2026/g, '...')
      .replace(/\[([^\]]*)\(([^)]*)\)([^\]]*)\]/g, '[$1$2$3]')
      .replace(/\r\n/g, '\n')
      .replace(/\n\s*\n\s*\n+/g, '\n\n')
      .trim()
  }

  // Analyze diagram complexity
  const analyzeDiagram = (svgContent: string): DiagramBounds => {
    const parser = new DOMParser()
    const doc = parser.parseFromString(svgContent, 'image/svg+xml')
    const svgElement = doc.querySelector('svg')
    
    if (!svgElement) return { width: 400, height: 300, complexity: 'simple' }
    
    const width = parseInt(svgElement.getAttribute('width') || '400')
    const height = parseInt(svgElement.getAttribute('height') || '300')
    
    // Count nodes and connections to determine complexity
    const nodes = doc.querySelectorAll('g.node, rect, circle, ellipse').length
    const edges = doc.querySelectorAll('g.edge, path, line').length
    
    let complexity: 'simple' | 'medium' | 'complex' = 'simple'
    if (nodes > 20 || edges > 25 || width > 800 || height > 600) {
      complexity = 'complex'
    } else if (nodes > 8 || edges > 10 || width > 500 || height > 400) {
      complexity = 'medium'
    }
    
    return { width, height, complexity }
  }

  // Reset viewport to fit diagram
  const resetViewport = useCallback(() => {
    if (!containerRef.current || !bounds.width || !bounds.height) return
    
    const container = containerRef.current.getBoundingClientRect()
    const scaleX = (container.width - 40) / bounds.width
    const scaleY = (container.height - 40) / bounds.height
    const scale = Math.min(scaleX, scaleY, 1)
    
    setViewport({
      scale,
      translateX: 0,
      translateY: 0,
      isDragging: false,
      dragStart: { x: 0, y: 0 },
      lastPanPoint: { x: 0, y: 0 }
    })
  }, [bounds])

  // Handle zoom with cursor-centered scaling
  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault()
    
    if (!containerRef.current || !svgRef.current) return
    
    const container = containerRef.current.getBoundingClientRect()
    const mouseX = e.clientX - container.left
    const mouseY = e.clientY - container.top
    
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    const newScale = Math.max(0.1, Math.min(5, viewport.scale * delta))
    
    // Calculate new translation to keep zoom centered on cursor
    const scaleChange = newScale / viewport.scale
    const newTranslateX = mouseX - (mouseX - viewport.translateX) * scaleChange
    const newTranslateY = mouseY - (mouseY - viewport.translateY) * scaleChange
    
    setViewport(prev => ({
      ...prev,
      scale: newScale,
      translateX: newTranslateX,
      translateY: newTranslateY
    }))
  }, [viewport.scale, viewport.translateX, viewport.translateY])

  // Handle mouse down for panning
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return // Only left mouse button
    
    setViewport(prev => ({
      ...prev,
      isDragging: true,
      dragStart: { x: e.clientX, y: e.clientY },
      lastPanPoint: { x: prev.translateX, y: prev.translateY }
    }))
  }, [])

  // Handle mouse move for panning
  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!viewport.isDragging) return
    
    const deltaX = e.clientX - viewport.dragStart.x
    const deltaY = e.clientY - viewport.dragStart.y
    
    setViewport(prev => ({
      ...prev,
      translateX: prev.lastPanPoint.x + deltaX,
      translateY: prev.lastPanPoint.y + deltaY
    }))
  }, [viewport.isDragging, viewport.dragStart, viewport.lastPanPoint])

  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    setViewport(prev => ({
      ...prev,
      isDragging: false
    }))
  }, [])

  // Handle keyboard navigation
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!containerRef.current?.contains(document.activeElement)) return
    
    const step = 20
    const zoomStep = 0.1
    
    switch (e.key) {
      case 'ArrowUp':
        e.preventDefault()
        setViewport(prev => ({ ...prev, translateY: prev.translateY + step }))
        break
      case 'ArrowDown':
        e.preventDefault()
        setViewport(prev => ({ ...prev, translateY: prev.translateY - step }))
        break
      case 'ArrowLeft':
        e.preventDefault()
        setViewport(prev => ({ ...prev, translateX: prev.translateX + step }))
        break
      case 'ArrowRight':
        e.preventDefault()
        setViewport(prev => ({ ...prev, translateX: prev.translateX - step }))
        break
      case '+':
      case '=':
        e.preventDefault()
        setViewport(prev => ({ ...prev, scale: Math.min(5, prev.scale + zoomStep) }))
        break
      case '-':
        e.preventDefault()
        setViewport(prev => ({ ...prev, scale: Math.max(0.1, prev.scale - zoomStep) }))
        break
      case '0':
        e.preventDefault()
        resetViewport()
        break
      case 'f':
      case 'F':
        if (!isFullscreen) {
          e.preventDefault()
          setIsFullscreen(true)
        }
        break
      case 'Escape':
        if (isFullscreen) {
          e.preventDefault()
          setIsFullscreen(false)
        }
        break
    }
  }, [resetViewport, isFullscreen])

  // Handle touch gestures for mobile
  const [touchState, setTouchState] = useState({
    initialDistance: 0,
    initialScale: 1,
    touches: [] as React.Touch[]
  })

  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    const touches = Array.from(e.touches) as React.Touch[]
    setTouchState(prev => ({ ...prev, touches }))
    
    if (touches.length === 2) {
      // Pinch to zoom
      const distance = Math.hypot(
        touches[0].clientX - touches[1].clientX,
        touches[0].clientY - touches[1].clientY
      )
      setTouchState(prev => ({
        ...prev,
        initialDistance: distance,
        initialScale: viewport.scale
      }))
    } else if (touches.length === 1) {
      // Single touch pan
      setViewport(prev => ({
        ...prev,
        isDragging: true,
        dragStart: { x: touches[0].clientX, y: touches[0].clientY },
        lastPanPoint: { x: prev.translateX, y: prev.translateY }
      }))
    }
  }, [viewport.scale])

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    e.preventDefault()
    const touches = Array.from(e.touches) as React.Touch[]
    
    if (touches.length === 2 && touchState.initialDistance > 0) {
      // Pinch to zoom
      const distance = Math.hypot(
        touches[0].clientX - touches[1].clientX,
        touches[0].clientY - touches[1].clientY
      )
      const scale = Math.max(0.1, Math.min(5, touchState.initialScale * (distance / touchState.initialDistance)))
      setViewport(prev => ({ ...prev, scale }))
    } else if (touches.length === 1 && viewport.isDragging) {
      // Single touch pan
      const deltaX = touches[0].clientX - viewport.dragStart.x
      const deltaY = touches[0].clientY - viewport.dragStart.y
      setViewport(prev => ({
        ...prev,
        translateX: prev.lastPanPoint.x + deltaX,
        translateY: prev.lastPanPoint.y + deltaY
      }))
    }
  }, [touchState.initialDistance, touchState.initialScale, viewport.isDragging, viewport.dragStart, viewport.lastPanPoint])

  const handleTouchEnd = useCallback(() => {
    setViewport(prev => ({ ...prev, isDragging: false }))
    setTouchState({ initialDistance: 0, initialScale: 1, touches: [] as React.Touch[] })
  }, [])

  // Zoom controls
  const zoomIn = () => {
    setViewport(prev => ({ ...prev, scale: Math.min(5, prev.scale * 1.2) }))
  }
  
  const zoomOut = () => {
    setViewport(prev => ({ ...prev, scale: Math.max(0.1, prev.scale / 1.2) }))
  }

  // Setup event listeners
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    container.addEventListener('wheel', handleWheel, { passive: false })
    document.addEventListener('keydown', handleKeyDown)
    
    if (viewport.isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    }

    return () => {
      container.removeEventListener('wheel', handleWheel)
      document.removeEventListener('keydown', handleKeyDown)
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [handleWheel, handleKeyDown, handleMouseMove, handleMouseUp, viewport.isDragging])

  // Mermaid rendering (enhanced from original)
  useEffect(() => {
    let cancelled = false
    
    const render = async () => {
      try {
        setState(prev => ({ ...prev, isLoading: true, error: '', svg: '' }))
        
        const mermaid = (await import('mermaid')).default
        
        mermaid.initialize({ 
          startOnLoad: false, 
          securityLevel: 'strict' as any,
          theme: 'dark',
          maxTextSize: 50000,
          maxEdges: 500
        })

        let processedCode = preprocessMermaid(code || '')
        if (!processedCode) {
          setState(prev => ({ ...prev, error: 'Empty diagram code', isLoading: false }))
          return
        }

        if (processedCode.length > 10000) {
          setState(prev => ({ ...prev, error: 'Diagram code too large', isLoading: false }))
          return
        }

        // Try rendering with fallbacks
        let svg = ''
        let correctionAttempted = false

        try {
          const result = await mermaid.render(`m-${id}`, processedCode)
          svg = result.svg
        } catch (firstError) {
          console.warn('First Mermaid render attempt failed:', firstError)
          
          const sanitizedCode = sanitizeCode(processedCode)
          if (sanitizedCode !== processedCode) {
            try {
              const result = await mermaid.render(`m-${id}-sanitized`, sanitizedCode)
              svg = result.svg
              correctionAttempted = true
            } catch (secondError) {
              console.warn('Second Mermaid render attempt failed:', secondError)
            }
          }
        }

        if (!svg) {
          setState(prev => ({ 
            ...prev, 
            error: 'Unable to render diagram', 
            isLoading: false,
            correctionAttempted: true
          }))
          return
        }

        if (!cancelled) {
          const diagramBounds = analyzeDiagram(svg)
          setBounds(diagramBounds)
          setShowMinimap(diagramBounds.complexity === 'complex')
          
          setState(prev => ({ 
            ...prev, 
            svg, 
            isLoading: false, 
            error: '', 
            correctionAttempted 
          }))
        }

      } catch (e) {
        console.error('Mermaid rendering error:', e)
        if (!cancelled) {
          setState(prev => ({ ...prev, error: 'Rendering system error', isLoading: false }))
        }
      }
    }

    render()
    return () => {
      cancelled = true
    }
  }, [code, id])

  // Reset viewport when diagram loads
  useEffect(() => {
    if (state.svg && bounds.width && bounds.height) {
      resetViewport()
    }
  }, [state.svg, bounds, resetViewport])

  if (state.isLoading) {
    return (
      <div className={`${className} flex items-center justify-center p-4 bg-slate-800/40 border border-slate-600/40 rounded`}>
        <div className="flex items-center gap-2 text-slate-400">
          <div className="animate-spin w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full"></div>
          <span>Rendering diagram...</span>
        </div>
      </div>
    )
  }

  if (state.error) {
    return (
      <div className={className}>
        <div className="border border-red-600/40 bg-red-900/30 rounded p-3 mb-2">
          <div className="flex items-center gap-2 text-red-400 mb-2">
            <AlertCircle size={16} />
            <span className="font-medium">Diagram Rendering Error</span>
            {state.correctionAttempted && <span className="text-xs">(after correction attempt)</span>}
          </div>
          <p className="text-red-300 text-sm">{state.error}</p>
        </div>
        <div className="text-xs text-slate-400 mb-2">Diagram source:</div>
        <pre className="p-3 bg-slate-900 border border-slate-600 rounded text-sm overflow-x-auto text-slate-300">
          {code}
        </pre>
      </div>
    )
  }

  const containerHeight = bounds.complexity === 'complex' ? '500px' : 
                         bounds.complexity === 'medium' ? '400px' : '300px'

  return (
    <>
      <div className={className}>
        {/* Header with title and controls */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {title && <h4 className="font-medium text-text-primary">{title}</h4>}
            {state.correctionAttempted && (
              <div className="flex items-center gap-1 text-green-400 text-xs px-2 py-1 bg-green-900/20 border border-green-600/30 rounded">
                <CheckCircle size={12} />
                <span>Auto-corrected</span>
              </div>
            )}
          </div>
          
          <div className="mermaid-controls">
            <button
              onClick={zoomOut}
              className="mermaid-control-btn"
              title="Zoom out"
              aria-label="Zoom out"
            >
              <ZoomOut size={16} />
            </button>
            <button
              onClick={zoomIn}
              className="mermaid-control-btn"
              title="Zoom in"
              aria-label="Zoom in"
            >
              <ZoomIn size={16} />
            </button>
            <button
              onClick={resetViewport}
              className="mermaid-control-btn"
              title="Reset view"
              aria-label="Reset view"
            >
              <RotateCcw size={16} />
            </button>
            <button
              onClick={() => setIsFullscreen(true)}
              className="mermaid-control-btn"
              title="Fullscreen"
              aria-label="Open in fullscreen"
            >
              <Maximize2 size={16} />
            </button>
            <button
              onClick={() => setShowHelp(!showHelp)}
              className="mermaid-control-btn"
              title="Keyboard shortcuts"
              aria-label="Show keyboard shortcuts"
            >
              <HelpCircle size={16} />
            </button>
          </div>
        </div>

        {/* Interactive diagram container */}
        <div 
          ref={containerRef}
          className={`interactive-mermaid-container relative border border-slate-600/40 rounded bg-slate-900/50 overflow-hidden ${viewport.isDragging ? 'dragging' : ''}`}
          style={{ 
            height: containerHeight,
            cursor: viewport.isDragging ? 'grabbing' : 'grab'
          }}
          onMouseDown={handleMouseDown}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          tabIndex={0}
          role="img"
          aria-label={title || 'Interactive diagram'}
        >
          <div
            ref={svgRef}
            className="mermaid-viewport absolute inset-0"
            style={{
              transform: `translate(${viewport.translateX}px, ${viewport.translateY}px) scale(${viewport.scale})`,
              transformOrigin: '0 0'
            }}
            dangerouslySetInnerHTML={{ __html: state.svg }}
          />
          
          {/* Zoom indicator */}
          <div className="zoom-indicator">
            {Math.round(viewport.scale * 100)}%
          </div>
          
          {/* Help tooltip */}
          {showHelp && (
            <div className="absolute top-12 right-2 bg-slate-800/95 border border-slate-600 rounded-lg p-3 text-xs text-slate-300 backdrop-blur-sm z-20 max-w-xs">
              <div className="font-medium mb-2">Keyboard Shortcuts</div>
              <div className="space-y-1">
                <div><kbd className="bg-slate-700 px-1 rounded">↑↓←→</kbd> Pan</div>
                <div><kbd className="bg-slate-700 px-1 rounded">+/-</kbd> Zoom</div>
                <div><kbd className="bg-slate-700 px-1 rounded">0</kbd> Reset view</div>
                <div><kbd className="bg-slate-700 px-1 rounded">F</kbd> Fullscreen</div>
                <div><kbd className="bg-slate-700 px-1 rounded">Esc</kbd> Exit fullscreen</div>
              </div>
              <div className="mt-2 pt-2 border-t border-slate-600 text-slate-400">
                Mouse: Scroll to zoom, drag to pan
              </div>
            </div>
          )}

          {/* Minimap for complex diagrams */}
          {showMinimap && (
            <div className="minimap">
              <div 
                className="w-full h-full"
                style={{ transform: `scale(${Math.min(128/bounds.width, 96/bounds.height)})`, transformOrigin: '0 0' }}
                dangerouslySetInnerHTML={{ __html: state.svg }}
              />
              {/* Viewport indicator */}
              <div 
                className="minimap-viewport"
                style={{
                  left: `${Math.max(0, Math.min(128, -viewport.translateX / bounds.width * 128))}px`,
                  top: `${Math.max(0, Math.min(96, -viewport.translateY / bounds.height * 96))}px`,
                  width: `${Math.min(128, 128 / viewport.scale)}px`,
                  height: `${Math.min(96, 96 / viewport.scale)}px`
                }}
              />
            </div>
          )}
        </div>
      </div>

      {/* Fullscreen Modal */}
      {isFullscreen && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
          <div className="w-full h-full max-w-7xl max-h-full bg-slate-900 rounded-lg border border-slate-600 flex flex-col">
            {/* Modal header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-600">
              <h3 className="font-medium text-white">{title || 'Diagram'}</h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={zoomOut}
                  className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded transition-colors"
                  title="Zoom out"
                >
                  <ZoomOut size={18} />
                </button>
                <button
                  onClick={zoomIn}
                  className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded transition-colors"
                  title="Zoom in"
                >
                  <ZoomIn size={18} />
                </button>
                <button
                  onClick={resetViewport}
                  className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded transition-colors"
                  title="Reset view"
                >
                  <RotateCcw size={18} />
                </button>
                <button
                  onClick={() => setIsFullscreen(false)}
                  className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded transition-colors"
                  title="Close fullscreen"
                >
                  <X size={18} />
                </button>
              </div>
            </div>
            
            {/* Modal content */}
            <div 
              className="flex-1 relative overflow-hidden"
              onMouseDown={handleMouseDown}
              style={{ cursor: viewport.isDragging ? 'grabbing' : 'grab' }}
            >
              <div
                className="absolute inset-0 transition-transform duration-75 ease-out"
                style={{
                  transform: `translate(${viewport.translateX}px, ${viewport.translateY}px) scale(${viewport.scale})`,
                  transformOrigin: '0 0'
                }}
                dangerouslySetInnerHTML={{ __html: state.svg }}
              />
              
              {/* Fullscreen zoom indicator */}
              <div className="absolute top-4 left-4 px-3 py-2 bg-slate-800/80 text-slate-300 text-sm rounded">
                {Math.round(viewport.scale * 100)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}