'use client'

import React, { useEffect, useId, useState } from 'react'
import { AlertCircle, CheckCircle } from 'lucide-react'
import InteractiveMermaid from './InteractiveMermaid'

interface MermaidProps {
  code: string
  className?: string
  interactive?: boolean
  title?: string
}

interface MermaidState {
  svg: string
  error: string
  isLoading: boolean
  correctionAttempted: boolean
}

export default function Mermaid({ code, className, interactive = true, title }: MermaidProps) {
  // Use interactive version by default
  if (interactive) {
    return <InteractiveMermaid code={code} className={className} title={title} />
  }
  const [state, setState] = useState<MermaidState>({
    svg: '',
    error: '',
    isLoading: true,
    correctionAttempted: false
  })
  const id = useId().replace(/:/g, '-')

  // Preprocess code to enforce delimiter/newline rules while preserving structure
  const preprocessMermaid = (raw: string): string => {
    if (!raw) return ''
    let s = raw
      .replace(/\r\n/g, '\n')
      .replace(/\\n/g, '\n') // convert escaped newlines
      .replace(/\uFF1B/g, ';') // fullwidth semicolon → ascii
      .trim()

    const hasFewNewlines = (s.match(/\n/g) || []).length < 2
    const isVeryLongSingleLine = !s.includes('\n') && s.length > 300

    if (hasFewNewlines || isVeryLongSingleLine) {
      // Split into statements by semicolon delimiters
      const parts = s
        .split(';')
        .map(p => p.trim())
        .filter(p => p.length > 0)
      s = parts.join('\n')
    }

    // Ensure init directive is on its own first line if present
    // Matches %%{init: ...}%% optionally followed by a semicolon
    const initMatch = s.match(/^(.*?%%\{\s*init[\s\S]*?\}%%);?/i)
    if (initMatch) {
      const initBlock = initMatch[1].trim()
      const remainder = s.slice(initMatch[0].length).trim()
      s = initBlock + '\n' + remainder
    }

    // Collapse multiple blank lines, but keep single newlines
    s = s.replace(/\n\s*\n\s*\n+/g, '\n\n')
    return s
  }

  // Sanitization without destroying newlines/structure
  const sanitizeCode = (rawCode: string): string => {
    if (!rawCode) return ''
    return rawCode
      // Replace problematic Unicode characters but preserve newlines
      .replace(/[‑–—]/g, '-')
      .replace(/[\u2018\u2019\u201A]/g, "'")
      .replace(/[\u201C\u201D\u201E]/g, '"')
      .replace(/\u2026/g, '...')
      // Remove parentheses from node labels (simplified approach)
      .replace(/\[([^\]]*)\(([^)]*)\)([^\]]*)\]/g, '[$1$2$3]')
      // Do NOT collapse all whitespace; only normalize triple blank lines
      .replace(/\r\n/g, '\n')
      .replace(/\n\s*\n\s*\n+/g, '\n\n')
      .trim()
  }

  useEffect(() => {
    let cancelled = false
    
    const render = async () => {
      try {
        setState(prev => ({ ...prev, isLoading: true, error: '', svg: '' }))
        
        // Dynamic import to avoid SSR issues
        const mermaid = (await import('mermaid')).default
        
        // Enhanced configuration for better reliability
        mermaid.initialize({ 
          startOnLoad: false, 
          securityLevel: 'strict' as any,
          theme: 'dark',
          maxTextSize: 50000,
          maxEdges: 500
        })

        // Input validation and preprocessing
        let processedCode = preprocessMermaid(code || '')
        if (!processedCode) {
          setState(prev => ({ ...prev, error: 'Empty diagram code', isLoading: false }))
          return
        }

        if (processedCode.length > 10000) {
          setState(prev => ({ ...prev, error: 'Diagram code too large', isLoading: false }))
          return
        }

        // First attempt with original code
        try {
          const { svg } = await mermaid.render(`m-${id}`, processedCode)
          if (!cancelled) {
            setState(prev => ({ 
              ...prev, 
              svg, 
              isLoading: false, 
              error: '', 
              correctionAttempted: false 
            }))
          }
          return
        } catch (firstError) {
          console.warn('First Mermaid render attempt failed:', firstError)
        }

        // Second attempt with sanitized code (preserves newlines)
        const sanitizedCode = sanitizeCode(processedCode)
        if (sanitizedCode !== processedCode) {
          try {
            const { svg } = await mermaid.render(`m-${id}-sanitized`, sanitizedCode)
            if (!cancelled) {
              setState(prev => ({ 
                ...prev, 
                svg, 
                isLoading: false, 
                error: '', 
                correctionAttempted: true 
              }))
            }
            return
          } catch (secondError) {
            console.warn('Second Mermaid render attempt failed:', secondError)
          }
        }

        // Optional fallback: try without init directive if present
        const hasInit = /%%\{\s*init[\s\S]*?\}%%/i.test(processedCode)
        if (hasInit) {
          const withoutInit = processedCode.replace(/\s*^.*?%%\{\s*init[\s\S]*?\}%%\s*/i, '').trim()
          if (withoutInit) {
            try {
              const { svg } = await mermaid.render(`m-${id}-noinit`, withoutInit)
              if (!cancelled) {
                setState(prev => ({ 
                  ...prev, 
                  svg, 
                  isLoading: false, 
                  error: '', 
                  correctionAttempted: true 
                }))
              }
              return
            } catch (thirdError) {
              console.warn('Third Mermaid render attempt (no init) failed:', thirdError)
            }
          }
        }

        // If both attempts failed
        if (!cancelled) {
          setState(prev => ({ 
            ...prev, 
            error: 'Unable to render diagram', 
            isLoading: false,
            correctionAttempted: true
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

  return (
    <div className={className}>
      {state.correctionAttempted && (
        <div className="flex items-center gap-2 text-green-400 text-xs mb-2 p-2 bg-green-900/20 border border-green-600/30 rounded">
          <CheckCircle size={14} />
          <span>Diagram auto-corrected for rendering</span>
        </div>
      )}
      <div className="mermaid-diagram" dangerouslySetInnerHTML={{ __html: state.svg }} />
    </div>
  )
}


